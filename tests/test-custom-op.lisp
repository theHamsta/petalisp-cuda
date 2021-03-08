(in-package :petalisp-cuda/tests)

(deftest test-lazy-convolution-compute
  (let ((a (lazy #'+ #4A((((2 2) (2 2)))) #4A((((2 2) (2 2))))))
        (b #4A((((2))))))
    (with-cuda-backend
      (compute
      (lazy #'+ a
           (lazy-reduce #'max
              (lazy-convolution
                a
                b
                :algorithm :cudnn-convolution-fwd-algo-implicit-gemm)))))))

(deftest test-lazy-reduction
  (with-cuda-backend
    (compute
      (lazy #'+ 2 (lazy-reduction (lazy #'+ 2 #2A((2 3) (4 5))) (~) #'+)))))

(deftest test-lazy-reduction-diff
  (let* ((x (petalisp.examples.machine-learning::make-trainable-parameter -2.0))
         (loss (lazy-reduction (lazy-reshape (lazy #'abs x) (~ 2 ~ 3)) (~) #'+))
         (inference (make-network loss))
         (gradient (make-network (funcall (differentiator (list loss) (list loss)) x) loss)))
  (with-cuda-backend
   (list (call-network inference x (petalisp.examples.machine-learning::trainable-parameter-value x))
         (call-network gradient x (petalisp.examples.machine-learning::trainable-parameter-value x))))))

(deftest test-lazy-convolution
  (let ((a (lazy #'+ #4A((((2 2) (2 2)))) #4A((((2 2) (2 2))))))
        (b #4A((((2))))))
    (with-cuda-backend
      (compute
           (lazy #'+ a
             (lazy-reduce #'max
                (lazy-convolution
                  a
                  b
                  :algorithm :cudnn-convolution-fwd-algo-implicit-gemm)))
          (lazy #'+ 2 (lazy-convolution #4A((((2 2)
                                            (2 2))))
                                     #4A((((2))))))))))


(deftest test-lazy-convolution-diff-filter
  (let* ((x (aops:rand* 'single-float '(20 30)))
         (w (petalisp.examples.machine-learning::make-trainable-parameter 2.0))
         (loss (lazy-convolution (lazy-reshape x (~ 1 ~ 1 ~ 20 ~ 30)) (lazy-reshape w (~ 1 ~ 1 ~ 2 ~ 2))))
         (inference (make-network loss))
         (gradient (make-network (funcall (differentiator (list loss) (list loss)) w) loss)))
  (with-cuda-backend
   (list (call-network inference w 3.0)
         (call-network gradient w 3.0)))))

(deftest test-lazy-convolution-diff-both
  (let* ((x (petalisp.examples.machine-learning::make-trainable-parameter (aops:rand* 'single-float '(20 30))))
         (w (petalisp.examples.machine-learning::make-trainable-parameter 2.0))
         (loss (lazy-convolution (lazy-reshape x (~ 1 ~ 1 ~ 20 ~ 30)) (lazy-reshape w (~ 1 ~ 1 ~ 2 ~ 2))))
         (inference (make-network loss))
         (gradient (make-network (funcall (differentiator (list loss) (list loss)) w)
                                 (funcall (differentiator (list loss) (list loss)) x)
                                 loss)))
    (with-cuda-backend
      ;(petalisp.graphviz:view (network-outputs inference))
      ;(petalisp.graphviz:view (network-outputs gradient))
      (list (call-network inference w 3.0 x (petalisp.examples.machine-learning::trainable-parameter-value x))
            (call-network gradient w 3.0 x (petalisp.examples.machine-learning::trainable-parameter-value x))))))

(deftest test-lazy-convolution-diff-data
  (let* ((x (petalisp.examples.machine-learning::make-trainable-parameter (aops:rand* 'single-float '(20 30))))
         (w 2.0)
         (loss (lazy-convolution (lazy-reshape x (~ 4 ~ 10 ~ 20 ~ 30)) (lazy-reshape w (~ 2 ~ 10 ~ 2 ~ 2))))
         (inference (make-network loss))
         (gradient (make-network (funcall (differentiator (list loss) (list loss)) x) loss)))
    (with-cuda-backend
      (list 
        (call-network inference x (petalisp.examples.machine-learning::trainable-parameter-value x))
        (call-network gradient x (petalisp.examples.machine-learning::trainable-parameter-value x))))))

(deftest unnormalizing-transformation
  (ok
    (equalp
      (~ 1 ~ 7 ~ 4)
      (let ((normalized (transform-shape
                          (~ 1 ~ 2 9 ~ 3 7)
                          (normalizing-transformation (~ 1 ~ 2 9 ~ 3 7)))))
        (transform-shape
          normalized
          (petalisp-cuda.cudnn-ops::unnormalizing-transformation
            normalized
            (~ 1 ~ 2 9 ~ 3 7))))))
  (ok
    (equalp
      (~ 1 ~ 7 ~ 4)
      (let ((collapsed (transform-shape
                          (~ 1 ~ 2 9 ~ 3 7)
                          (collapsing-transformation (~ 1 ~ 2 9 ~ 3 7)))))
        (transform-shape
          collapsed
          (petalisp-cuda.cudnn-ops::unnormalizing-transformation
            collapsed
            (~ 1 ~ 2 9 ~ 3 7)))))))

(deftest unnormalizing-transformation
  (let* ((cl-cuda:*show-messages* nil)
         (normalized (transform-shape
                       (~ 1 ~ 2 9 ~ 3 7)
                       (normalizing-transformation (~ 1 ~ 2 9 ~ 3 7))))
         (cu-array (make-cuda-array '(7 4) 'float)))
    (ok
      (petalisp-cuda.stride-tricks:transform-cuda-array
        cu-array
        (petalisp-cuda.cudnn-ops::unnormalizing-transformation
          normalized
          (~ 1 ~ 2 9 ~ 3 7))))
    (free-cuda-array cu-array)))
