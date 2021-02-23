(in-package :petalisp-cuda/tests)

(deftest test-descriptor
  (if (not petalisp-cuda.cudalibs::*cudnn-found*)
      (skip "No cudnn!")
      (with-cuda-backend
        (petalisp-cuda.cudnn-handler::cudnn-create-tensor-descriptor
          (make-cuda-array '(10 20) 'float)
          (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)))))

(deftest test-reduction
  (if (not petalisp-cuda.cudalibs::*cudnn-found*)
      (skip "No cudnn!")
      (approximately-equal
        (with-cuda-backend
          (compute (let ((petalisp-cuda.options:*transfer-back-to-lisp* t))
                     (let ((a (make-cuda-array '(10 20) 'float))
                           (b (make-cuda-array '(10 1) 'float)))
                       (petalisp-cuda.cudnn-handler::cudnn-reduce-array a b #'+ (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)))
                     (let ((a (make-cuda-array #2A ((2 3 4)
                                                    (1 2 3))
                                               'float))
                           (b (make-cuda-array '(2 1) 'float)))
                       (petalisp-cuda.cudnn-handler::cudnn-reduce-array a b #'+ (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
                       b))))
        #2a ((9) (6)))))

(deftest test-convolution
  (if (not petalisp-cuda.cudalibs::*cudnn-found*)
      (skip "No cudnn!")
      (progn
        ;(with-cuda-backend
          ;(let ((x (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
                ;(y (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
                ;(w (make-cuda-array #4A((((1 2 2) (2 4 2) (2 3 2)))) 'float)))
            ;(petalisp-cuda.cudnn-handler::cudnn-convolution x w y (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
            ;y)
          ;(let ((x (make-cuda-array (aops:rand* 'float '(1 1 3 3 3)) 'float))
                ;(y (make-cuda-array (aops:rand* 'float '(1 1 3 3 3)) 'float))
                ;(w (make-cuda-array #5A(((((1 2 2) (2 4 2) (2 3 2)) ((1 2 2) (2 4 2) (2 3 2)) ((1 2 2) (2 4 2) (2 3 2))))) 'float)))
            ;(petalisp-cuda.cudnn-handler::cudnn-convolution x w y (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
            ;y)
          ;)
        (with-cuda-backend
          (let ((x (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
                (y (make-cuda-array (aops:rand* 'float '(1 1 13 14)) 'float))
                (w (make-cuda-array #4A((((1 2) (3 4)))) 'float))
                (petalisp-cuda.options:*cudnn-autotune* t))
            (petalisp-cuda.cudnn-handler::cudnn-convolution x
                                                            w
                                                            y
                                                            (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)
                                                            :algorithm :cudnn-convolution-fwd-algo-implicit-gemm)
            (petalisp-cuda.cudnn-handler::cudnn-convolution x
                                                            w
                                                            y
                                                            (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)
                                                            :algorithm :cudnn-convolution-fwd-algo-implicit-gemm)
            y))
        ;(with-cuda-backend
          ;(let ((x (make-cuda-array (aops:rand* 'float '(1 1 9 9)) 'float))
                ;(y (make-cuda-array (aops:rand* 'float '(1 1 3 2)) 'float))
                ;(w (make-cuda-array #4A((((1 2 2) (2 4 2) (2 3 2)))) 'float)))
            ;(petalisp-cuda.cudnn-handler::cudnn-convolution x w y (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)
                                                            ;:dilations '(1 3)
                                                            ;:filter-strides '(3 3))
            ;y))
        )))
