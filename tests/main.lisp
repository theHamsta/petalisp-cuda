(in-package :petalisp-cuda/tests)

(petalisp.test-suite:check-package ':petalisp-cuda)
(petalisp.test-suite:check-package '#:petalisp-cuda.backend)
(petalisp.test-suite:check-package '#:petalisp-cuda.jitexecution)
; NOTE: To run this test file, execute `(asdf:test-system :petalisp-cuda)' in your Lisp.

(defparameter *test-backend* (make-testing-backend))

(deftest test-make-cuda-backend
  (let ((cl-cuda:*show-messages* nil))
   (ok (make-instance 'petalisp-cuda.backend:cuda-backend))))

(deftest test-make-cuda-backend2
  (let ((cl-cuda:*show-messages* nil))
    (with-cuda (0)
      (make-instance 'petalisp-cuda.backend:cuda-backend))))

(deftest jacobi-test
  (with-testing-backend
    (ok (compute (jacobi (aops:rand* 'single-float '(24)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(25)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 5)))))

(deftest with-cuda-backend-raii
  (with-cuda-backend-raii
    (ok (compute (jacobi (aops:rand* 'single-float '(24)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(25)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 5)))))

(deftest jacobi-test-recompile
  (with-testing-backend
    (ok (compute (jacobi (aops:rand* 'single-float '(12)) 0.0 1.0 1)))
    (ok (compute (jacobi (ndarray 1) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 2) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 3) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 3) 0.0 1.0 5)))))

(deftest multiple-arguments
  (with-testing-backend
    (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1)))))

(deftest mixed-calculations
  (= (compute
      (α #'1+ (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1)))))
    (let ((petalisp-cuda.backend:*transfer-back-to-lisp* t))
      (compute
        (α #'1+ (with-cuda-backend
                  (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1)))))))))

(deftest mixed-culculations-no-explict-transfer
  (let ((petalisp-cuda.backend:*transfer-back-to-lisp* nil))
   (compute
    (α #'1+ (with-cuda-backend
             (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1))))))))

(deftest resume-cuda-calculations
  (with-cuda-backend
    (compute
      (α #'1+ (with-cuda-backend
                (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1))))))))

(deftest rbgs-test-fixed-size
  (with-testing-backend
    (ok (compute (rbgs (aops:rand* 'single-float '(4)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(5)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'double-float '(6)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(7)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(25)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(24)) 0.0 1.0 2)))))

(deftest rbgs-test
  (with-testing-backend
    (ok (compute (rbgs (aops:rand* 'single-float '(25)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(26)) 0.0 1.0 1)))
    (ok (compute (rbgs (aops:rand* 'single-float '(27)) 0.0 1.0 2)))
    (ok (compute (rbgs (ndarray 1) 0.0 1.0 2)))
    (ok (compute (rbgs (ndarray 2) 0.0 1.0 2)))
    (ok (compute (rbgs (ndarray 3) 0.0 1.0 2)))
    (ok (compute (rbgs (ndarray 3) 0.0 1.0 5)))))

(deftest indices-test
  (with-testing-backend
    (compute (array-indices #(5 6 7)))
    (let ((a (make-array '(2 3 4))))
      (compute (array-indices a 1))
      (compute (α #'+
                  (array-indices a 0)
                  (array-indices a 1)
                  (array-indices a 2))))))

(deftest lazy-map-test
  (with-testing-backend
    (compute
      (α #'+ 2 3))
    (compute
      (α #'+ #(2 3 4) #(5 4 3)))
    (compute
      (α #'+ #2A((1 2) (3 4)) #2A((4 3) (2 1))))
    (compute
      (α #'floor #(1 2.5 1/2) 2))))

(deftest sqrt-test
  (with-testing-backend
    (compute
      (α #'sqrt 4))))

(deftest reshape-test
  (with-testing-backend
    (compute (reshape 4 (~ 5)))
    (compute (reshape #(1 2 3) (τ (i) ((- i)))) #(3 2 1))
    (compute (reshape #(1 2 3 4) (~ 1 3)))
    (compute (reshape (shape-indices (~ 1 10)) (~ 3 ~ 3)))
    (compute (reshape #2A((1 2) (3 4)) (τ (i j) (j i))))
    (compute (reshape #(1 2 3 4) (~ 1 3) (~ 0 2 ~ 0 2)))
    (compute
      (fuse*
        (reshape #2A((1 2 3) (4 5 6)) (τ (i j) ((+ 2 i) (+ 3 j))))
        (reshape 9 (τ () (3 4)))))))

(declaim (optimize (debug 3)))
(defun max* (x)
  (β (lambda (lv li rv ri)
       (if (> lv rv)
           (values lv li)
           (values rv ri)))
     x (array-indices x)))

(deftest multi-value-floor
  (with-testing-backend
  (compute (nth-value 1 (max* #(2 4 2 1 2 1))))
  (compute (nth-value 1 (α* 2 #'floor #(2 4 2 1 2 1.1) 0.5)))
  (multiple-value-call #'compute (max* #(2 2 3 2 4 1 2 1)))
  (multiple-value-call #'compute (α* 2 #'floor #(2 4 2 1 2 1.1) 0.5))
  (compute (α #'+ (α #'floor #(2 2 .2 3 2 2 2 3 3) 0.5)))))

(deftest linear-algebra-test
  (with-testing-backend
    (ok (compute (eye 3)))
    (ok (compute (petalisp.examples.linear-algebra:dot #(1 2 3) #(4 5 6))))
    (ok (compute (norm #(1 2 3))))
    (ok (compute (max* #(2 4 1 2 1))))
    (ok (compute (nth-value 1 (max* #(2 4 1 2 1)))))
    (ok (multiple-value-call #'compute (max* #(2 4 1 2 1))))
    (loop repeat 10 do
          (ok (let* ((a (generate-matrix))
                     (b (compute (transpose a))))
                (compute (matmul a b)))))))

(deftest lu
  (if t
      (skip "conversion T -> float causes this to fail")
      (mapcar (lambda (matrix)
                (format t "Lisp: ~A~%CUDA: ~A~%"
                        (multiple-value-bind (P L R) (lu matrix)
                          (multiple-value-list (compute P L R)))
                        (with-cuda-backend-raii
                          (multiple-value-bind (P L R) (lu matrix)
                            (multiple-value-list (compute P L R))))))
              '(#2A((42))
                #2A((1. 1.) (1. 2.))
                #2A((1 3 5) (2 4 7) (1 1 0))
                #2A((2 3 5) (6 10 17) (8 14 28))
                #2A((1 2 3) (4 5 6) (7 8 0))
                #2A(( 1 -1  1 -1  5)
                    (-1  1 -1  4 -1)
                    ( 1 -1  3 -1  1)
                    (-1  2 -1  1 -1)
                    ( 1 -1  1 -1  1))))))

(deftest pivot-and-value
  (if t
      (skip "conversion T -> float causes this to fail")
      (with-testing-backend
        (pivot-and-value #2A((1. 1.) (1. 2.)) 0))))

(deftest num-values
    (ok (= 1 (petalisp-cuda.jitexecution::num-values '(values 1))))
    (ok (= 2 (petalisp-cuda.jitexecution::num-values '(values 2 3))))
    (ok (= 2 (petalisp-cuda.jitexecution::num-values '(defun (a)
                                                         (foo (values 2 3))))))
    (ok (= 1 (petalisp-cuda.jitexecution::num-values '(defun (a)
                                                         (foo 2))))))

(deftest nth-value-lambda
  (ok (= 1 (petalisp-cuda.jitexecution::num-values '(values 1))))
  (ok (= 2 (petalisp-cuda.jitexecution::nth-value-lambda 0 '(values 2 3))))
  (ok (= 3 (petalisp-cuda.jitexecution::nth-value-lambda 1 '(values 2 3))))
  (ok (equal '(defun (a) (foo 2))
             (petalisp-cuda.jitexecution::nth-value-lambda 0 '(defun (a)
                                                                  (foo (values 2 3))))))
  (ok (equal '(defun (a) (foo 3))
             (petalisp-cuda.jitexecution::nth-value-lambda 1 '(defun (a)
                                                                  (foo (values 2 3)))))))
(deftest analyze-multiple-value-lambda
  (ok (equal 
        '((DEFUN (A) (FOO 2)))
        (petalisp-cuda.jitexecution::analyze-multiple-value-lambda '(defun (a)
                                                                        (foo 2)))))
  (ok (equal 
        '((DEFUN (A) (FOO 2)) (DEFUN (A) (FOO 3)))
        (petalisp-cuda.jitexecution::analyze-multiple-value-lambda '(defun (a)
                                                                        (foo (values 2 3)))))))

(deftest test-type-conversion
  (with-testing-backend
    (compute (α #'coerce #(1 2 3) 'double-float))
    (compute (α #'coerce #(1 2 3) 'single-float))
    (compute (α #'coerce (aops:rand* 'double-float '(20 20)) 'single-float))
    (compute (α #'truncate (aops:rand* 'double-float '(20 20))))
    (compute (α #'round (aops:rand* 'double-float '(20 20))))
    (compute (α #'coerce (aops:rand* 'single-float '(20 20)) 'double-float))))


(deftest v-cycle-test
  (with-testing-backend
    (compute (v-cycle (reshape 1.0 (~ 5 ~ 5)) 0.0 1.0 2 1))
    (compute (v-cycle (reshape 1.0 (~ 9 ~ 9)) 0.0 1.0 2 1))
    (compute (v-cycle (reshape 1.0 (~ 17 ~ 17)) 0.0 1.0 2 1))
    (compute (v-cycle (reshape 1.0 (~ 33 ~ 33)) 0.0 1.0 2 1))
    (compute (v-cycle (reshape 1.0 (~ 65 ~ 65)) 0.0 1.0 3 3))))

(deftest reduction-test
  (with-testing-backend
    (compute
      (β #'+ #(1 2 3)))
    (compute
      (β #'+ #2A((1 2 3) (6 5 4))))
    ;;; lambdas with multiple return values do not work currently
    (compute
      (β (lambda (lmax lmin rmax rmin)
           (values (max lmax rmax) (min lmin rmin)))
         #(+1 -1 +2 -2 +3 -3)
         #(+1 -1 +2 -2 +3 -3)))
    (compute
      (β (lambda (a b) (values a b)) #(3 2 1))
      (β (lambda (a b) (values b a)) #(3 2 1)))))

(deftest network-test
  (with-testing-backend
  (let* ((shape (~ 10))
         (x1 (make-instance 'parameter :shape shape :element-type 'double-float))
         (x2 (make-instance 'parameter :shape shape :element-type 'double-float))
         (v1 (α #'+
                (α #'coerce (α #'log x1) 'double-float)
                (α #'* x1 x2)
                (α #'sin x2)))
         (network
           (make-network v1))
         (g1 (make-instance 'parameter
               :shape (array-shape v1)
               :element-type (element-type v1)))
         (gradient-fn (differentiator (list v1) (list g1)))
         (gradient-network
           (make-network
            (funcall gradient-fn x1)
            (funcall gradient-fn x2))))
    (call-network network x1 5d0 x2 1d0)
    (call-network gradient-network x1 1d0 x2 1d0 g1 1d0))))

;(deftest test-descriptor
  ;(unless petalisp-cuda.cudalibs::*cudnn-found*
    ;(skip "No cudnn!"))
  ;(with-cuda (0)
    ;(petalisp-cuda.cudalibs::cudnn-create-tensor-descriptor
      ;(make-cuda-array '(10 20) 'float))))

;(deftest test-descriptor2
  ;(unless petalisp-cuda.cudalibs::*cudnn-found*
    ;(skip "No cudnn!"))
  ;(with-cuda (0)
    ;(progn
      ;(petalisp-cuda:use-cuda-backend)
      ;(let ((a (make-cuda-array '(10 20) 'float))
            ;(b (make-cuda-array '(10 1) 'float)))
        ;(petalisp-cuda.cudalibs::cudnn-reduce-array a b #'+)))))

(deftest test-petalisp.test-suite
  (with-testing-backend
    (mapcar (lambda (test) (let ((test-name (format nil "~A" test)))
                                      (if (find test-name '("LINEAR-ALGEBRA-TEST") :test #'equalp)
                                      (skip test-name)
                                      (progn
                                        (testing test-name) 
                                        (petalisp.test-suite:run-tests test)))))
            (petalisp.test-suite::all-tests))))

(deftest reclaim-memory
  (with-cuda-backend (compute (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))
  (reclaim-cuda-memory)
  (let* ((mem-pool (petalisp-cuda.backend::cuda-memory-pool petalisp-cuda.backend::*cuda-backend*))
             (all-blocks (hash-set:hs-to-list (hash-set:hs-map #'cl-cuda::memory-block-device-ptr (allocated-cuda-arrays mem-pool))))
             (available-blocks (mapcar #'cl-cuda::memory-block-device-ptr (apply #'concatenate `(list ,@(alexandria:hash-table-values (array-table mem-pool)))))))
        (ok (= (length available-blocks) (length all-blocks)))))

