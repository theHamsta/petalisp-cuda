(in-package :petalisp-cuda/tests)

(petalisp.test-suite:check-package ':petalisp-cuda)
(petalisp.test-suite:check-package ':petalisp-cuda.backend)
(petalisp.test-suite:check-package ':petalisp-cuda.jit-execution)

(defvar *test-backend* nil)

(deftest test-make-cuda-backend
  (let ((cl-cuda:*show-messages* nil))
   (ok (make-instance 'petalisp-cuda.backend:cuda-backend))))

(deftest test-make-cuda-backend2
  (let ((cl-cuda:*show-messages* nil))
    (with-cuda (0)
      (make-instance 'petalisp-cuda.backend:cuda-backend))))

(deftest jacobi-test-cuda-only
  (with-cuda-backend
    (ok (compute (jacobi (aops:rand* 'single-float '(24)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(25)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 5)))))

(deftest jacobi-prepare-cuda-only
  (let ((petalisp-cuda.options:*transfer-back-to-lisp* nil))
    (with-cuda-backend
      (ok (cuda-immediate-p (prepare (jacobi (aops:rand* 'single-float '(24)) 0.0 1.0 2))))
      (ok (cuda-immediate-p (prepare (jacobi (aops:rand* 'single-float '(25)) 0.0 1.0 2))))
      (ok (cuda-immediate-p (prepare (jacobi (aops:rand* 'single-float '(26)) 0.0 1.0 2))))
      (ok (cuda-immediate-p (prepare (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))))))

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
    (compute 1 2 3 4 5 6 7 8 9 (lazy #'+ 5 5) (lazy-reduce #'+ #(1 2 3 4 1)))))

(deftest can-calculate-identity
  (with-cuda-backend
    (compute 1)))

(deftest double-compute
  (let ((petalisp-cuda.options:*transfer-back-to-lisp* nil))
    (with-cuda-backend
      (prepare (lazy #'+ (prepare (lazy #'+ #(1 2 3 4 1) 5)) 2)))))

(deftest mixed-calculations
  (ok
    (= (compute
         (lazy #'1+ (compute 1 2 3 4 5 6 7 8 9 (lazy #'+ 5 5) (lazy-reduce #'+ #(1 2 3 4 1)))))
       (let ((petalisp-cuda.options:*transfer-back-to-lisp* nil))
         (compute
           (lazy #'1+ (with-cuda-backend
                     (prepare 1 2 3 4 5 6 7 8 9 (lazy #'+ 5 5) (lazy-reduce #'+ #(1 2 3 4 1))))))))))

(deftest mixed-calculations-no-explict-transfer
  (let ((petalisp-cuda.options:*transfer-back-to-lisp* nil))
   (compute
    (lazy #'1+ (with-cuda-backend
             (prepare 1 2 3 4 5 6 7 8 9 (lazy #'+ #(1 2 3 4 1) 5) (lazy-reduce #'+ #(1 2 3 4 1))))))))

(deftest resume-cuda-calculations
  (with-cuda-backend
    (compute
      (lazy #'1+ (with-cuda-backend
                (compute 1 2 3 4 5 6 7 8 9 (lazy #'+ 5 5) (lazy-reduce #'+ #(1 2 3 4 1))))))))

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
    (compute (lazy-array-indices #(5 6 7)))
    (let ((a (make-array '(2 3 4))))
      (compute (lazy-array-indices a 1))
      (compute (lazy #'+
                  (lazy-array-indices a 0)
                  (lazy-array-indices a 1)
                  (lazy-array-indices a 2))))))

(deftest lazy-map-test
  (with-testing-backend
    (compute
      (lazy #'+ 2 3))
    (compute
      (lazy #'+ #(2 3 4) #(5 4 3)))
    (compute
      (lazy #'+ #2A((1 2) (3 4)) #2A((4 3) (2 1))))
    (compute
      (lazy #'floor #(1 2.5 1/2) 2))))

(deftest sqrt-test
  (with-testing-backend
    (compute
      (lazy #'sqrt 4))))

(deftest reshape-test
  (with-testing-backend
    (compute (lazy-reshape 4 (~ 5)))
    (compute (lazy-reshape #(1 2 3) (transform i to (- i))) #(3 2 1))
    (compute (lazy-reshape #(1 2 3 4) (~ 1 3)))
    (compute (lazy-reshape (lazy-shape-indices (~ 1 10)) (~ 3 ~ 3)))
    (compute (lazy-reshape #2A((1 2) (3 4)) (transform i j to j i)))
    (compute (lazy-reshape #(1 2 3 4) (~ 1 3) (~ 0 2 ~ 0 2)))
    (compute
      (lazy-overwrite
        (lazy-reshape #2A((1 2 3) (4 5 6)) (transform i j to (+ 2 i) (+ 3 j)))
        (lazy-reshape 9 (transform to 3 4))))))

(declaim (optimize (debug 3)))
(defun max* (x)
  (lazy-reduce (lambda (lv li rv ri)
                 (if (> lv rv)
                     (values lv li)
                     (values rv ri)))
               x (lazy-array-indices x)))

(deftest multi-value-floor
  (with-testing-backend
    (compute (nth-value 1 (max* #(2 4 2 1 2 1))))
    (compute (nth-value 1 (lazy-multiple-value #'floor 2 #(2 4 2 1 2 1.1) 0.5)))
    (multiple-value-call #'compute (max* #(2 2 3 2 4 1 2 1)))
    (multiple-value-call #'compute (lazy-multiple-value #'floor 2 #(2 4 2 1 2 1.1) 0.5))
    (compute (lazy #'+ (lazy #'floor #(2 2 .2 3 2 2 2 3 3) 0.5)))))

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
    (ok (= 1 (petalisp-cuda.jit-execution::num-values '(values 1))))
    (ok (= 2 (petalisp-cuda.jit-execution::num-values '(values 2 3))))
    (ok (= 2 (petalisp-cuda.jit-execution::num-values '(defun (a)
                                                         (foo (values 2 3))))))
    (ok (= 1 (petalisp-cuda.jit-execution::num-values '(defun (a)
                                                         (foo 2))))))

(deftest nth-value-lambda
  (ok (= 1 (petalisp-cuda.jit-execution::num-values '(values 1))))
  (ok (= 2 (petalisp-cuda.jit-execution::nth-value-lambda 0 '(values 2 3))))
  (ok (= 3 (petalisp-cuda.jit-execution::nth-value-lambda 1 '(values 2 3))))
  (ok (equal '(defun (a) (foo 2))
             (petalisp-cuda.jit-execution::nth-value-lambda 0 '(defun (a)
                                                                  (foo (values 2 3))))))
  (ok (equal '(defun (a) (foo 3))
             (petalisp-cuda.jit-execution::nth-value-lambda 1 '(defun (a)
                                                                  (foo (values 2 3)))))))
(deftest analyze-multiple-value-lambda
  (ok (equal 
        '((DEFUN (A) (FOO 2)))
        (petalisp-cuda.jit-execution::analyze-multiple-value-lambda '(defun (a)
                                                                        (foo 2)))))
  (ok (equal 
        '((DEFUN (A) (FOO 2)) (DEFUN (A) (FOO 3)))
        (petalisp-cuda.jit-execution::analyze-multiple-value-lambda '(defun (a)
                                                                        (foo (values 2 3)))))))

(deftest test-type-conversion
  (with-testing-backend
    (compute (lazy #'coerce #(1 2 3) 'double-float))
    (compute (lazy #'coerce #(1 2 3) 'single-float))
    (compute (lazy #'coerce (aops:rand* 'double-float '(20 20)) 'single-float))
    (compute (lazy #'truncate (aops:rand* 'double-float '(20 20))))
    (compute (lazy #'round (aops:rand* 'double-float '(20 20))))
    (compute (lazy #'coerce (aops:rand* 'single-float '(20 20)) 'double-float))))


(deftest v-cycle-test
  (with-testing-backend
    (compute (v-cycle (lazy-reshape 1.0 (~ 5 ~ 5)) 0.0 1.0 2 1))
    (compute (v-cycle (lazy-reshape 1.0 (~ 9 ~ 9)) 0.0 1.0 2 1))
    (compute (v-cycle (lazy-reshape 1.0 (~ 17 ~ 17)) 0.0 1.0 2 1))
    (compute (v-cycle (lazy-reshape 1.0 (~ 33 ~ 33)) 0.0 1.0 2 1))
    (compute (v-cycle (lazy-reshape 1.0 (~ 65 ~ 65)) 0.0 1.0 3 3))))

(deftest reduction-test
  (with-testing-backend
    (compute
      (lazy-reduce #'+ #(1 2 3)))
    (compute
      (lazy-reduce #'+ #2A((1 2 3) (6 5 4))))
    ;;; lambdas with multiple return values do not work currently
    (compute
      (lazy-reduce (lambda (lmax lmin rmax rmin)
           (values (max lmax rmax) (min lmin rmin)))
         #(+1 -1 +2 -2 +3 -3)
         #(+1 -1 +2 -2 +3 -3)))
    (compute
      (lazy-reduce (lambda (a b) (values a b)) #(3 2 1))
      (lazy-reduce (lambda (a b) (values b a)) #(3 2 1)))))

(deftest network-test
  (with-testing-backend
  (let* ((shape (~ 10))
         (x1 (make-instance 'parameter :shape shape :element-type 'double-float))
         (x2 (make-instance 'parameter :shape shape :element-type 'double-float))
         (v1 (lazy #'+
                (lazy #'coerce (lazy #'log x1) 'double-float)
                (lazy #'* x1 x2)
                (lazy #'sin x2)))
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

(device-host-function foo (x)
  (+ x 1))

(deftest test-device-function
  (with-testing-backend
    (compute (lazy #'foo 1))))

(defun bar (x)
  (* x 2))

(device-function bar (x)
  (* x 2))

(deftest test-device-function
  (with-testing-backend
    (compute (lazy #'bar 1))))

(defun baz (x)
  (* x 2))

(device-function baz (x)
  (* x 4))

(deftest test-device-function-contradicting-definitions
  (signals
      (with-testing-backend
        (compute (lazy #'baz 1)))))

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

