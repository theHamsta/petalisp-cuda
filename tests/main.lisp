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

(deftest test-make-cuda-array
  (let ((cl-cuda:*show-messages* nil))
   (with-cuda (0)
    (make-cuda-array '(10 20) 'float))))

(deftest jacobi-test
  (with-testing-backend
    (ok (compute (jacobi (aops:rand* 'single-float '(24)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 2)))
    (ok (compute (jacobi (aops:rand* 'single-float '(24 26 30)) 0.0 1.0 5)))))

(deftest jacobi-test-recompile
  (with-testing-backend
    (ok (compute (jacobi (ndarray 1) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 2) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 3) 0.0 1.0 2)))
    (ok (compute (jacobi (ndarray 3) 0.0 1.0 5)))))

(deftest multiple-arguments
  (with-testing-backend
    (compute 1 2 3 4 5 6 7 8 9 (α #'+ 5 5) (β #'+ #(1 2 3 4 1)))))

(deftest mixed-culculations
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

(deftest rbgs-test
  (with-testing-backend
    (compute (rbgs (ndarray 1) 0.0 1.0 2))
    (compute (rbgs (ndarray 2) 0.0 1.0 2))
    (compute (rbgs (ndarray 3) 0.0 1.0 2))
    (compute (rbgs (ndarray 3) 0.0 1.0 5))))

(deftest v-cycle-test
  (compute (v-cycle (reshape 1.0 (~ 5 ~ 5)) 0.0 1.0 2 1))
  (compute (v-cycle (reshape 1.0 (~ 9 ~ 9)) 0.0 1.0 2 1))
  (compute (v-cycle (reshape 1.0 (~ 17 ~ 17)) 0.0 1.0 2 1))
  #+(or)
  (compute (v-cycle (reshape 1.0 (~ 33 ~ 33)) 0.0 1.0 2 1))
  #+(or)
  (compute (v-cycle (reshape 1.0 (~ 65 ~ 65)) 0.0 1.0 3 3)))

(defmethod approximately-equal ((a t) (b single-float))
  (< (abs (- a b)) (* 100 single-float-epsilon)))
(defmethod approximately-equal ((a single-float) (b t))
  (< (abs (- a b)) (* 100 single-float-epsilon)))
(defmethod approximately-equal ((a t) (b double-float))
  (< (abs (- a b)) (* 100 double-float-epsilon)))
(defmethod approximately-equal ((a double-float) (b t))
  (< (abs (- a b)) (* 100 double-float-epsilon)))

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

(deftest test-mem-roundtrip
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((foo (aops:rand* 'single-float '(20 9)))
               (a (make-cuda-array foo 'float))
               (b (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp a)))
          (loop for i below (reduce #'* (array-dimensions foo))
                do (progn
                     (assert (equal (row-major-aref foo i) (row-major-aref b i))))))))))


(deftest test-petalisp.test-suite
  (with-testing-backend
    (mapcar (lambda (test) (testing (format nil "~A" test) (petalisp.test-suite:run-tests test))) (petalisp.test-suite::all-tests))))


