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

(deftest linear-algebra-test
  (with-testing-backend
  (compute (petalisp.examples.linear-algebra:dot #(1 2 3) #(4 5 6)))
  (compute (norm #(1 2 3)))
  ;(compute (max* #(2 4 1 2 1)))
  ;(compute (nth-value 1 (max* #(2 4 1 2 1))))
  ;(multiple-value-call #'compute (max* #(2 4 1 2 1)))
  (loop repeat 10 do
    (let* ((a (generate-matrix))
           (b (compute (transpose a))))
      (compute (matmul a b))))

  (let ((invertible-matrices
          '(#2A((42))
            #2A((1 1) (1 2))
            #2A((1 3 5) (2 4 7) (1 1 0))
            #2A((2 3 5) (6 10 17) (8 14 28))
            #2A((1 2 3) (4 5 6) (7 8 0))
            #2A(( 1 -1  1 -1  5)
                (-1  1 -1  4 -1)
                ( 1 -1  3 -1  1)
                (-1  2 -1  1 -1)
                ( 1 -1  1 -1  1)))))
    (loop for matrix in invertible-matrices do
      (multiple-value-bind (P L R) (lu matrix)
        (compute
         (matmul P (matmul L R))))))))

(deftest v-cycle-test
  (compute (v-cycle (reshape 1.0 (~ 5 ~ 5)) 0.0 1.0 2 1))
  (compute (v-cycle (reshape 1.0 (~ 9 ~ 9)) 0.0 1.0 2 1))
  (compute (v-cycle (reshape 1.0 (~ 17 ~ 17)) 0.0 1.0 2 1))
  #+(or)
  (compute (v-cycle (reshape 1.0 (~ 33 ~ 33)) 0.0 1.0 2 1))
  #+(or)
  (compute (v-cycle (reshape 1.0 (~ 65 ~ 65)) 0.0 1.0 3 3)))

(defmethod approximately-equal ((a t) (b single-float))
  (< (abs (- a b)) (* 64 single-float-epsilon)))
(defmethod approximately-equal ((a single-float) (b t))
  (< (abs (- a b)) (* 64 single-float-epsilon)))
(defmethod approximately-equal ((a t) (b double-float))
  (< (abs (- a b)) (* 64 double-float-epsilon)))
(defmethod approximately-equal ((a double-float) (b t))
  (< (abs (- a b)) (* 64 double-float-epsilon)))

(deftest reduction-test
  (with-testing-backend
  (compute
   (β #'+ #(1 2 3)))
  (compute
   (β #'+ #2A((1 2 3) (6 5 4))))
  ;;; lambdas with multiple return values do not work currently
  ;(compute
   ;(β (lambda (lmax lmin rmax rmin)
        ;(values (max lmax rmax) (min lmin rmin)))
      ;#(+1 -1 +2 -2 +3 -3)
      ;#(+1 -1 +2 -2 +3 -3)))
  ;(compute
   ;(β (lambda (a b) (values a b)) #(3 2 1))
   ;(β (lambda (a b) (values b a)) #(3 2 1)))
    ))

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


