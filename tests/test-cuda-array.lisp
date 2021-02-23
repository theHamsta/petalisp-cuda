(in-package :petalisp-cuda/tests)

(deftest test-make-cuda-array
  (let ((cl-cuda:*show-messages* nil))
   (with-cuda (0)
    (make-cuda-array '(10 20) 'float))))

(deftest test-aligned-allocation
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (let* ((a (make-cuda-array '(20 9) 'float nil nil 32))
               (b (make-cuda-array '(20 9) 'float))
               (c (make-cuda-array '(9 3) 'double nil nil 16)))
          (ok (equalp (cuda-array-strides a) '(32 1)))
          (ok (equalp (cuda-array-size a) (* 20 32)))
          (ok (equalp (cuda-array-strides b) '(9 1)))
          (ok (equalp (cuda-array-size b) (* 20 9)))
          (ok (equalp (cuda-array-strides c) '(16 1)))
          (ok (equalp (cuda-array-size c) (* 9 16)))
          (free-cuda-array a)
          (free-cuda-array b)
          (free-cuda-array c)))))

(deftest half-array
  (if (boundp 'cl-cuda::+has-half-dtype+)
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (let* ((a (make-cuda-array '(20 9) :half))
               (b (make-cuda-array '(9) :bfloat16)))
          (ok (equalp (cuda-array-strides a) '(10 1)))
          (ok (equalp (cuda-array-size a) 200))
          (ok (equalp (cuda-array-strides b) '(1)))
          (ok (equalp (cuda-array-size b) 10))
          (free-cuda-array a)
          (free-cuda-array b)))))
    (skip "No half support in cl-cuda (use theHamsta's fork)"))

(deftest test-cuda-array-device
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((a (make-cuda-array '(20 9) 'float)))
          (ok (= 0 (cuda-array-device a)))
                  (free-cuda-array a))))))

(deftest test-mem-roundtrip
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (let* ((foo (aops:rand* 'single-float '(20 9)))
               (a (make-cuda-array foo 'float))
               (b (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp a)))
          (loop for i below (reduce #'* (array-dimensions foo))
                do (progn
                     (assert (equal (row-major-aref foo i) (row-major-aref b i)))))
                  (mapcar #'free-cuda-array (list a))))))

(deftest test-stride-tricks
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (let* ((a (make-cuda-array (aops:rand* 'single-float '(20 9)) 'float))
             (b (transform-cuda-array a (τ (a b) (b a))))
             (c (transform-cuda-array a (τ (a b) (a b 3)))))
        (ok (equalp (cuda-array-shape b) '(9 20)))
        (ok (equalp (cuda-array-strides c) '(9 1 0)))
        (ok (equalp (cuda-array-shape c) '(20 9 4)))
        (free-cuda-array a)))))

(deftest test-dont-allow-casts-in-strict-mode
  ;; should not work
  (signals
      (let ((petalisp-cuda.options:*strict-cast-mode* t))
        (with-cuda-backend
          (compute (α #'+ 1 #(1 3 4))))))
  ;; should work
  (ok
      (let ((petalisp-cuda.options:*strict-cast-mode* t))
        (with-cuda-backend
          (compute (α #'+ 1 (coerce #(1.0 3.0 4.0) '(array single-float (*))))))))
  ;; should work
  (ok
    (let ((petalisp-cuda.options:*strict-cast-mode* nil))
      (with-cuda-backend
        (compute (α #'+ 1 #(1 3 4)))))))
