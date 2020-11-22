(in-package :petalisp-cuda/tests)

(deftest half-array
  (unless (boundp 'cl-cuda::+has-half-dtype+)
    (skip "No half support in cl-cuda (use theHamsta's fork)"))
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((a (make-cuda-array '(20 9) :half)))
          (ok (format t "~A~%" a))
          (ok (equalp (cuda-array-strides a) '(10 1)))
          (ok (equalp (cuda-array-size a) 200))
          (free-cuda-array a))))))

(deftest test-cuda-array-device
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((a (make-cuda-array '(20 9) 'float)))
          (ok (= 0  (cuda-array-device a)))
		  (free-cuda-array a))))))

(deftest test-mem-roundtrip
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((foo (aops:rand* 'single-float '(20 9)))
               (a (make-cuda-array foo 'float))
               (b (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp a)))
          (loop for i below (reduce #'* (array-dimensions foo))
                do (progn
                     (assert (equal (row-major-aref foo i) (row-major-aref b i)))))
		  (mapcar #'free-cuda-array (list a)))))))

(deftest test-cuda-array-from-cuda-array
  (let ((cl-cuda:*show-messages* nil))
    (cl-cuda:with-cuda (0)
      (progn
        (let* ((foo (aops:rand* 'single-float '(20 9)))
               (aa (make-cuda-array foo 'float))
               (bb (make-cuda-array aa 'float)))
          (format t "~A ~A~%" aa bb)
         (mapcar #'free-cuda-array (list aa bb)))))))
