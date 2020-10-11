(defpackage petalisp-cuda/tests
  (:use :cl
        :petalisp
        :petalisp.core
        :petalisp-cuda
        :cl-cuda
        :rove)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:export :run-tests))
(in-package :petalisp-cuda/tests)

; NOTE: To run this test file, execute `(asdf:test-system :petalisp-cuda)' in your Lisp.

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


(defclass cuda-testing-backend (petalisp.test-suite::testing-backend)
  ((%cuda-backend
    :reader cuda-backend
    :initform (make-instance 'petalisp-cuda.backend:cuda-backend))))

(defun make-testing-backend ()
  (make-instance 'cuda-testing-backend))

(defmethod compute-immediates ((data-structures list) (testing-backend cuda-testing-backend))
  (with-accessors ((reference-backend petalisp.test-suite::reference-backend)
                   (ir-backend petalisp.test-suite::ir-backend)
                   (native-backend petalisp.test-suite::native-backend)
                   (cuda-backend cuda-backend)) testing-backend
    (let ((reference-solutions
            (compute-immediates data-structures reference-backend))
          (ir-backend-solutions
            (compute-immediates data-structures ir-backend))
          (native-backend-solutions
            (compute-immediates data-structures native-backend))
          (cuda-backend-solutions
            (compute-immediates data-structures cuda-backend)))
      (petalisp.test-suite::compare-solutions reference-solutions ir-backend-solutions)
      (petalisp.test-suite::compare-solutions reference-solutions native-backend-solutions)
      (petalisp.test-suite::compare-solutions reference-solutions cuda-backend-solutions)
      reference-solutions)))

(defmethod delete-backend ((testing-backend cuda-testing-backend))
  (delete-backend (cuda-backend testing-backend))
  (call-next-method))

(defun call-with-testing-backend (thunk)
  (let ((*backend* (make-testing-backend)))
    (unwind-protect (funcall thunk)
      (delete-backend *backend*))))

(defmacro with-testing-backend (&body body)
  `(call-with-testing-backend (lambda () ,@body)))

(deftest test-petalisp.test-suite
  (let ((petalisp-cuda.backend:*transfer-back-to-lisp* t)
        (cl-cuda:*show-messages* nil))
    (with-testing-backend
      (mapcar (lambda (test) (testing (format nil "~A" test) (petalisp.test-suite:run-tests test))) (petalisp.test-suite::all-tests)))))
