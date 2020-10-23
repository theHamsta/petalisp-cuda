(in-package :petalisp-cuda/tests)

(defclass cuda-testing-backend (petalisp.test-suite::testing-backend)
  ((%cuda-backend
    :reader cuda-backend
    :initform (make-instance 'petalisp-cuda.backend:cuda-backend))))

(defun make-testing-backend ()
  (let ((cl-cuda:*show-messages* nil))
   (make-instance 'cuda-testing-backend)))

(defmethod compute-immediates ((data-structures list) (testing-backend cuda-testing-backend))
  (with-accessors ((reference-backend petalisp.test-suite::reference-backend)
                   (ir-backend petalisp.test-suite::ir-backend-compiled)
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

(defmacro with-testing-backend (&body body)
  `(let ((petalisp-cuda.backend:*transfer-back-to-lisp* t)
         (cl-cuda:*show-messages* nil)
         (petalisp:*backend* *test-backend*)
         (PETALISP.TEST-SUITE::*PASS-COUNT* 0))
     (call-with-testing-backend (lambda () ,@body))))

(defun call-with-testing-backend (thunk)
  (funcall thunk))
