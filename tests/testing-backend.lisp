(in-package :petalisp-cuda/tests)

(defclass cuda-testing-backend (petalisp.test-suite::testing-backend)
  ((%cuda-backend
    :reader cuda-backend
    :initform (make-instance 'petalisp-cuda.backend:cuda-backend))))

(defun make-testing-backend ()
  (let ((cl-cuda:*show-messages* nil))
   (make-instance 'cuda-testing-backend)))

(defmethod backend-compute
    ((testing-backend cuda-testing-backend)
     (data-structures list))
  (with-accessors ((multicore-backend petalisp.test-suite::multicore-backend)
                   (cuda-backend cuda-backend)) testing-backend
    (let ((native-backend-solutions
            (backend-compute multicore-backend data-structures))
          (cuda-backend-solutions
            (backend-compute cuda-backend data-structures)))
      (petalisp.test-suite::compare-solutions native-backend-solutions cuda-backend-solutions)
      native-backend-solutions)))

(defmethod delete-backend ((testing-backend cuda-testing-backend))
  (delete-backend (cuda-backend testing-backend))
  (call-next-method))

(defmacro with-testing-backend (&body body)
  `(let ((petalisp-cuda.backend:*transfer-back-to-lisp* t)
         (cl-cuda:*show-messages* (if petalisp-cuda.options:*silence-cl-cuda* nil cl-cuda:*show-messages*))
         (petalisp:*backend* *test-backend*)
         (PETALISP.TEST-SUITE::*PASS-COUNT* 0))
     (call-with-testing-backend (lambda () ,@body))))

(defun call-with-testing-backend (thunk)
  (funcall thunk))

(defmethod approximately-equal ((a t) (b single-float))
  (< (abs (- a b)) (* 64 single-float-epsilon)))
(defmethod approximately-equal ((a single-float) (b t))
  (< (abs (- a b)) (* 64 single-float-epsilon)))
(defmethod approximately-equal ((a t) (b double-float))
  (< (abs (- a b)) (* 64 double-float-epsilon)))
(defmethod approximately-equal ((a double-float) (b t))
  (< (abs (- a b)) (* 64 double-float-epsilon)))

