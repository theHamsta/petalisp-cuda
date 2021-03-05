(defpackage petalisp-cuda
  (:use :cl)
  (:import-from :petalisp-cuda.backend :use-cuda-backend
                                       :with-cuda-backend
                                       :with-cuda-backend-raii
                                       :cuda-memory-pool
                                       :cuda-backend
                                       :*transfer-back-to-lisp*)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:export :use-cuda-backend
           :reclaim-cuda-memory
           :*transfer-back-to-lisp*
           :with-cuda-backend
           :with-cuda-backend-raii
           :device-function
           :device-host-function))
(in-package :petalisp-cuda)

(defun reclaim-cuda-memory (&optional (backend (or petalisp-cuda.backend::*cuda-backend* petalisp:*backend*)))
  (petalisp-cuda.memory.memory-pool:reclaim-cuda-memory (cuda-memory-pool backend))
  nil)

(defmacro device-function (lisp-function lambda-list body-as-single-form)
  `(setf (gethash ',lisp-function petalisp-cuda.jit-execution:*device-function-mapping*)
         (list ',lisp-function ',lambda-list ',body-as-single-form)))

(defmacro device-host-function (function-name lambda-list body-as-single-form)
  `(prog1
       (defun ,function-name ,lambda-list ,body-as-single-form)
     (setf (gethash ',function-name petalisp-cuda.jit-execution:*device-function-mapping*)
           (list ',function-name ',lambda-list ',body-as-single-form))))

(device-function petalisp.type-inference::argmax (x y)
  (if (> x y)
     0
     1))
