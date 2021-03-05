(defpackage petalisp-cuda
  (:use :cl)
  (:import-from :petalisp-cuda.backend :use-cuda-backend
                :with-cuda-backend
                :with-cuda-backend-raii
                :cuda-memory-pool
                :cuda-backend
                :*transfer-back-to-lisp*)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:import-from :petalisp-cuda.jit-execution :device-function
                :device-host-function)
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
  (values))
