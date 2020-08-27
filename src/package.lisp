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
           :with-cuda-backend-raii))
(in-package :petalisp-cuda)

(defun reclaim-cuda-memory ()
  (if (cuda-backend-p petalisp:*backend*)
      (petalisp-cuda.memory.memory-pool:reclaim-cuda-memory (cuda-memory-pool petalisp:*backend*))
      (error "petalisp:*backend* is not a CUDA backend: ~A" petalisp:*backend*)))
