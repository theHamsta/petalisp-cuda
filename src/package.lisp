(defpackage petalisp-cuda
  (:use :cl)
  (:import-from :petalisp-cuda.backend :use-cuda-backend
                                       :cuda-memory-pool
                                       :cuda-backend
                                       :*transfer-back-to-lisp*)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:export :use-cuda-backend
           :reclaim-cuda-memory
           :*transfer-back-to-lisp*))
(in-package :petalisp-cuda)

(defun reclaim-cuda-memory ()
  (unless (typep petalisp:*backend* 'cuda-backend)
    (error "petalisp:*backend* is not a CUDA backend: ~A" petalisp:*backend*))
  (petalisp-cuda.memory.memory-pool:reclaim-cuda-memory (cuda-memory-pool petalisp:*backend*)))
