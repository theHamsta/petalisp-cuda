(defpackage petalisp-cuda
  (:shadowing-import-from #:petalisp #:set-difference)
  (:use :cl
        :petalisp
        :iterate)
  (:import-from :petalisp-cuda.backend :use-cuda-backend :cuda-memory-pool :*transfer-back-to-lisp*)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:export :use-cuda-backend
           :make-cuda-array
           :reclaim-cuda-memory
           :*transfer-back-to-lisp*))
(in-package :petalisp-cuda)

(defun reclaim-cuda-memory ()
  (petalisp-cuda.memory.memory-pool:reclaim-cuda-memory (cuda-memory-pool *backend*)))
