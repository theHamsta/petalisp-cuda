(defpackage petalisp-cuda
  (:shadowing-import-from #:petalisp #:set-difference)
  (:use :cl
        :petalisp
        :iterate)
  (:shadowing-import-from :petalisp-cuda.backend :use-cuda-backend)
  (:shadowing-import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:export :use-cuda-backend
           :make-cuda-array
           :*transfer-back-to-lisp*))
(in-package :petalisp-cuda)

