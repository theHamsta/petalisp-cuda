(defpackage petalisp-cuda
  (:shadowing-import-from #:petalisp #:set-difference)
  (:use :cl
        :petalisp
        :iterate)
  (:shadowing-import-from :petalisp-cuda.backend #:use-cuda-backend)
  (:export :use-cuda-backend))
(in-package :petalisp-cuda)

