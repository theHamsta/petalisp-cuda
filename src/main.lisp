(defpackage betalisp
  (:shadowing-import-from #:petalisp #:set-difference)
  (:use :cl :petalisp :petalisp :cl-cuda :iterate))
(in-package :betalisp)

