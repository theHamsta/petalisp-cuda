(defpackage petalisp-cuda.utils.petalisp
  (:use :cl
        :petalisp.ir
        :petalisp)
  (:export :pass-as-scalar-argument-p
           :scalar-buffer-p))

(in-package petalisp-cuda.utils.petalisp)

(defun scalar-buffer-p (buffer)
  (= 0 (shape-rank (buffer-shape buffer))))

(defun pass-as-scalar-argument-p (buffer)
  (and (scalar-buffer-p buffer)
       (arrayp (buffer-storage buffer))
       ; we pass the arguments with a pointer array right now
       ; and all are types are 8 bytes or smaller
       (= 8 (cffi:foreign-type-size :pointer))))
