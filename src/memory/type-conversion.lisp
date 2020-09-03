(defpackage :petalisp-cuda.memory.type-conversion
  (:use :cl)
  (:import-from petalisp-cuda.memory.cuda-array :cuda-array-type
                                                :type-from-cl-cuda-type)
  (:export :cl-cuda-type-from-ntype
           :cl-cuda-type-from-buffer
           :ntype-from-cl-cuda-type
           :ntype-cuda-array))
(in-package :petalisp-cuda.memory.type-conversion)

(defun cl-cuda-type-from-ntype (ntype)
  (petalisp.type-inference:ntype-subtypecase ntype
    (integer            'cl-cuda:int)
    ((unsigned-byte 2)  'uint8) 
    ((unsigned-byte 4)  'uint8)
    ((unsigned-byte 8)  'uint8)
    ((unsigned-byte 16) 'uint16)
    ((unsigned-byte 32) 'uint32)
    ((unsigned-byte 64) 'uint64)
    ((signed-byte 2)    'int8) 
    ((signed-byte 4)    'int8)
    ((signed-byte 8)    'int8)
    ((signed-byte 16)   'int16)
    ((signed-byte 32)   'cl-cuda:int)
    ((signed-byte 64)   'int64)
    (single-float       'cl-cuda:float)
    (double-float       'cl-cuda:double)
    (number             'cl-cuda:float)
    ; in doubt use float!
    (t                  'cl-cuda:float)
    ;(t (error "Cannot convert ~S to a CFFI type."
              ;(petalisp.type-inference:type-specifier ntype)))
    ))

(defun cl-cuda-type-from-buffer (buffer)
  (cl-cuda-type-from-ntype (petalisp.ir:buffer-ntype buffer)))

(defun ntype-from-cl-cuda-type (element-type)
  (petalisp.type-inference:ntype (type-from-cl-cuda-type element-type)))

(defun ntype-cuda-array (cu-array)
  (ntype-from-cl-cuda-type (cuda-array-type cu-array)))
