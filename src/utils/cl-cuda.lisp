(defpackage petalisp-cuda.utils.cl-cuda
  (:use :cl)
  (:export :record-cu-event
           :with-cuda-stream))

(in-package petalisp-cuda.utils.cl-cuda)

;; same as cl-cuda.api.timer:record-cu-evetn except with stream
(defun record-cu-event (cu-event)
  (cl-cuda.driver-api:cu-event-record cu-event cl-cuda:*cuda-stream*))

;; from cl-cuda README (mgl-mat)
(defmacro with-cuda-stream ((stream) &body body)
  (alexandria:with-gensyms (stream-pointer)
    `(cffi:with-foreign-objects
         ((,stream-pointer 'cl-cuda.driver-api:cu-stream))
       (cl-cuda.driver-api:cu-stream-create ,stream-pointer 0)
       (let ((,stream (cffi:mem-ref ,stream-pointer
                                    'cl-cuda.driver-api:cu-stream)))
         (unwind-protect
              (locally ,@body)
           (cl-cuda.driver-api:cu-stream-destroy ,stream))))))
