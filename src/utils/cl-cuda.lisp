(defpackage petalisp-cuda.utils.cl-cuda
  (:use :cl)
  (:import-from :cl-cuda
                :*cuda-stream*)
  (:import-from :cl-cuda.driver-api
                :cu-stream-wait-event)
  (:import-from :cl-cuda.api.timer
                :create-cu-event)
  (:export :create-corresponding-event
           :record-cu-event
           :record-corresponding-event
           :wait-for-correspoding-event
           :with-cuda-stream
           :*cu-events*))

(in-package petalisp-cuda.utils.cl-cuda)

(defvar *cu-events*)

;; same as cl-cuda.api.timer:record-cu-evetn except with stream
(defun record-cu-event (cu-event)
  (cl-cuda.driver-api:cu-event-record cu-event cl-cuda:*cuda-stream*))

(defun create-corresponding-event (thing &optional (event-map *cu-events*))
  (let ((event (gethash thing event-map (create-cu-event))))
    (setf (gethash thing event-map (create-cu-event)) event)))

(defun record-corresponding-event (thing &optional (event-map *cu-events*))
  (let ((event (gethash thing event-map)))
    (record-cu-event event)))

(defun wait-for-correspoding-event (thing &optional (event-map *cu-events*))
  (let ((event (gethash thing event-map)))
    (when event
     (cu-stream-wait-event *cuda-stream* event 0))))

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
