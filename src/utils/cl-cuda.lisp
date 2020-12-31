(defpackage petalisp-cuda.utils.cl-cuda
  (:use :cl
        :cl-cuda.api.memory)
  (:import-from :cl-cuda.api.memory
                :memcpy-host-to-device-async
                :memcpy-device-to-host-async)
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
           :*cu-events*
           :sync-memory-block-async))

(in-package petalisp-cuda.utils.cl-cuda)

(defvar *cu-events*)

;; same as cl-cuda.api.timer:record-cu-event except with stream
(defun record-cu-event (cu-event)
  (cl-cuda.driver-api:cu-event-record cu-event cl-cuda:*cuda-stream*))

(defun create-corresponding-event (thing &optional (event-map *cu-events*))
  (setf (gethash thing event-map) (create-cu-event)))

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

(defun sync-memory-block-async (memory-block direction)
  (declare ((member :host-to-device :device-to-host) direction))
  (let ((device-ptr (memory-block-device-ptr memory-block))
        (host-ptr (memory-block-host-ptr memory-block))
        (type (memory-block-type memory-block))
        (size (memory-block-size memory-block)))
    (ecase direction
      (:host-to-device
       (memcpy-host-to-device-async device-ptr host-ptr type size cl-cuda:*cuda-stream*))
      (:device-to-host
       (memcpy-device-to-host-async host-ptr device-ptr type size cl-cuda:*cuda-stream*)))))
