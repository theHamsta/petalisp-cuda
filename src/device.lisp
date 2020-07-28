(defpackage petalisp-cuda.device
  (:use :cl
        :petalisp-cuda.memory.memory-pool)
  (:export :cuda-device
           :make-cuda-device
           :device-id
           :device-memory-size))
(in-package petalisp-cuda.device)

;;; CUDA device
(defclass cuda-device ()
  ((%name :initarg :name
          :initform (alexandria:required-argument :name)
          :type string
          :reader :device-name)
   (%memory-size :initarg :memory-size
                 :initform (alexandria:required-argument :memory-size)
                 :type integer
                 :reader :device-memory-size)
   (%device-id :initarg :device-id
               :initform (alexandria:required-argument :device-id)
               :type integer
               :reader :device-foo)
   (compute-capability :initarg :compute-capability
                       :initform (alexandria:required-argument :compute-capability))))


(defun make-cuda-device (device-id)
  (multiple-value-bind (cc-major cc-minor) (cl-cuda.api.context::device-compute-capability device-id)
  (cffi:with-foreign-object (mem-size '(:pointer :pointer)); :pointer has same size as size-t
    (progn
      ( cl-cuda.driver-api:cu-device-total-mem mem-size device-id)
      (make-instance 'cuda-device 
                     :device-id device-id
                     :name (get-device-name device-id)
                     :compute-capability (+ (* 10 cc-major) cc-minor)
                     :memory-size (cffi:pointer-address (cffi:mem-ref mem-size :pointer)))))))

(defun get-device-name (device-id)
  (let* ((name "                                                                  ")
         (length (length name)))
    (cffi:with-foreign-string (name-ptr name)
      (progn
        (cl-cuda.driver-api:cu-device-get-name name-ptr length device-id)
        (cffi:foreign-string-to-lisp name-ptr)))))


;;TODO
(defgeneric allocate-buffer (buffer device))

(defgeneric deallocate-buffer (buffer))
