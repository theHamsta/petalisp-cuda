(in-package :betalisp)

(defvar *gpu-storage-table*)
(defvar *preferred-block-size* (16 16 1))

(defclass cuda-backend (petalisp.core:backend)
  ())

(defmethod initialize-instance :after ((backend cuda-backend) &key)
  (progn 
    (unless *cuda-context*
        (progn
          (init-cuda)
          (setq *cuda-device* (get-cuda-device 0))
          (setq *cuda-context* (create-cuda-context))))

(defgeneric compile-kernel (backend kernel)
  (:method ((backend cuda-backend) kernel)
    ;; Surprisingly, compiling these things isn't too slow. We don't need this message.
    ;; (format *debug-io* "~&Compiling ~s...~%" (petalisp.ir:kernel-blueprint kernel))
    (with-standard-io-syntax 
      (let* ((gpu-code (kernel->gpu-code kernel))
             (oclcl-code (gpu-code->oclcl-code gpu-code))
             (program (eazy-opencl.host:create-program-with-source
                       (oclcl-context backend)
                       (oclcl-info-program oclcl-code))))
        (eazy-opencl.host:build-program program)
        (make-gpu-kernel :program program
                         :load-instructions
                         (oclcl-info-load-instructions oclcl-code))))))

(defgeneric find-kernel (backend kernel)
  (:documentation "Find a GPU kernel that will run the code in the kernel KERNEL.")
  (:method ((backend cuda-backend) kernel)
    (let ((blueprint (petalisp.ir:kernel-blueprint kernel)))
    (multiple-value-bind (gpu-kernel win?)
        (gethash blueprint (oclcl-kernel-cache backend))
      (if win?
          gpu-kernel
          (setf (gethash blueprint (oclcl-kernel-cache backend))
                (compile-kernel backend kernel)))))))

(defgeneric buffer-suitable-p (backend buffer)
  (:documentation "A predicate that is satisfied when the buffer BUFFER can be used by the backend BACKEND with an alien OpenCL device.")
  (:method ((backend cuda-backend) buffer)
    (let ((storage (petalisp.ir:buffer-storage buffer)))
      (or (gpu-array-p storage)
          (subtypep (array-element-type storage) 'float)
          (every #'floatp (make-array (array-total-size storage)
                                      :displaced-to storage
                                      :element-type (array-element-type storage)))))))

(defgeneric execute-kernel (backend kernel)
  (:method ((backend cuda-backend) kernel)
    (execute-gpu-kernel backend
                        (find-kernel backend kernel)
                        kernel)))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
  (let ((*gpu-storage-table* (make-hash-table))
        (*running-in-oclcl* t))
    (petalisp.scheduler:schedule-on-workers
     lazy-arrays
     1
     (lambda (tasks)
       (loop for task in tasks
             for kernel = (petalisp.scheduler:task-kernel task)
             do (execute-kernel backend kernel)))
     (constantly nil)
     (lambda (buffer)
       (let* ((dimensions (mapcar #'petalisp:range-size
                                  (petalisp:shape-ranges
                                   (petalisp.ir:buffer-shape buffer))))
              (storage (make-array dimensions
                                   :element-type 'nil)))
         (setf (petalisp.ir:buffer-storage buffer)
               storage
               (gethash storage *gpu-storage-table*)
               (make-gpu-array backend dimensions))))
     (lambda (buffer)
       (unless (null (petalisp.ir:buffer-storage buffer))
         (remhash (petalisp.ir:buffer-storage buffer) *gpu-storage-table*)
         (setf (petalisp.ir:buffer-storage buffer) nil))))))

(defmethod petalisp.core:coerce-to-lazy-array :around ((array array))
  (if *running-in-oclcl*
      (let ((gpu-array (gethash array *gpu-storage-table*)))
        (if (null gpu-array)
            (call-next-method)
            (petalisp.core:coerce-to-lazy-array
             (gpu-array->array gpu-array))))
      (let ((*running-in-oclcl* nil))
        (call-next-method))))

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array cuda-array))
  (gpu-array->array gpu-array)) 
