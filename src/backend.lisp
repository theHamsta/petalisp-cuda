(defpackage :petalisp-cuda.backend
  (:use :cl)
  (:export cuda-backend
           use-cuda-backend))
(in-package :petalisp-cuda.backend)

(defvar *gpu-storage-table*)
(defvar *preferred-block-size* '(16 16 1))
(defvar +petalisp-cuda-typemapping+ (make-hash-table))
(defparameter *allocated-cuda-context nil)
(setf (gethash '(unsigned-byte 32) +petalisp-cuda-typemapping+) 'int) 
(setf (gethash 'single-float +petalisp-cuda-typemapping+) 'float) 
(setf (gethash 'double-float +petalisp-cuda-typemapping+) 'double) 

(defun petalisp-to-cuda-type (petalisp-type)
  (gethash petalisp-type +petalisp-to-cuda-type+))


(defun use-cuda-backend ()
  (unless (typep petalisp:*backend* 'cuda-backend)
    (progn 
      (when petalisp:*backend*
        (petalisp.core:delete-backend petalisp:*backend*))
      (setq petalisp:*backend* (make-instance 'cuda-backend)))))

(defclass cuda-backend (petalisp.core:backend)
  ((kernel-cache :initform (make-hash-table) :reader cuda-kernel-cache)
   (allocated-cuda-context :initform nil :accessor allocated-cuda-context)))

(defmethod initialize-instance :after ((backend cuda-backend) &key)
  (unless (and (boundp 'cl-cuda:*cuda-context*) cl-cuda:*cuda-context*)
    (progn
      (cl-cuda:init-cuda)
      (setf cl-cuda:*cuda-device* (cl-cuda:get-cuda-device 0))
      (setf cl-cuda:*cuda-context* (cl-cuda:create-cuda-context cl-cuda:*cuda-device*))))
      (setf (allocated-cuda-context backend) cl-cuda:*cuda-context*))

(defgeneric compile-kernel (backend kernel)
  (:method ((backend cuda-backend) kernel)
    print kernel))

(defgeneric find-kernel (backend kernel)
  (:documentation "Find a GPU kernel that will run the code in the kernel KERNEL.")
  (:method ((backend cuda-backend) kernel)
    (let ((blueprint (petalisp.ir:kernel-blueprint kernel)))
      (multiple-value-bind (cuda-kernel in-cache?)
          (gethash blueprint (kernel-cache backend))
        (if in-cache?
            gpu-kernel
            (setf (gethash blueprint (kernel-cache backend))
                  (compile-kernel backend kernel)))))))

(defgeneric execute-kernel (backend kernel)
   (:method ((backend cuda-backend) kernel)
      (print kernel)))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
  (petalisp.scheduler:schedule-on-workers
    lazy-arrays
    1 ; single worker
    ;; Execute.
    (lambda (tasks)
      (loop for task in tasks do
            (let* ((kernel (petalisp.scheduler:task-kernel task)))
              (execute-kernel backend kernel))))
    ;; Barrier.
    (lambda () ())
    ;; Allocate. TODO: use cuda memory pool
    (lambda (buffer)
      (setf (buffer-storage buffer)
            (petalisp-cuda.cuda-array:make-cuda-array (buffer-shape buffer) 
              (petalisp-to-cuda-type (petalisp.type-inference:type-specifier
                (buffer-ntype buffer)))))
    ;; Deallocate.
    (lambda (buffer)
      (let ((storage (buffer-storage buffer)))
        (unless (null storage)
          (petalisp-cuda.cuda-array:free-cuda-array storage)
          (setf (buffer-storage buffer) nil)
         #| (when (buffer-reusablep buffer)|#
            #|(memory-pool-free memory-pool storage)|#))))))

;(defmethod petalisp.core:coerce-to-lazy-array :around ((array array))
  ;(if *running-in-oclcl*
      ;(let ((gpu-array (gethash array *gpu-storage-table*)))
        ;(if (null gpu-array)
            ;(call-next-method)
            ;(petalisp.core:coerce-to-lazy-array
             ;(gpu-array->array gpu-array))))
      ;(let ((*running-in-oclcl* nil))
        ;(call-next-method))))

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array cl-cuda.api.memory::memory-block))
  (petalisp-cuda.cuda-array:copy-memory-block-to-lisp)) 

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array petalisp-cuda.cuda-array:cuda-array))
  (petalisp-cuda.cuda-array:copy-cuda-array-to-lisp)) 

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (let ((context? (allocated-cuda-context backend)))
    (when context? (progn
                     (cl-cuda:destroy-cuda-context context?) 
                     (setf (allocated-cuda-context backend) nil)))))
