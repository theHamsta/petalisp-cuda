(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp.ir
        :petalisp-cuda.memory.memory-pool)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array)
  (:export cuda-backend
           use-cuda-backend))
(in-package :petalisp-cuda.backend)

(defvar *preferred-block-size* '(16 16 1))

(defun cl-cuda-type-from-ntype (ntype &optional avoid-64bit-types)
  (petalisp.type-inference:ntype-subtypecase ntype
    (integer          'cl-cuda:int)
    ((signed-byte 32) 'cl-cuda:int)
    (single-float     'cl-cuda:float)
    (double-float     (if avoid-64bit-types
                          'cl-cuda:float
                          'cl-cuda:double))
    ;(t                'cl-cuda:bool)))
    (t (error "Cannot convert ~S to a CFFI type."
              (petalisp.type-inference:type-specifier ntype)))))


(defun use-cuda-backend ()
  (if (typep petalisp:*backend* 'cuda-backend)
      petalisp:*backend*
      (progn 
        (when petalisp:*backend*
          (petalisp.core:delete-backend petalisp:*backend*))
        (setq petalisp:*backend* (make-instance 'cuda-backend)))))

(defclass cuda-backend (petalisp.core:backend)
  ((kernel-cache :initform (make-hash-table)
                 :reader cuda-kernel-cache)
   (allocated-cuda-context :initform nil
                           :accessor allocated-cuda-context)
   (cudnn-handler :initform (petalisp-cuda.cudalibs:make-cudnn-handler)
                  :accessor cudnn-handler)
   (memory-pool :initform (make-cuda-memory-pool)
                :accessor cuda-memory-pool)
   (avoid-64bit-types :initform nil
                      :accessor avoid-64bit-types)
   (device :initform nil
           :accessor backend-device)))

(defmethod initialize-instance :after ((backend cuda-backend) &key)
  (progn
    (unless (and (boundp 'cl-cuda:*cuda-context*) cl-cuda:*cuda-context*)
      (progn
        (cl-cuda:init-cuda)
        (setf cl-cuda:*cuda-device* (cl-cuda:get-cuda-device 0))
        (setf cl-cuda:*cuda-context* (cl-cuda:create-cuda-context cl-cuda:*cuda-device*))
        (setf (allocated-cuda-context backend) cl-cuda:*cuda-context*)))
    (setf (backend-device backend) (petalisp-cuda.device:make-cuda-device cl-cuda:*cuda-device*))))
      

(defgeneric compile-kernel (backend kernel)
  (:method ((backend cuda-backend) kernel)
    (print kernel)))

(defgeneric execute-kernel (backend kernel)
   (:method ((backend cuda-backend) kernel)
      (print kernel)))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
  (let ((memory-pool (cuda-memory-pool backend))
        (avoid-64bit-types (avoid-64bit-types backend)))
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
      ;; Allocate.
      (lambda (buffer)
        (progn
        (setf (slot-value buffer 'petalisp.ir::device) (backend-device backend)); actually here's already to late to set device. ir generator should know about the device
        (setf (buffer-storage buffer)
              (petalisp-cuda.memory.cuda-array:make-cuda-array (buffer-shape buffer) 
                                                        (cl-cuda-type-from-ntype (buffer-ntype buffer) avoid-64bit-types)
                                                        nil
                                                        (lambda (type size) (memory-pool-allocate memory-pool type size))))))
      ;; Deallocate.
      (lambda (buffer)
        (let ((storage (buffer-storage buffer)))
          (unless (null storage)
            (setf (buffer-storage buffer) nil)
            (when (buffer-reusablep buffer)
              (petalisp-cuda.memory.cuda-array:free-cuda-array storage (lambda (mem-block) (memory-pool-free memory-pool mem-block))))))))))

;(defclass cuda-immediate (petalisp.core:array-immediate))

;(defun make-cuda-array-immediate (array &optional reusablep)
  ;(progn
    ;(check-type array cuda-array)
    ;(make-instance 'cuda-array-immediate
                   ;:shape (shape array)
                   ;:storage array
                   ;:reusablep reusablep
                   ;:ntype (petalisp.type-inference:array-element-ntype array)))) ;TODO

(defmethod lazy-array (array )
  )
  
;(defmethod petalisp.core:coerce-to-lazy-array :around ((array array))
  ;(if *running-in-oclcl*
      ;(let ((gpu-array (gethash array *gpu-storage-table*)))
        ;(if (null gpu-array)
            ;(call-next-method)
            ;(petalisp.core:coerce-to-lazy-array
             ;(gpu-array->array gpu-array))))
      ;(let ((*running-in-oclcl* nil))
        ;(call-next-method))))

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array petalisp-cuda.memory.cuda-array:cuda-array))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp cuda-array)) 

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (progn
    (let ((context? (allocated-cuda-context backend)))
      (when context? (progn
                       (cl-cuda:destroy-cuda-context context?) 
                       (setf (allocated-cuda-context backend) nil))))
    (petalisp-cuda.cudalibs:finalize-cudnn-handler (cudnn-handler backend))))
