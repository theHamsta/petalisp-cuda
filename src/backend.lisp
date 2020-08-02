(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp-cuda.memory.memory-pool)
  (:import-from :petalisp-cuda.type-conversion
                :cl-cuda-type-from-buffer
                :ntype-cuda-array
                :ntype-from-cl-cuda-type
                :ntype-cuda-array)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array
                :cuda-array-type
                :type-from-cl-cuda-type)
  (:import-from :cl-cuda.lang.type
                :float
                :int
                :double)
  (:export cuda-backend
           use-cuda-backend
           cl-cuda-type-from-buffer
           compile-cache
           preferred-block-size
           execute-kernel
           *transfer-back-to-lisp*))
(in-package :petalisp-cuda.backend)

(defparameter *silence-cl-cuda* t)
(defparameter *transfer-back-to-lisp* t)

; push missing cffi types
(push '(int8 :int8 "int8_t") cl-cuda.lang.type::+scalar-types+)
(push '(int16 :int16 "int16_t") cl-cuda.lang.type::+scalar-types+)
(push '(int64 :int64 "int64_t") cl-cuda.lang.type::+scalar-types+)

(push '(uint8 :uint8 "uint8_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint16 :uint16 "uint16_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint32 :uint32 "uint32_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint64 :uint64 "uint64_t") cl-cuda.lang.type::+scalar-types+)

(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'max)
      '(((float float) float nil "fmaxf")
        ((double double) double nil "fmax")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'min)
      '(((float float) float nil "fminf")
        ((double double) double nil "fmin")))

(defun use-cuda-backend ()
  (let ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
   (if (typep petalisp:*backend* 'cuda-backend)
      petalisp:*backend*
      (progn 
        (when petalisp:*backend*
          (petalisp.core:delete-backend petalisp:*backend*))
        (setq petalisp:*backend* (make-instance 'cuda-backend))))))

(defclass cuda-backend (petalisp.core:backend)
  ((backend-context :initform nil
                    :accessor backend-context)
   (cudnn-handler :initform (petalisp-cuda.cudalibs:make-cudnn-handler)
                  :accessor cudnn-handler)
   (memory-pool :initform (make-cuda-memory-pool)
                :accessor cuda-memory-pool)
   (device :initform nil
           :accessor backend-device)
   (device-id :initform nil
              :accessor backend-device-id)
   (preferred-block-size :initform '(16 16 1)
                         :accessor preferred-block-size)
   (%compile-cache :initform (make-hash-table :test 'equal) :reader compile-cache :type hash-table)))

(defmethod initialize-instance :after ((backend cuda-backend) &key)
  (unless (and (boundp 'cl-cuda:*cuda-context*) cl-cuda:*cuda-context*)
    (cl-cuda:init-cuda)
    (setf cl-cuda:*cuda-device* (cl-cuda:get-cuda-device 0))
    (setf cl-cuda:*cuda-context* (cl-cuda:create-cuda-context cl-cuda:*cuda-device*)))
  (setf (backend-context backend) cl-cuda:*cuda-context*)
  (setf (backend-device-id backend) cl-cuda:*cuda-device*)
  (setf (backend-device backend) (petalisp-cuda.device:make-cuda-device cl-cuda:*cuda-device*)))


(defgeneric execute-kernel (kernel backend))

(defmethod petalisp.core:compute-on-backend ((lazy-arrays list) (backend cuda-backend))
  (let* ((collapsing-transformations
           (mapcar (alexandria:compose #'collapsing-transformation #'shape)
                   lazy-arrays))
         (immediates
           (petalisp.core:compute-immediates
            (mapcar #'transform lazy-arrays collapsing-transformations)
            backend)))
    (loop for lazy-array in lazy-arrays
          for collapsing-transformation in collapsing-transformations
          for immediate in immediates
          do (petalisp.core:replace-lazy-array
              lazy-array
              (petalisp.core:lazy-reshape immediate (shape lazy-array) collapsing-transformation)))
    (values-list (mapcar (if *transfer-back-to-lisp* #'petalisp.core:lisp-datum-from-immediate #'storage) immediates))))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
  (let* ((cl-cuda:*cuda-device* (backend-device-id backend))
         (cl-cuda:*cuda-context* (backend-context backend))
         (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
         (cl-cuda.api.nvcc:*nvcc-options*
           (if (cl-cuda.api.context::arch-exists-p
                 cl-cuda.api.nvcc:*nvcc-options*)
               cl-cuda.api.nvcc:*nvcc-options*
               (cl-cuda.api.context::append-arch cl-cuda.api.nvcc:*nvcc-options* cl-cuda:*cuda-device*))))
    (let ((memory-pool (cuda-memory-pool backend)))
      (petalisp.scheduler:schedule-on-workers
        lazy-arrays
        1 ; single worker
        ;; Execute.

        (lambda (tasks)
          (loop for task in tasks do
                (let* ((kernel (petalisp.scheduler:task-kernel task)))
                  (execute-kernel kernel backend))))
        ;; Barrier.
        (lambda () ())
        ;; Allocate.
        (lambda (buffer)
          (setf (petalisp.ir:buffer-storage buffer)
                (petalisp-cuda.memory.cuda-array:make-cuda-array (buffer-shape buffer) 
                                                                 (cl-cuda-type-from-buffer buffer)
                                                                 nil
                                                                 (lambda (type size)
                                                                   (memory-pool-allocate memory-pool type size)))))
        ;; Deallocate.
        (lambda (buffer)
          (let ((storage (buffer-storage buffer)))
            (unless (null storage)
              (setf (buffer-storage buffer) nil)
              (when (buffer-reusablep buffer)
                (petalisp-cuda.memory.cuda-array:free-cuda-array
                  storage
                  (lambda (mem-block) (memory-pool-free memory-pool mem-block)))))))))))

(defclass cuda-immediate (petalisp.core:immediate)
  ((%reusablep :initarg :reusablep :initform nil :accessor reusablep)
   (%ntype :initarg :ntype :initform nil :accessor petalisp.core:element-ntype)
   (%shape :initarg :shape :initform nil :accessor petalisp.core:shape)
   (%storage :initarg :storage :accessor petalisp.core:storage)))

(defun make-cuda-immediate (cu-array &optional reusablep)
    (check-type cu-array cuda-array)
    (make-instance 'cuda-immediate
                   :shape (shape cu-array)
                   :storage cu-array
                   :reusablep reusablep
                   :ntype (ntype-cuda-array cu-array)))

(defmethod petalisp.core:lazy-array ((array cuda-array))
  (make-cuda-immediate array))
  
(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array petalisp-cuda.memory.cuda-array:cuda-array))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp cuda-array)) 

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-immediate cuda-immediate))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp (petalisp.core:storage cuda-immediate))) 

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (let ((context? (backend-context backend)))
    (when context? (progn
                     (cl-cuda:destroy-cuda-context context?) 
                     (setf (backend-context backend) nil))))
  (petalisp-cuda.cudalibs:finalize-cudnn-handler (cudnn-handler backend)))


(defmethod petalisp.core:replace-lazy-array ((instance lazy-array) (replacement cuda-immediate))
  (change-class instance (class-of replacement)
    :storage (storage replacement)
    :ntype (petalisp.core:element-ntype replacement)
    :shape (shape replacement)))
