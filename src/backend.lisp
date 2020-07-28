(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp.ir
        :petalisp-cuda.memory.memory-pool)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array)
  (:export cuda-backend
           use-cuda-backend
           cl-cuda-type-from-buffer
           compile-cache
           preferred-block-size
           execute-kernel
           ntype-from-cl-cuda-type))
(in-package :petalisp-cuda.backend)

; push missing cffi types
(push '(int8 :int8 "int8_t") cl-cuda.lang.type::+scalar-types+)
(push '(int16 :int16 "int16_t") cl-cuda.lang.type::+scalar-types+)
(push '(int64 :int64 "int64_t") cl-cuda.lang.type::+scalar-types+)

(push '(uint8 :uint8 "uint8_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint16 :uint16 "uint16_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint32 :uint32 "uint32_t") cl-cuda.lang.type::+scalar-types+)
(push '(uint64 :uint64 "uint64_t") cl-cuda.lang.type::+scalar-types+)

(push '(float* :float* "float*") cl-cuda.lang.type::+scalar-types+)
    

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
    (t (error "Cannot convert ~S to a CFFI type."
              (petalisp.type-inference:type-specifier ntype)))))

(defun cl-cuda-type-from-buffer (buffer)
  (cl-cuda-type-from-ntype (petalisp.ir:buffer-ntype buffer)))

(defun ntype-from-cl-cuda-type (element-type)
  (petalisp.type-inference:ntype
    (cond
      ((equal element-type :uint8)  '(unsigned-byte 8))
      ((equal element-type :uint16) '(unsigned-byte 16))
      ((equal element-type :uint32) '(unsigned-byte 32))
      ((equal element-type :uint64) '(unsigned-byte 64))
      ((equal element-type :int8)   '(signed-byte 8))
      ((equal element-type :int16)  '(signed-byte 16))
      ((equal element-type :int)    '(signed-byte 32))
      ((equal element-type :int64)  '(signed-byte 64))
      ((equal element-type :float)  'single-float)
      ((equal element-type :double) 'double-float)     
       (t (error "Cannot convert ~S to ntype." element-type)))))

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
   (device :initform nil
           :accessor backend-device)
   (preferred-block-size :initform '(16 16 1)
                         :accessor preferred-block-size)
   (%compile-cache :initform (make-hash-table) :reader compile-cache :type hash-table)))

(defmethod initialize-instance :after ((backend cuda-backend) &key)
  (progn
    (unless (and (boundp 'cl-cuda:*cuda-context*) cl-cuda:*cuda-context*)
      (progn
        (cl-cuda:init-cuda)
        (setf cl-cuda:*cuda-device* (cl-cuda:get-cuda-device 0))
        (setf cl-cuda:*cuda-context* (cl-cuda:create-cuda-context cl-cuda:*cuda-device*))
        (setf (allocated-cuda-context backend) cl-cuda:*cuda-context*)))
    (setf (backend-device backend) (petalisp-cuda.device:make-cuda-device cl-cuda:*cuda-device*))))
      

(defgeneric execute-kernel (kernel backend))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
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
                (lambda (mem-block) (memory-pool-free memory-pool mem-block))))))))))


(defclass cuda-immediate (petalisp.core:immediate)
  ((%reusablep :initarg :reusablep :initform nil :reader reusablep)
   (%storage :initarg :storage :reader petalisp.ir::storage)))


(defun make-cuda-immediate (array &optional reusablep)
  (progn
    (check-type array cuda-array)
    (make-instance 'cuda-immediate
                   :shape (petalisp.core:shape array)
                   :storage array
                   :reusablep reusablep
                   :ntype (petalisp.type-inference:ntype-of array))))

(defmethod lazy-array ((array cuda-array))
  (make-cuda-immediate array))
  
(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array petalisp-cuda.memory.cuda-array:cuda-array))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp cuda-array)) 

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-immediate cuda-immediate))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp (petalisp.core:storage cuda-immediate))) 

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (progn
    (let ((context? (allocated-cuda-context backend)))
      (when context? (progn
                       (cl-cuda:destroy-cuda-context context?) 
                       (setf (allocated-cuda-context backend) nil))))
    (petalisp-cuda.cudalibs:finalize-cudnn-handler (cudnn-handler backend))))
