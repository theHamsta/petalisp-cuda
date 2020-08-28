(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp-cuda.memory.memory-pool)
  (:import-from :petalisp.core
                :identity-transformation)
  (:import-from :petalisp.native-backend
                :make-worker-pool
                :worker-pool-size
                :worker-pool-enqueue)
  (:import-from :petalisp-cuda.type-conversion
                :cl-cuda-type-from-buffer
                :ntype-cuda-array
                :ntype-from-cl-cuda-type
                :ntype-cuda-array)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array
                :cuda-array-p
                :cuda-array-type
                :type-from-cl-cuda-type)
  (:import-from :cl-cuda.lang.type
                :float
                :int
                :double)
  (:import-from :petalisp-cuda.utils.cl-cuda
                :with-cuda-stream
                :*cu-events*
                :create-corresponding-event
                :record-corresponding-event)
  (:import-from :cl-cuda.driver-api
                :cu-stream-wait-event)
  (:import-from :petalisp-cuda.utils.petalisp
                :pass-as-scalar-argument-p)
  (:export cuda-backend
           cuda-memory-pool
           use-cuda-backend
           cl-cuda-type-from-buffer
           compile-cache
           preferred-block-size
           execute-kernel
           cuda-backend-event-map
           *transfer-back-to-lisp*))
(in-package :petalisp-cuda.backend)

(defparameter *silence-cl-cuda* t)
(defparameter *transfer-back-to-lisp* nil)
(defparameter *single-threaded* t)
(defparameter *single-stream* t)

(defmacro with-cuda-backend-magic (backend &body body)
  `(let* ((cl-cuda:*cuda-context* (backend-context ,backend))
          (cl-cuda:*cuda-device* (backend-device-id ,backend))
          (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
          (cl-cuda.api.nvcc:*nvcc-options*
            (if (cl-cuda.api.context::arch-exists-p
                  cl-cuda.api.nvcc:*nvcc-options*)
                cl-cuda.api.nvcc:*nvcc-options*
                (cl-cuda.api.context::append-arch cl-cuda.api.nvcc:*nvcc-options* cl-cuda:*cuda-device*))))
     (petalisp-cuda.cudalibs::cuCtxPushCurrent_v2 cl-cuda:*cuda-context*)
         (if *single-stream*
             (let ((cl-cuda:*cuda-stream* (cffi:null-pointer)))
               ,@body)
             (with-cuda-stream (cl-cuda:*cuda-stream*)
               ,@body))))

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
   (worker-pool :initform (make-worker-pool (petalisp.utilities:number-of-cpus))
                :accessor cuda-backend-worker-pool)
   (%compile-cache :initform (make-hash-table :test #'equalp) :reader compile-cache :type hash-table)
   (event-map :initform (make-hash-table) :accessor cuda-backend-event-map)))

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
           (mapcar (alexandria:compose #'collapsing-transformation #'array-shape)
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
               (petalisp.core:lazy-reshape immediate (array-shape lazy-array) collapsing-transformation)))
    (with-cuda-backend-magic backend
      (values-list (mapcar (if *transfer-back-to-lisp* (lambda (immediate)
                                                         (let ((lisp-array (petalisp.core:lisp-datum-from-immediate immediate)))
                                                           (memory-pool-free (cuda-memory-pool backend) (storage immediate))
                                                           lisp-array))
                               #'storage)
                           immediates)))))

(defmethod petalisp.core:compute-immediates ((lazy-arrays list) (backend cuda-backend))
  (let* ((memory-pool (cuda-memory-pool backend))
         (worker-pool (cuda-backend-worker-pool backend))
         (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
         (event-map (cuda-backend-event-map backend))
         (rtn (petalisp.scheduler:schedule-on-workers
                lazy-arrays
                (worker-pool-size worker-pool)
                ;; Execute.
                (lambda (tasks)
                  ;; Ensure all kernels and buffers have events to wait for
                  (loop for task in tasks do
                        (let* ((kernel (petalisp.scheduler:task-kernel task)))
                          (create-corresponding-event kernel event-map)
                          (mapcar (lambda (buffer)
                                    (unless (or (cuda-array-p (buffer-storage buffer))
                                                (pass-as-scalar-argument-p buffer))
                                      (create-corresponding-event buffer event-map)))
                                  (kernel-buffers kernel))))
                  (loop for task in tasks do
                        (let* ((kernel (petalisp.scheduler:task-kernel task)))
                          (if *single-threaded* ; TODO: only multiple streams single thread works right now
                              (with-cuda-backend-magic backend
                                (execute-kernel kernel backend))
                              (worker-pool-enqueue
                                (lambda (worker-id)
                                  (declare (ignore worker-id))
                                  (with-cuda-backend-magic backend
                                    (execute-kernel kernel backend)))
                                worker-pool)))))
                ;; Barrier.
                (lambda () ())
                ;; Allocate.
                (lambda (buffer)
                  (with-cuda-backend-magic backend
                    (setf (petalisp.ir:buffer-storage buffer)
                          (petalisp-cuda.memory.cuda-array:make-cuda-array (buffer-shape buffer) 
                                                                           (cl-cuda-type-from-buffer buffer)
                                                                           nil
                                                                           (lambda (type size)
                                                                             (memory-pool-allocate memory-pool type size))))))
                ;; Deallocate.
                (lambda (buffer)
                  (with-cuda-backend-magic backend
                    (let ((storage (buffer-storage buffer)))
                      (unless (null storage)
                        (setf (buffer-storage buffer) nil)
                        (when (buffer-reusablep buffer)
                          (memory-pool-free memory-pool storage)))))))))
    (mapcar (lambda (event) (cu-stream-wait-event cl-cuda:*cuda-stream* event 0)) (alexandria:hash-table-values event-map))
    (mapcar #'cl-cuda.api.timer::destroy-cu-event (alexandria:hash-table-values event-map))
    (cl-cuda.api.context:synchronize-context)
    (clrhash event-map)
    rtn))

(defclass cuda-immediate (petalisp.core:non-empty-immediate)
  ((%reusablep :initarg :reusablep :initform nil :accessor reusablep)
   (%ntype :initarg :ntype :initform nil :accessor petalisp.core:element-ntype)
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
    :shape (array-shape replacement)))
