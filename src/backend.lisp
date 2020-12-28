(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp-cuda.memory.memory-pool
        :petalisp-cuda.options)
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
           *transfer-back-to-lisp*
           with-cuda-backend))
(in-package :petalisp-cuda.backend)

(defparameter *cuda-backend* nil)

(defmacro with-cuda-backend-magic (backend &body body)
  `(let* ((cl-cuda:*cuda-context* (backend-context ,backend))
          (cl-cuda:*cuda-device* (backend-device-id ,backend))
          (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
          (cl-cuda.api.nvcc:*nvcc-options*
            (if (cl-cuda.api.context::arch-exists-p cl-cuda.api.nvcc:*nvcc-options*)
                cl-cuda.api.nvcc:*nvcc-options*
                (cl-cuda.api.context::append-arch cl-cuda.api.nvcc:*nvcc-options* cl-cuda:*cuda-device*))))
     (nconc cl-cuda:*nvcc-options* *nvcc-extra-options*)
     (petalisp-cuda.cudalibs::cuCtxPushCurrent_v2 cl-cuda:*cuda-context*)
     (if *single-stream*
         (let ((cl-cuda:*cuda-stream* (cffi:null-pointer)))
           ,@body)
         (with-cuda-stream (cl-cuda:*cuda-stream*)
           ,@body))))

; push missing cffi types
(pushnew '(int8 :int8 "int8_t") cl-cuda.lang.type::+scalar-types+)
(pushnew '(int16 :int16 "int16_t") cl-cuda.lang.type::+scalar-types+)
(pushnew '(int64 :int64 "int64_t") cl-cuda.lang.type::+scalar-types+)

(pushnew '(uint8 :uint8 "uint8_t") cl-cuda.lang.type::+scalar-types+)
(pushnew '(uint16 :uint16 "uint16_t") cl-cuda.lang.type::+scalar-types+)
(pushnew '(uint32 :uint32 "uint32_t") cl-cuda.lang.type::+scalar-types+)
(pushnew '(uint64 :uint64 "uint64_t") cl-cuda.lang.type::+scalar-types+)

(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'max)
      '(((float float) float nil "fmaxf")
        ((double double) double nil "fmax")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'min)
      '(((float float) float nil "fminf")
        ((double double) double nil "fmin")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'floor)
      '(((float) float nil "floor")
        ((double) double nil "floor")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'ceil)
      '(((float) float nil "ceil")
        ((double) double nil "ceil")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'sqrt)
      '(((int) float nil "sqrt")
        ((float) float nil "sqrt")
        ((double) float nil "sqrt")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'abs)
      '(((int) float nil "abs")
        ((float) float nil "abs")
        ((double) float nil "abs")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'coerce)
      '(((int float) float t "+ 0 *")
        ((float int) int t "+ 0 *")
        ((float float) float t "+ 0 *")
        ((int int) int t "+ 0 *")
        ((double float) float t "+ 0 *")
        ((float int) float t "+ 0 *")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ :coerce-float)
      '(((int float) float t "+ 0 *")
        ((float float) float t "+ 0 *")
        ((float) float nil "(float)")
        ((double) float nil "(float)")
        ((double float) float t "+ 0 *")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ :coerce-int)
      '(((int float) int t "+ 0 *")
        ((float float) int t "+ 0 *")
        ((double float) int t "+ 0 *")
        ((int) int nil "(int)")
        ((float) int nil "(int)")
        ((double) int nil "(int)")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ :coerce-double)
      '(((int double) int t "+ 0 *")
        ((float double) int t "+ 0 *")
        ((double double) int t "+ 0 *")
        ((int) float nil "(double)")
        ((float) float nil "(double)")
        ((double) float nil "(double)")
        ))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ :round)
      '(((int) int nil "1*")
        ((float) int nil "roundf")
        ((double) int nil "round")))
(setf (getf cl-cuda.lang.built-in::+built-in-functions+ 'rem)
      '(((float float) float nil "fmodf")
        ((float int) float nil "fmodf")
        ((double int) double nil "fmod")
        ((double double) double nil "fmod")))

(defun use-cuda-backend ()
  (let ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
   (if (cuda-backend-p petalisp:*backend*)
      petalisp:*backend*
      (progn 
        (when petalisp:*backend*
          (petalisp.core:delete-backend petalisp:*backend*))
        (setq petalisp:*backend* (or *cuda-backend* (make-instance 'cuda-backend)))))))

(defmacro with-cuda-backend-raii (&body body)
  `(let* ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
          (petalisp:*backend* (make-instance 'cuda-backend))
          (*transfer-back-to-lisp* T)
          (result (unwind-protect
                      ,@body
                    (petalisp.core:delete-backend petalisp:*backend*))))
     result))

(defmacro with-cuda-backend (&body body)
  `(let* ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
          (backend (or *cuda-backend* (make-instance 'cuda-backend)))
          (petalisp:*backend* (or *cuda-backend* backend)))
     (unless *cuda-backend*
       (setq *cuda-backend* backend))
     ,@body))

(defclass cuda-backend (petalisp.core:backend)
  ((backend-context :initform nil
                    :accessor backend-context)
   (cudnn-handler :initform (petalisp-cuda.cudnn-handler:make-cudnn-handler)
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

(defgeneric cuda-backend-p (thing))
(defmethod cuda-backend-p ((thing T))
  nil)
(defmethod cuda-backend-p ((thing cuda-backend))
  T)

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
                                                           (memory-pool-free (cuda-memory-pool backend) (cuda-immediate-storage immediate))
                                                           lisp-array))
                               #'cuda-immediate-storage)
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
                  ;; Ensure all kernels and buffers to be uploaded have events to wait for
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
                        (when (petalisp.ir:interior-buffer-p buffer)
                          (memory-pool-free memory-pool storage)))))))))
    (mapcar (lambda (event) (cu-stream-wait-event cl-cuda:*cuda-stream* event 0)) (alexandria:hash-table-values event-map))
    (mapcar #'cl-cuda.api.timer::destroy-cu-event (alexandria:hash-table-values event-map))
    (cl-cuda.api.context:synchronize-context)
    (clrhash event-map)
    rtn))


(defclass cuda-immediate (petalisp.core:non-empty-immediate)
  ((%storage :initarg :storage :accessor cuda-immediate-storage)))

(defgeneric cuda-immediate-storage (immediate))
(defmethod cuda-immediate-storage ((immediate array-immediate))
  (petalisp.core:array-immediate-storage immediate))

(defun make-cuda-immediate (cu-array)
    (check-type cu-array cuda-array)
    (make-instance 'cuda-immediate
                   :shape (shape cu-array)
                   :storage cu-array
                   :ntype (ntype-cuda-array cu-array)))

(defmethod petalisp.core:lazy-array ((array cuda-array))
  (make-cuda-immediate array))
  
(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-array petalisp-cuda.memory.cuda-array:cuda-array))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp cuda-array)) 

(defmethod petalisp.core:lisp-datum-from-immediate ((cuda-immediate cuda-immediate))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp (cuda-immediate-storage cuda-immediate))) 

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (petalisp-cuda.cudnn-handler:finalize-cudnn-handler (cudnn-handler backend)))
  ;(let ((context? nil (backend-context backend)))
    ;(when context?
      ;(cl-cuda:destroy-cuda-context context?))))

(defmethod petalisp.core:replace-lazy-array ((instance lazy-array) (replacement cuda-immediate))
  (change-class instance (class-of replacement)
    :storage (cuda-immediate-storage replacement)
    :ntype (petalisp.core:element-ntype replacement)
    :shape (array-shape replacement)))
