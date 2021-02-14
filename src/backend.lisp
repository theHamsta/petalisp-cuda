(defpackage :petalisp-cuda.backend
  (:use :cl
        :petalisp
        :petalisp-cuda.cuda-immediate
        :petalisp.ir
        :petalisp.core
        :petalisp-cuda.memory.memory-pool
        :petalisp-cuda.options)
  (:import-from :petalisp.core
                :identity-transformation)
  (:import-from :petalisp.native-backend
                :make-worker-pool
                :worker-pool-size
                :delete-worker-pool
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
           execute-kernel
           cuda-backend-event-map
           cuda-backend-predecessor-map
           *transfer-back-to-lisp*
           with-cuda-backend))
(in-package :petalisp-cuda.backend)

(defvar *cuda-backend* nil)

(defmacro with-cuda-backend-magic (backend &body body)
  `(let* ((cl-cuda:*cuda-context* (backend-context ,backend))
          (cl-cuda:*cuda-device* (backend-device-id ,backend))
          (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
          (cl-cuda.lang.built-in::+built-in-functions+ petalisp-cuda.cl-cuda-functions:+built-in-functions+)
          (cl-cuda.lang.type::+scalar-types+ petalisp-cuda.cl-cuda-functions:+scalar-types+)
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
   ;; TODO: refactor to use an allocator interface
   (memory-pool :initform (make-cuda-memory-pool)
                :accessor cuda-memory-pool)
   (device :initform nil
           :accessor backend-device)
   (device-id :initform nil
              :accessor backend-device-id)
   (%predecessor-map :initform (make-hash-table) :reader cuda-backend-predecessor-map :type hash-table)
   (%scheduler-queue :initform (lparallel.queue:make-queue) :reader cuda-backend-scheduler-queue)
   (%scheduler-thread :accessor cuda-backend-scheduler-thread)
   (worker-pool :initform (make-worker-pool (petalisp.utilities:number-of-cpus))
                :accessor cuda-backend-worker-pool)
   (%compile-cache :initform (make-hash-table :test #'equalp) :reader compile-cache :type hash-table)
   (event-map :initform (make-hash-table) :accessor cuda-backend-event-map)))

(defun cuda-backend-deallocate (backend buffer)
  (with-cuda-backend-magic backend
    (let ((storage (buffer-storage buffer))
          (predecessor-map (cuda-backend-predecessor-map backend)))
      (when (cuda-array-p storage)
        (setf (buffer-storage buffer) nil)
        (when (petalisp.ir:interior-buffer-p buffer)
          (alexandria:ensure-gethash storage predecessor-map (make-array 0 :fill-pointer 0))
          (push (gethash storage predecessor-map) buffer)
          (memory-pool-free (cuda-memory-pool backend) storage))))))

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
  (setf (backend-device backend) (petalisp-cuda.device:make-cuda-device cl-cuda:*cuda-device*))
  (let ((queue (cuda-backend-scheduler-queue backend)))
    (setf (cuda-backend-scheduler-thread backend)
          (bt:make-thread
           (lambda ()
             (loop for item = (lparallel.queue:pop-queue queue) do
               (if (functionp item)
                   (funcall item)
                   (loop-finish))))
           :name (format nil "~A scheduler thread"
                         (class-name (class-of backend)))))))

(defgeneric execute-kernel (kernel backend))

(defmethod backend-schedule
  ((backend cuda-backend)
   (lazy-arrays list)
   (finalizer function))
  (let ((promise (lparallel.promise:promise)))
    (lparallel.queue:push-queue
      (lambda ()
        (lparallel.promise:fulfill promise
                                   (funcall finalizer (backend-compute backend lazy-arrays))))
      (cuda-backend-scheduler-queue backend))
    promise))

(defmethod backend-wait
    ((backend cuda-backend)
     (promise t))
  (lparallel.promise:force promise))

(defmethod backend-compute ((backend cuda-backend) (lazy-arrays list))
  (let* ((memory-pool (cuda-memory-pool backend))
         (worker-pool (cuda-backend-worker-pool backend))
         (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*))
         (event-map (cuda-backend-event-map backend))
         (allocate (lambda (type size) (memory-pool-allocate memory-pool type size)))
         (deallocate (lambda (buffer) (cuda-backend-deallocate backend buffer)))
         (immediates (petalisp.scheduler:schedule-on-workers
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
                                                                                  allocate))))
                       ;; Deallocate.
                       deallocate)))
    (with-cuda-backend-magic backend
      (mapcar (alexandria:compose #'lazy-array
                                  (if *transfer-back-to-lisp* (lambda (immediate)
                                                                (let ((lisp-array (petalisp.core:array-from-immediate immediate)))
                                                                  (when (cuda-array-p (cuda-immediate-storage immediate))
                                                                    (memory-pool-free (cuda-memory-pool backend) (cuda-immediate-storage immediate)))
                                                                  lisp-array))
                                      #'cuda-immediate-storage))
              immediates))))

(defmethod petalisp.core:delete-backend ((backend cuda-backend))
  (with-accessors ((queue cuda-backend-scheduler-queue)
                   (thread cuda-backend-scheduler-thread)) backend
    (lparallel.queue:push-queue :quit queue)
    (bt:join-thread thread))
  (delete-worker-pool (cuda-backend-worker-pool backend))
  (petalisp-cuda.cudnn-handler:finalize-cudnn-handler (cudnn-handler backend))
  (memory-pool-reset (cuda-memory-pool backend)))
  ;(let ((context? (backend-context backend)))
    ;(when context?
      ;(cl-cuda:destroy-cuda-context context?))))
