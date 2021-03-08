(defpackage petalisp-cuda
  (:use :cl)
  (:import-from :petalisp-cuda.backend
                :use-cuda-backend
                :cuda-backend-p
                :with-cuda-backend
                :with-cuda-backend-raii
                :cuda-memory-pool
                :cudnn-handler
                :cuda-backend
                :*transfer-back-to-lisp*)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:import-from :petalisp-cuda.jit-execution
                :device-function
                :device-host-function)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:export :use-cuda-backend
           :reclaim-cuda-memory
           :*transfer-back-to-lisp*
           :with-cuda-backend
           :with-cuda-backend-raii
           :device-function
           :device-host-function
           :with-memory-usage-report))
(in-package :petalisp-cuda)

;; TODO: move these functions to a memory management file
(defun reclaim-cuda-memory (&optional (backend (or petalisp-cuda.backend::*cuda-backend* petalisp:*backend*)))
  (when (cuda-backend-p backend)
   (petalisp-cuda.memory.memory-pool:reclaim-cuda-memory (cuda-memory-pool backend))))

(defun reset-cuda-memory (&optional (backend (or petalisp-cuda.backend::*cuda-backend* petalisp:*backend*)))
  (when (cuda-backend-p backend)
    (petalisp-cuda.memory.memory-pool::memory-pool-reset (cuda-memory-pool backend))
    (petalisp-cuda.cudnn-handler:free-workspace-memory (cudnn-handler backend)))
  (values))

(defun reset-and-print-memory-report ()
  (petalisp-cuda::reclaim-cuda-memory)
  (format t "CUDNN workspace size: ~A MiB~%" (/ (petalisp-cuda.backend::cuda-backend-cudnn-workspace-size
                                                 petalisp-cuda.backend::*cuda-backend*) (* 1024.0 1024.0)))
  (format t "Memory pool total size: ~A MiB~%" (/ (loop for k being each hash-key in (petalisp-cuda.memory.memory-pool::array-table (cuda-memory-pool petalisp-cuda.backend::*cuda-backend*))
                                                       using (hash-value v)
                                                       sum (* (length v) (cdr k) (cffi-type-size (car k)))) (* 1024.0 1024.0)))
  (loop for k being each hash-key in (petalisp-cuda.memory.memory-pool::array-table (cuda-memory-pool petalisp-cuda.backend::*cuda-backend*))
        using (hash-value v)
        do (format t "~A x ~A kiB~%" (length v) (/ (* (cffi-type-size (car k)) (cdr k)) (* 1024.0)))))

(defmacro with-memory-usage-report (&body body)
  `(progn
     (petalisp-cuda::reset-cuda-memory)
     (let ((rtn (progn
                  ,@body)))
       (reset-and-print-memory-report)
       rtn)))
