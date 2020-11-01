
(defpackage petalisp-cuda.memory.memory-pool
  (:use :cl)
  (:import-from :petalisp.native-backend :memory-pool-allocate
                                         :memory-pool-free
                                         :memory-pool
                                         :memory-pool-reset
                                         :array-table)
  (:export :make-cuda-memory-pool
           :memory-pool-allocate
           :memory-pool-free
           :memory-pool
           :memory-pool-reset
           :reclaim-cuda-memory))

(in-package petalisp-cuda.memory.memory-pool)

(defclass cuda-memory-pool (memory-pool)
  ((%allocated-arrays :accessor allocated-cuda-arrays
                      :initform '())
   (%lock :accessor cuda-memory-pool-lock
          :initform (bt:make-lock))))

(defun make-cuda-memory-pool ()
  (make-instance 'cuda-memory-pool))

(defmethod memory-pool-allocate ((memory-pool cuda-memory-pool)
                                 (array-element-type t)
                                 (array-size number))
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (let ((array (or (pop (gethash (cons array-element-type array-size)
                                   (array-table memory-pool)))(pop (gethash (cons array-element-type array-size)
                                   (array-table memory-pool)))
                     (cl-cuda.api.memory::%make-memory-block :device-ptr (cl-cuda:alloc-device-memory array-element-type
                                                                                                      array-size)
                                                             :host-ptr (cffi:null-pointer)
                                                             :type array-element-type
                                                             :size array-size))))
      (push array (allocated-cuda-arrays memory-pool))
      array)))

(defmethod memory-pool-free ((memory-pool cuda-memory-pool)
                             (array cl-cuda.api.memory::memory-block))
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (push array (gethash (cons (cl-cuda:memory-block-type array)
                               (cl-cuda:memory-block-size array))
                         (array-table memory-pool)))
    (values)))

(defmethod memory-pool-free ((memory-pool cuda-memory-pool)
                             (array petalisp-cuda.memory.cuda-array:cuda-array))
   (memory-pool-free memory-pool (petalisp-cuda.memory.cuda-array:cuda-array-memory-block array)))

(defmethod memory-pool-reset :before ((memory-pool cuda-memory-pool))
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (mapcar #'cl-cuda:free-memory-block (allocated-cuda-arrays memory-pool))
    (setf (allocated-cuda-arrays memory-pool) '())))

(defmethod reclaim-cuda-memory ((memory-pool cuda-memory-pool))
  (mapcar (lambda (array) (memory-pool-free memory-pool array)) (allocated-cuda-arrays memory-pool)))
