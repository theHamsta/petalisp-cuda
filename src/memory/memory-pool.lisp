
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
                      :initform '())))

(defun make-cuda-memory-pool ()
  (make-instance 'cuda-memory-pool))

(defmethod memory-pool-allocate ((memory-pool cuda-memory-pool)
                                 (array-element-type t)
                                 (array-dimensions number))
  (let ((array (or (pop (gethash (cons array-element-type array-dimensions)
                                 (array-table memory-pool)))(pop (gethash (cons array-element-type array-dimensions)
                                 (array-table memory-pool)))
                   (cl-cuda:alloc-memory-block array-element-type array-dimensions))))
    (push array (allocated-cuda-arrays memory-pool))
    array))

(defmethod memory-pool-free ((memory-pool cuda-memory-pool)
                             (array cl-cuda.api.memory::memory-block))
  (push array (gethash (cons (cl-cuda:memory-block-type array)
                             (cl-cuda:memory-block-size array))
                       (array-table memory-pool)))
  (values))

(defmethod memory-pool-reset :before ((memory-pool cuda-memory-pool))
  (mapcar #'cl-cuda:free-memory-block (allocated-cuda-arrays memory-pool)))


(defmethod reclaim-cuda-memory ((memory-pool cuda-memory-pool))
  (mapcar #'memory-pool-free (allocated-cuda-arrays memory-pool)))
