(defpackage petalisp-cuda.memory.memory-pool
  (:use :cl)
  (:use :hash-set)
  (:import-from :petalisp.native-backend :memory-pool-allocate
                                         :memory-pool-free
                                         :memory-pool
                                         :memory-pool-reset
                                         :array-table)
  (:import-from :petalisp-cuda.memory.cuda-array :memory-block-device-ptr
                                                 :cuda-array-memory-block)
  (:export :make-cuda-memory-pool
           :memory-pool-allocate
           :memory-pool-free
           :memory-pool
           :memory-pool-reset
           :reclaim-cuda-memory))

(in-package petalisp-cuda.memory.memory-pool)

(defun pointer-equality (a b)
  (= (memory-block-device-ptr a) (memory-block-device-ptr b)))

(defclass cuda-memory-pool (memory-pool)
  ((%allocated-arrays :accessor allocated-cuda-arrays
                      :initform (make-hash-set))
   (%lock :accessor cuda-memory-pool-lock
          :initform (bt:make-lock))))

(defun make-cuda-memory-pool ()
  (make-instance 'cuda-memory-pool))

(defmethod memory-pool-allocate ((memory-pool cuda-memory-pool)
                                 (array-element-type t)
                                 (array-size number))
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (or (pop (gethash (cons array-element-type array-size)
                      (array-table memory-pool)))
        (let ((array
                (cl-cuda.api.memory::%make-memory-block :device-ptr (cl-cuda:alloc-device-memory array-element-type
                                                                                                 array-size)
                                                        :host-ptr (cffi:null-pointer)
                                                        :type array-element-type
                                                        :size array-size)))
          (hs-ninsert (allocated-cuda-arrays memory-pool) array)
          array))))

(defmethod memory-pool-free ((memory-pool cuda-memory-pool)
                             (array cl-cuda.api.memory::memory-block))
  (when (hs-memberp (allocated-cuda-arrays memory-pool) array)
    (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
      (pushnew array (gethash (cons (cl-cuda:memory-block-type array)
                                    (cl-cuda:memory-block-size array))
                              (array-table memory-pool))
               :test #'pointer-equality)
      (values))))

(defmethod memory-pool-free ((memory-pool cuda-memory-pool)
                             (array petalisp-cuda.memory.cuda-array:cuda-array))
   (memory-pool-free memory-pool (petalisp-cuda.memory.cuda-array:cuda-array-memory-block array)))

(defmethod memory-pool-reset :before ((memory-pool cuda-memory-pool))
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (hs-map #'cl-cuda:free-memory-block (allocated-cuda-arrays memory-pool))
    (hs-nremove-if (lambda (item) (declare (ignore item)) t) (allocated-cuda-arrays memory-pool))))

(defmethod reclaim-cuda-memory ((memory-pool cuda-memory-pool))
  (hs-map (lambda (memory-block) (memory-pool-free memory-pool memory-block)) (allocated-cuda-arrays memory-pool)))
