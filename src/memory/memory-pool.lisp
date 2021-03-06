(defpackage petalisp-cuda.memory.memory-pool
  (:use :cl
        :petalisp-cuda.cudalibs
        :cffi
        :cl-cuda.lang.type)
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
           :cuda-memory-pool-allocate
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
  (cuda-memory-pool-allocate memory-pool array-element-type array-size))

(defun alloc-device-memory-managed (element-type size)
  (with-foreign-object (ptr '(:pointer (:pointer :void)))
    (let ((result (cuMemAllocManaged ptr (make-pointer (* size (cffi-type-size element-type))) :cu-mem-attach-global)))
      (when (/= result 0)
        (error "Failed to allocate ~A MB of managed memory (~A)~%"
               (/ (* size (cffi-type-size element-type)) (* 1024.0 1024.0))
               (if (= result 2)
                 "out of memory"
                 result))))
    (pointer-address
      (mem-ref ptr :pointer))))

(defun cuda-memory-pool-allocate (memory-pool array-element-type array-size &key managedp)
  (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (or (pop (gethash (cons array-element-type array-size)
                      (array-table memory-pool)))
        (let ((array
                (cl-cuda.api.memory::%make-memory-block :device-ptr 
                                                        (if managedp
                                                            (alloc-device-memory-managed array-element-type array-size)
                                                            (cl-cuda:alloc-device-memory array-element-type array-size))
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
  (let ((cl-cuda:*show-messages* nil))
   (bt:with-lock-held ((cuda-memory-pool-lock memory-pool))
    (hs-map #'cl-cuda:free-memory-block (allocated-cuda-arrays memory-pool))
    (hs-nremove-if (lambda (item) (declare (ignore item)) t) (allocated-cuda-arrays memory-pool)))))

(defmethod reclaim-cuda-memory ((memory-pool cuda-memory-pool))
  (hs-map (lambda (memory-block) (memory-pool-free memory-pool memory-block)) (allocated-cuda-arrays memory-pool)))
