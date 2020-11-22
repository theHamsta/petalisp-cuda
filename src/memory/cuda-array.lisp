(defpackage petalisp-cuda.memory.cuda-array
  (:use :cl
        :iterate
        :cl-itertools)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :cl-cuda
                :memory-block-device-ptr
                :memory-block-host-ptr
                :memory-block-type
                :memory-block-size)
  (:import-from :petalisp.core
                :rank)
  (:import-from :alexandria :if-let)
  (:import-from :petalisp-cuda.utils.cl-cuda
                :sync-memory-block-async)
  (:import-from :petalisp-cuda.options
                :*max-array-printing-length*
                :*silence-cl-cuda*
                :*page-locked-host-memory*)
  (:export :make-cuda-array
           :cuda-array
           :cuda-array-shape
           :cuda-array-type
           :cuda-array-device
           :cuda-array-from-lisp
           :cuda-array-memory-block
           :free-cuda-array
           :copy-memory-block-to-lisp
           :copy-cuda-array-to-lisp
           :device-ptr
           :nd-iter
           :cuda-array-p
           :type-from-cl-cuda-type
           :lisp-type-from-cl-cuda-type
           :element-type))

(in-package :petalisp-cuda.memory.cuda-array)

(defun type-from-cl-cuda-type (element-type)
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
    (t (error "Cannot convert ~S to ntype." element-type))))

(defun lisp-type-cuda-array (cu-array)
  (type-from-cl-cuda-type (cuda-array-type cu-array)))

; TODO: generalize to (memory-block memory-layout) ?
(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)

(declaim (inline nd-iter))
(defiter nd-iter (shape)
  (let* ((ndim (length shape))
         (cur (mapcar (lambda (x) (* 0 x)) shape))
         (last-idx (1- ndim))
         (last-element (mapcar #'1- shape))) 
    (loop do (progn
               (yield cur)
               (loop for i from last-idx downto 0 
                     do (progn
                          (incf (nth i cur))
                          (if (= (nth i cur) (nth i shape))
                              (setf (nth i cur) 0)
                              (loop-finish))))
               (when (equal last-element cur)
                 (progn
                   (yield cur)
                   (loop-finish)))))))

;TODO: redo this with allocator type
(defgeneric make-cuda-array (shape dtype &optional strides alloc-function)
  (:method ((array cuda-array) dtype &optional strides alloc-function)
    (cuda-array-from-cuda-array array dtype strides alloc-function))
  (:method ((array array) dtype &optional strides alloc-function)
    (cuda-array-from-lisp array dtype strides alloc-function))
  ;; from raw shape
  (:method ((shape list) dtype &optional strides alloc-function)
    (let ((alloc-function (or alloc-function
                              #'cl-cuda:alloc-memory-block))
          (alignment (alexandria:switch (dtype :test #'equal)
                       (:half 2))))
      (multiple-value-bind (size strides) (mem-layout-from-shape shape strides alignment)
        (%make-cuda-array :memory-block (funcall alloc-function dtype (max size 1))
                          :shape shape
                          :strides strides))))
  ;; from raw petalisp:shape
  (:method ((shape petalisp:shape) dtype &optional strides alloc-function)
    (let ((dimensions (mapcar #'petalisp:range-size (petalisp:shape-ranges shape))))
      (make-cuda-array dimensions dtype strides alloc-function))))

(defun cuda-array-from-lisp (lisp-array dtype &optional strides alloc-function)
  (let* ((shape (array-dimensions lisp-array))
         (cuda-array (make-cuda-array shape dtype strides alloc-function)))
    (copy-lisp-to-cuda-array lisp-array cuda-array)))

(defun cuda-array-from-cuda-array (cuda-array dtype &optional strides alloc-function)
  (let* ((shape (cuda-array-shape cuda-array))
         (strides (cuda-array-strides cuda-array))
         (size (cuda-array-size cuda-array))
         (new-cuda-array (make-cuda-array shape dtype strides alloc-function))
         (from-ptr (memory-block-device-ptr (cuda-array-memory-block cuda-array)))
         (to-ptr (memory-block-device-ptr (cuda-array-memory-block new-cuda-array))))
    ;; TODO: memcpy3d in order to change layout?
    (assert (= 0 (petalisp-cuda.cudalibs::cuMemcpyDtoDAsync_v2 to-ptr
                                                               from-ptr
                                                               (cffi:make-pointer (* (cuda-array-size cuda-array)
                                                                                     (cffi-type-size dtype)))
                                                               cl-cuda:*cuda-stream*)))
    new-cuda-array))

(defun cuda-array-size (cuda-array)
    (memory-block-size (cuda-array-memory-block cuda-array)))

(defun free-cuda-array (array &optional free-function)
  ;TODO: redo this with allocator type
  (let ((free-function (or free-function
                           #'cl-cuda:free-memory-block)))
    (funcall free-function (cuda-array-memory-block array))
    (setf (cuda-array-memory-block array) nil)))


(defun cuda-array-aref (array indices)
  (let ((memory-block (slot-value array 'memory-block))
        (strides (slot-value array 'strides)))
    (cl-cuda:memory-block-aref memory-block (reduce #'+ (mapcar #'* indices strides)))))

(defun set-cuda-array-aref (array indices value)
  (let ((memory-block (slot-value array 'memory-block))
        (strides (slot-value array 'strides)))
    (setf (cl-cuda:memory-block-aref memory-block (reduce #'+ (mapcar #'* indices strides))) value)))

(defun cuda-array-type (array)
  (cffi-type (cl-cuda:memory-block-type (slot-value array 'memory-block))))


(defun device-ptr (array)
  (cl-cuda:memory-block-device-ptr (slot-value array 'memory-block)))


(defun element-size (array)
  (cffi-type-size (cuda-array-type array)))


(defun raw-memory-strides (array)
  (mapcar (lambda (s) (* s (element-size array))) (slot-value array 'strides)))


(defmethod petalisp.core:rank ((array cuda-array))
  (length (cuda-array-shape array)))

(defun can-do-dark-pointer-magic-p (cuda-array &optional lisp-array)
  #+sbcl
  (let* ((lisp-array-type (if lisp-array
                              (array-element-type lisp-array)
                              (lisp-type-cuda-array cuda-array))))
    (and
      (c-layout-p cuda-array) ; at least the host array should have c-layout
      (case lisp-array-type
        (single-float t)
        (double-float t)
        ((signed-byte 32) (= 4 (element-size cuda-array)))))))

;; TODO: add with-host-memory ensured to only temporarily add host memory and re-use a common host-mem staging area?
(defun host-alloc (element-type size)
  (if *page-locked-host-memory*
      (cffi:with-foreign-object (ptr '(:pointer (:pointer :void)))
        (assert (= 0 (petalisp-cuda.cudalibs::cuMemAllocHost_v2 ptr (cffi:make-pointer (* size (cffi-type-size element-type))))))
        (cffi:mem-ref ptr :pointer))
      (cl-cuda:alloc-host-memory element-type size)))

(defun ensure-host-memory (cuda-array)
  (let ((memory-block (cuda-array-memory-block cuda-array)))
    (when (cffi:null-pointer-p (memory-block-host-ptr memory-block))
      (setf (cuda-array-memory-block cuda-array)
            (cl-cuda.api.memory::%make-memory-block :device-ptr (memory-block-device-ptr memory-block)
                                                    :host-ptr (host-alloc (memory-block-type memory-block)
                                                                          (memory-block-size memory-block))
                                                    :type (memory-block-type memory-block)
                                                    :size (memory-block-size memory-block))))))

(defun copy-cuda-array-to-lisp (cuda-array)
  (declare (optimize (debug 0)(speed 3)(safety 0)))
  (ensure-host-memory cuda-array)
  (let ((memory-block (cuda-array-memory-block cuda-array))
        (shape (cuda-array-shape cuda-array))
        (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
    (if shape
        (if (c-layout-p cuda-array) 
            (if (can-do-dark-pointer-magic-p cuda-array)
                ;; c-layout and sbcl: pin array and cudaMemcpy from Lisp
              (let ((lisp-array (make-array (cuda-array-shape cuda-array) :element-type (lisp-type-cuda-array cuda-array))))
                #+sbcl
                (sb-sys:with-pinned-objects ((sb-ext:array-storage-vector lisp-array))
                  (let ((alien (sb-sys:vector-sap (sb-ext:array-storage-vector lisp-array))))
                    (let* ((new-memory-block (cl-cuda.api.memory::%make-memory-block :device-ptr (memory-block-device-ptr memory-block)
                                                                                     :host-ptr alien
                                                                                     :type (memory-block-type memory-block)
                                                                                     :size (memory-block-size memory-block))))
                      ;; not aync since we pinning the lisp array
                      (cl-cuda:sync-memory-block new-memory-block :device-to-host))))
                lisp-array)
              ;; c-layout: cffi-package
              (progn
                (cl-cuda:sync-memory-block memory-block :device-to-host)
                (cffi:foreign-array-to-lisp (cl-cuda:memory-block-host-ptr (cuda-array-memory-block cuda-array)) `(:array ,(cuda-array-type cuda-array) ,@(cuda-array-shape cuda-array)))))
          ;; No c-layout: slow generate
          (progn
            (cl-cuda:sync-memory-block memory-block :device-to-host)
            (aops:generate* (lisp-type-cuda-array cuda-array) (lambda (indices) (cuda-array-aref cuda-array indices)) shape :subscripts)))
        (progn 
          (cl-cuda:sync-memory-block memory-block :device-to-host)
          (cuda-array-aref cuda-array '(0))))))

(defun cuda-array-device (cuda-array)
  "Returns the device index on which the array was allocated"
  (let ((ptr (memory-block-device-ptr (cuda-array-memory-block cuda-array))))
   (cffi:with-foreign-object (data '(:pointer :int))
    (assert (= 0
               (petalisp-cuda.cudalibs::cuPointerGetAttribute data
                                                              (cffi:foreign-enum-value 'petalisp-cuda.cudalibs::cupointer-attribute-enum
                                                                                       :cu-pointer-attribute-device-ordinal)
                                                              ptr)))
     (cffi:mem-ref data :int))))

(defun copy-lisp-to-cuda-array-slow-fallback (lisp-array cuda-array)
  (declare (optimize (debug 0)(speed 3)(safety 0)))
  (ensure-host-memory cuda-array)
  (iterate (for idx in-it (petalisp-cuda.memory.cuda-array:nd-iter (cuda-array-shape cuda-array)))
           (set-cuda-array-aref cuda-array idx (if (equal t (array-element-type lisp-array))
                                                   (coerce (apply #'aref `(,lisp-array ,@idx)) 'single-float)
                                                   (apply #'aref `(,lisp-array ,@idx))))))

(defun copy-lisp-to-cuda-array (lisp-array cuda-array)
  (let ((dark-pointer-magic-p (can-do-dark-pointer-magic-p cuda-array lisp-array)))
    (unless dark-pointer-magic-p
      (ensure-host-memory cuda-array))
    (let ((memory-block (cuda-array-memory-block cuda-array))
          (cuda-shape (cuda-array-shape cuda-array))
          (cl-cuda:*show-messages* (unless *silence-cl-cuda* cl-cuda:*show-messages*)))
      (assert (equalp (array-dimensions lisp-array) cuda-shape))
      (if (c-layout-p cuda-array)
          (handler-case 
              ;; dirty internals
            (if dark-pointer-magic-p
              #-sbcl (error "dark-pointer-magic-p is always nil without SBCL")
              #+sbcl
              (sb-sys:with-pinned-objects ((sb-ext:array-storage-vector lisp-array))
                (let ((alien (sb-sys:vector-sap (sb-ext:array-storage-vector lisp-array))))
                  (let* ((mem-block (cuda-array-memory-block cuda-array))
                         (new-memory-block (cl-cuda.api.memory::%make-memory-block :device-ptr (memory-block-device-ptr mem-block)
                                                                                   :host-ptr alien
                                                                                   :type (memory-block-type mem-block)
                                                                                   :size (memory-block-size mem-block))))
                    ;; not aync since we pinning the lisp array
                    (cl-cuda:sync-memory-block new-memory-block :host-to-device))))
              ;; copy to foreign
              (cffi:lisp-array-to-foreign lisp-array
                                          (cl-cuda:memory-block-host-ptr (cuda-array-memory-block cuda-array))
                                          `(:array ,(cuda-array-type cuda-array) ,@(cuda-array-shape cuda-array))))
            (type-error (e)
              (declare (ignore e))
              (copy-lisp-to-cuda-array-slow-fallback lisp-array cuda-array)))
          (copy-lisp-to-cuda-array-slow-fallback lisp-array cuda-array))
      (unless dark-pointer-magic-p
        (sync-memory-block-async memory-block :host-to-device))
      cuda-array)))

(defun round-up (number multiple)
  (+ number (rem number multiple)) )

(defun mem-layout-from-shape (shape &optional strides alignment)
  (c-mem-layout-from-shape shape strides alignment))

(defun c-mem-layout-from-shape (shape &optional strides (alignment 1))
  (let* ((strides (or strides
                      (reverse (iter (for element in (reverse shape))
                                 (accumulate element by #'* :initial-value 1 into acc)
                                 (collect (if (= acc element)
                                              1
                                              (round-up (/ acc element) alignment)))))))
         (size (reduce #'max (mapcar #'* strides shape)
                            ; even with all-zeros strides we need at least one element
                            :initial-value 1)))
    (values (round-up size alignment) strides)))

(defun c-layout-p (cuda-array)
  (multiple-value-bind (size strides) (c-mem-layout-from-shape (cuda-array-shape cuda-array))
    (declare (ignore size))
    (equalp strides (cuda-array-strides cuda-array))))

(defmethod petalisp.core:shape ((array cuda-array))
  (let* ((shape (cuda-array-shape array)))
    (petalisp.core::make-shape
      (mapcar (lambda (s) (petalisp.core:range s)) shape))))

(defmethod print-object :after ((cuda-array cuda-array) stream)
  (unless (or *print-readably* (= *max-array-printing-length* 0))
    (ensure-host-memory cuda-array)
    (let  ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
      (cl-cuda:sync-memory-block (cuda-array-memory-block cuda-array) :device-to-host)
      (let* ((shape (cuda-array-shape cuda-array))
             (rank (length shape))
             (max-idx (mapcar #'1- shape))
             (max-border (mapcar (lambda (i) (- i *max-array-printing-length*)) shape)))
        (format stream "~%~%")
        (if (= rank 0)
            (format stream "~A~%" (cuda-array-aref cuda-array '(0)))
            (iterate (for idx in-it (petalisp-cuda.memory.cuda-array:nd-iter shape))
              (when (every (lambda (i max) (or (<= i *max-array-printing-length*) (> i max))) idx max-border)
                (dotimes (i rank)
                  (when (every (lambda (i) (= i 0)) (subseq idx i))
                    (format stream "(")))
                  (if (some (lambda (i) (= i *max-array-printing-length*)) idx)
                      (format stream "... ")
                      (format stream "~A " (cuda-array-aref cuda-array idx)))
                  (dotimes (i rank)
                    (when (every (lambda (i max) (= i max)) (subseq idx i) (subseq max-idx i))
                      (format stream ")")))
                (when (= (car (last idx)) (1- (car (last shape))))
                  (format stream "~%")))))))))
