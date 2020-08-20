(defpackage petalisp-cuda.memory.cuda-array
  (:use :cl
        :iterate
        :cl-itertools)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :petalisp.core :rank)
  (:export :make-cuda-array
           :cuda-array
           :cuda-array-shape
           :cuda-array-type
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

(defparameter *silence-cl-cuda* t)

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
  (:method ((array array) dtype &optional strides alloc-function)
    (cuda-array-from-lisp array dtype strides alloc-function))
  ;; from raw shape
  (:method ((shape list) dtype &optional strides alloc-function)
    (let ((alloc-function (or alloc-function
                              #'cl-cuda:alloc-memory-block)))
      (multiple-value-bind (size strides) (mem-layout-from-shape shape strides)
        (%make-cuda-array :memory-block (funcall alloc-function dtype size)
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

(defun copy-cuda-array-to-lisp (array)
  (declare (optimize (debug 0)(speed 3)(safety 3)))
  (let ((memory-block (slot-value array 'memory-block))
        (shape (slot-value array 'shape))
        (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
    (cl-cuda:sync-memory-block memory-block :device-to-host)
    (if (c-layout-p array) 
        (cffi:foreign-array-to-lisp (cl-cuda:memory-block-host-ptr (cuda-array-memory-block array)) `(:array ,(cuda-array-type array) ,@(cuda-array-shape array)))
        (aops:generate* (lisp-type-cuda-array array) (lambda (indices) (cuda-array-aref array indices)) shape :subscripts))))

(defun copy-lisp-to-cuda-array-slow-fallback (lisp-array cuda-array)
  (declare (optimize (debug 0)(speed 3)(safety 0)))
  (iterate (for idx in-it (petalisp-cuda.memory.cuda-array:nd-iter (cuda-array-shape cuda-array)))
    (set-cuda-array-aref cuda-array idx (if (equal t (array-element-type lisp-array))
                                            (coerce (apply #'aref `(,lisp-array ,@idx)) 'single-float)
                                            (apply #'aref `(,lisp-array ,@idx))))))

(defun copy-lisp-to-cuda-array (lisp-array cuda-array)
  (let ((memory-block (slot-value cuda-array 'memory-block))
        (cuda-shape (slot-value cuda-array 'shape))
        (cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
    (assert (equalp (array-dimensions lisp-array) cuda-shape))
    (if (c-layout-p cuda-array)
        (handler-case (cffi:lisp-array-to-foreign lisp-array (cl-cuda:memory-block-host-ptr (cuda-array-memory-block cuda-array)) `(:array ,(cuda-array-type cuda-array) ,@(cuda-array-shape cuda-array)))
          (type-error (e)
            (declare (ignore e))
            (copy-lisp-to-cuda-array-slow-fallback lisp-array cuda-array)))
          (copy-lisp-to-cuda-array-slow-fallback lisp-array cuda-array))
        (cl-cuda:sync-memory-block memory-block :host-to-device))
    cuda-array)

(defun mem-layout-from-shape (shape &optional strides)
  (c-mem-layout-from-shape shape strides))

(defun c-mem-layout-from-shape (shape &optional strides)
  (let* ((strides (or strides
                      (reverse (iter (for element in (reverse shape))
                                 (accumulate element by #'* :initial-value 1 into acc)
                                 (collect (/ acc element))))))
         (size (reduce #'max (mapcar #'* strides shape)
                       ; even with all-zeros strides we need at least one element
                       :initial-value 1)))
    (values size strides)))

(defun c-layout-p (cuda-array)
  (multiple-value-bind (size strides) (c-mem-layout-from-shape (cuda-array-shape cuda-array))
    (declare (ignore size))
    (equalp strides (cuda-array-strides cuda-array))))

(defmethod petalisp.core:shape ((array cuda-array))
  (let* ((shape (cuda-array-shape array))
         (rank (length shape)))
    (petalisp.core::%make-shape
      (mapcar (lambda (s) (petalisp.core:range 0 (1- s))) shape)
      rank)))

(defparameter *max-printing-length* 5)

(defmethod print-object :after ((cuda-array cuda-array) stream)
  (let  ((cl-cuda:*show-messages* (if *silence-cl-cuda* nil cl-cuda:*show-messages*)))
    (cl-cuda:sync-memory-block (cuda-array-memory-block cuda-array) :device-to-host)
    (let* ((shape (cuda-array-shape cuda-array))
           (rank (length shape))
           (max-idx (mapcar #'1- shape))
           (max-border (mapcar (lambda (i) (- i *max-printing-length*)) shape)))
      (format stream "~%~%")
      (iterate (for idx in-it (petalisp-cuda.memory.cuda-array:nd-iter shape))
               (when (every (lambda (i max) (or (<= i *max-printing-length*) (> i max))) idx max-border)
                 (dotimes (i rank)
                   (when (every (lambda (i) (= i 0)) (subseq idx i))
                     (format stream "(")))
                   (if (some (lambda (i) (= i *max-printing-length*)) idx)
                       (format stream "... ")
                       (format stream "~A " (cuda-array-aref cuda-array idx)))
                   (dotimes (i rank)
                     (when (every (lambda (i max) (= i max)) (subseq idx i) (subseq max-idx i))
                       (format stream ")")))
                 (when (= (car (last idx)) (1- (car (last shape))))
                   (format stream "~%")))))))
