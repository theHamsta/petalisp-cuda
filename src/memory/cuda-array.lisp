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
           :free-cuda-array
           :copy-memory-block-to-lisp
           :copy-cuda-array-to-lisp
           :device-ptr
           :nd-iter
           :cuda-array-p
           :element-type))

(in-package :petalisp-cuda.memory.cuda-array)

; TODO rename my shape to dimenstions

; TODO rename cuda-nd-array? cu-ndarry?
; TODO: generalize to (memory-block memory-layout) ?
(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)

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

;(defgeneric cuda-array-p (object))
;(defmethod cuda-array-p ((object cuda-array))
  ;t)
;(defmethod cuda-array-p ((object t))
  ;nil)


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
  (length (slot-value array 'shape)))

(defun copy-cuda-array-to-lisp (array)
  (let ((memory-block (slot-value array 'memory-block))
        (shape (slot-value array 'shape)))
    (progn
      (cl-cuda:sync-memory-block memory-block :device-to-host)
      (aops:generate* (petalisp-cuda.backend:lisp-type-cuda-array array) (lambda (indices) (cuda-array-aref array indices)) shape :subscripts))))

(defun copy-lisp-to-cuda-array (lisp-array cuda-array)
  (let ((memory-block (slot-value cuda-array 'memory-block))
        (cuda-shape (slot-value cuda-array 'shape)))
    (progn
      (assert (equalp (array-dimensions lisp-array) cuda-shape))
      ;; TODO: probably very slow
      (iterate (for i in-it (petalisp-cuda.memory.cuda-array:nd-iter cuda-shape))
        (let ((args i))
          (progn
            (push lisp-array args)
            (set-cuda-array-aref cuda-array i (apply #'aref args))))))
      (cl-cuda:sync-memory-block memory-block :host-to-device)
      cuda-array))

(defun mem-layout-from-shape (shape &optional strides)
  (let* ((strides (or strides
                      (reverse (iter (for element in (reverse shape))
                                 (accumulate element by #'* :initial-value 1 into acc)
                                 (collect (/ acc element))))))
         (size (reduce #'max (mapcar #'* strides shape)
                       ; even with all-zeros strides we need at least one element
                       :initial-value 1)))
    (values size strides)))


(defmethod petalisp.core:shape ((array cuda-array))
  (let* ((shape (cuda-array-shape array))
         (rank (length shape)))
    (petalisp.core::%make-shape
      (mapcar (lambda (s) (petalisp.core:range 0 1 (1- s))) shape)
      rank)))
