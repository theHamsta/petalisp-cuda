(defpackage petalisp-cuda.memory.cuda-array
  (:use :cl
        :iterate)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :petalisp.core :rank)
  (:export :make-cuda-array
           :cuda-array
           :free-cuda-array
           :copy-memory-block-to-lisp
           :copy-cuda-array-to-lisp
           :device-ptr
           :element-type))

(in-package :petalisp-cuda.memory.cuda-array)

; TODO rename my shape to dimenstions

; TODO rename cuda-nd-array? cu-ndarry?
; TODO: generalize to (memory-block memory-layout) ?
(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)


  ;TODO: redo this with allocator type
(defgeneric make-cuda-array (shape dtype &optional strides alloc-function)
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


(defun free-cuda-array (array &optional free-function)
  ;TODO: redo this with allocator type
  (let ((free-function (or free-function
                           #'cl-cuda:free-memory-block)))
    (funcall free-function (slot-value array 'memory-block))
    (setf (slot-value array 'memory-block) nil)))


(defun cuda-array-aref (array indices)
  (let ((memory-block (slot-value array 'memory-block))
            (strides (slot-value array 'strides)))
        (cl-cuda:memory-block-aref memory-block (reduce #'+ (mapcar #'* indices strides)))))


(defun element-type (array)
  (cffi-type (cl-cuda:memory-block-type (cl:slot-value array 'memory-block))))


(defun device-ptr (array)
  (cl-cuda:memory-block-device-ptr (cl:slot-value array 'memory-block)))


(defun element-size (array)
  (cffi-type-size (element-type array)))


(defun raw-memory-strides (array)
  (mapcar (lambda (s) (* s (element-size array))) (slot-value array 'strides)))


(defmethod petalisp.core:rank ((array cuda-array))
  (length (slot-value array 'shape)))


(defun copy-cuda-array-to-lisp (array)
  (let ((memory-block (slot-value array 'memory-block))
        (shape (slot-value array 'shape)))
    (progn
      (cl-cuda:sync-memory-block memory-block :device-to-host)
      (aops:generate (lambda (indices) (cuda-array-aref array indices)) shape :subscripts))))


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
  (petalisp.core:make-shape
       (mapcar (lambda (s) (petalisp.core:range 0 1 (1- s))) (slot-value array 'strides))))
