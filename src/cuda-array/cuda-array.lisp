(defpackage petalisp-cuda.cuda-array
  (:use :cl
        :iterate)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:export :make-cuda-array
           :cuda-array
           :free-cuda-array
           :copy-memory-block-to-lisp
           :copy-cuda-array-to-lisp
           :device-ptr
           :element-type))

(in-package :petalisp-cuda.cuda-array)

; TODO rename cuda-nd-array

; TODO: generalize to (memory-block memory-layout) ?
(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)

(defun mem-layout-from-shape (shape &optional strides)
  (let* ((strides (or strides
                      (reverse (iter (for element in (reverse shape))
                                 (accumulate element by #'* :initial-value 1 into acc)
                                 (collect (/ acc element))))))
         (size (reduce #'max (mapcar #'* strides shape)
                       ; even with all-zeros strides we need at least one element
                       :initial-value 1)))
    (values size strides)))

(defun make-cuda-array (shape dtype &optional strides)
  (multiple-value-bind (size strides) (mem-layout-from-shape shape strides)
    (%make-cuda-array :memory-block (cl-cuda:alloc-memory-block dtype size)
                      :shape shape
                      :strides strides)))

(defun free-cuda-array (array)
  (progn
    (cl-cuda:free-memory-block (slot-value array 'memory-block))
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

(defun type-specific-strides (array)
  (mapcar (lambda (s) (* s (element-size array))) (slot-value array 'strides)))

(defun copy-cuda-array-to-lisp (array)
  (let ((memory-block (slot-value array 'memory-block))
        (shape (slot-value array 'shape)))
    (progn
      (cl-cuda:sync-memory-block memory-block :device-to-host)
      (aops:generate (lambda (indices) (cuda-array-aref array indices)) shape :subscripts))))
