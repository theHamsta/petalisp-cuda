(defpackage petalisp-cuda.cuda-array
  (:use :cl
        :iterate)
  (:export :make-cuda-array
           :cuda-array
           :free-cuda-array
           :copy-memory-block-to-lisp
           :copy-cuda-array-to-lisp))

(in-package :petalisp-cuda.cuda-array)

; TODO rename cuda-nd-array

; TODO: generalize to (memory-block memory-layout) ?
(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)

(defun mem-layout-from-shape (shape &optional strides)
  (let* ((strides (or strides
                      (reverse (iter (for element in shape)
                                 (accumulate element by #'* :initial-value 1 into acc)
                                 (collect (/ acc element))))))
         (size (reduce #'max (mapcar #'* strides shape))))
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

(defun memory-block-aref (indices)
  (let ((memory-block (slot-value array 'memory-block))
            (strides (slot-value array 'strides)))
        (memory-block-aref memory-block (reduce #'+ (mapcar #'* indices strides)))))

(defun copy-memory-block-to-lisp (memory-block)
  (progn
      (cl-cuda:sync-memory-block memory-block :device-to-host)
      (array-operations:generate (aops:generate (lambda (indices) (cuda-aref memory-block indices)) :subscripts))))

(defun copy-cuda-array-to-lisp (array)
  (let ((memory-block (slot-value array 'memory-block)))
    (memory-block-to-lisp memory-block)))

