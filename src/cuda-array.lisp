(defpackage petalisp-cuda.cuda-array
  (:use :cl
        :iterate)
  (:import-from :cl-cuda)
  (:export :make-cuda-array
           :free-cuda-array))
(in-package :petalisp-cuda.cuda-array)

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
                      :strides strides
                      )))

(defun free-cuda-array (array)
  (progn
    (cl-cuda:free-memory-block (slot-value array 'memory-block))
    (setf (slot-value array 'memory-block) nil)
    ))
