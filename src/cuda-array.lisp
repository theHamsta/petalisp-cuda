(in-package :betalisp)

(defstruct (cuda-array (:constructor %make-cuda-array))
  memory-block shape strides)

(defun mem-layout-from-shape (shape &optional strides)
  (let* ((chosen-strides (if strides
                             strides
                             (reverse (iter (for element in shape)
                                        (accumulate element by #'* :initial-value 1 into acc)
                                        (collect (/ acc element))))))
         (size (reduce #'max (mapcar #'* chosen-strides shape))))
    (values size chosen-strides)))

(defun make-cuda-array (shape dtype &optional strides)
  (multiple-value-bind (size chosen-strides) (mem-layout-from-shape shape strides)
    (%make-cuda-array :memory-block (cl-cuda:alloc-memory-block dtype size)
                      :shape shape
                      :strides chosen-strides
                      )))

  (defun free-cuda-array (array)
    (progn
      (cl-cuda:free-memory-block (slot-value array 'memory-block))
      (setq (slot-value array 'memory-block) nil)
      ))
