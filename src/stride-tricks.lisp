(defpackage :petalisp-cuda.stride-tricks
  (:use :cl
        :petalisp
        :petalisp.core
        :petalisp-cuda.memory.cuda-array)
  (:import-from :petalisp-cuda.memory.cuda-array
                :%make-cuda-array)
  (:export :transform-cuda-array))
(in-package :petalisp-cuda.stride-tricks)

(defun transform-cuda-array (cuda-array transformation)
  (assert (loop for offset across (transformation-offsets transformation)
                for mask across (transformation-output-mask transformation)
                always (or (= offset 0) (not mask))))
  (%make-cuda-array :memory-block (cuda-array-memory-block cuda-array)
                    :strides (map 'list
                                  (lambda (x mask)
                                    (if mask x 0))
                                  (transform (cuda-array-strides cuda-array) transformation)
                                  (transformation-output-mask transformation))
                    :shape (transform (cuda-array-shape cuda-array) transformation)))
