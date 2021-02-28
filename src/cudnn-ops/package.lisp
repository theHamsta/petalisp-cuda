(defpackage petalisp-cuda.cudnn-ops
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp.core
        :petalisp-cuda.backend
        :petalisp-cuda.custom-op
        :petalisp-cuda.stride-tricks
        :petalisp-cuda.options)
  (:import-from :petalisp.core
                :substitute-array)
  (:export :lazy-convolution
           :lazy-reduction))
(in-package petalisp-cuda.cudnn-ops)

(defun unnormalizing-transformation (input-shape output-shape)
  (let ((output-shape (transform output-shape (collapsing-transformation output-shape))))
    (if (equalp input-shape output-shape)
        (identity-transformation (shape-rank input-shape))
        (invert-transformation (normalizing-transformation output-shape)))))

