(in-package petalisp-cuda.custom-op)


(defclass lazy-convolution (lazy-custom-op)
  ())

;;outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;

(defun lazy-convolution (input filter)
  (let* ((input (lazy-array input))
         (filter (lazy-array filter))
         (shape (lazy-array-shape input)))
    (make-instance 'lazy-convolution
                   :inputs (list input filter)
                   :shape shape
                   :ntype (element-ntype input)
                   :number-of-values 1)))


;(petalisp-cuda.cudnn-handler::cudnn-convolution x
                                                ;w
                                                ;y
                                                ;(petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
