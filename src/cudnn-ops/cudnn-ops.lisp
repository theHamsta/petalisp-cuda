(defpackage petalisp-cuda.cudnn-ops
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp.core
        :petalisp-cuda.backend
        :petalisp-cuda.custom-op)
  (:export :lazy-convolution))
(in-package petalisp-cuda.cudnn-ops)

(defclass lazy-convolution (lazy-custom-op)
  ())

(defclass lazy-reduction (lazy-custom-op)
  ((%reduction-operation 
    :initarg :reduction-operation
    :accessor lazy-reduction-reduction-operation
    :type (or function symbol))))

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

(defun lazy-reduction (input output-shape reduction-operation)
  (let* ((input (lazy-array input)))
    (make-instance 'lazy-reduction
                   :inputs (list input)
                   :shape output-shape
                   :ntype (element-ntype input)
                   :number-of-values 1
                   :reduction-operation reduction-operation)))

(defmethod lazy-custom-op-execute ((custom-op lazy-convolution)
                                    (backend cuda-backend)
                                    (input-buffers list)
                                    (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-convolution (buffer-storage (nth 0 input-buffers))
                                                  (buffer-storage (nth 1 input-buffers))
                                                  (buffer-storage (nth 0 output-buffers))
                                                  (petalisp-cuda.backend::cudnn-handler backend)))

(defmethod lazy-custom-op-execute ((custom-op lazy-reduction)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (buffer-storage (nth 0 input-buffers))
                                                   (buffer-storage (nth 0 output-buffers))
                                                   (lazy-reduction-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))
