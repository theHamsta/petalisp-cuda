(defpackage petalisp-cuda.cudnn-ops
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp.core
        :petalisp-cuda.backend
        :petalisp-cuda.custom-op
        :petalisp-cuda.stride-tricks)
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
         (shape (lazy-array-shape input))
         (filter-shape (lazy-array-shape input)))
    (make-instance 'lazy-convolution
                   :inputs (list input filter)
                   :shape shape
                   :ntype (element-ntype input)
                   :number-of-batches (range-size (nth 0 (shape-ranges shape)))
                   :input-channels (range-size (nth 1 (shape-ranges shape)))
                   :output-channels (range-size (nth 1 (shape-ranges filter-shape))))))

(defun unnormalizing-transformation (input-shape output-shape)
  (let ((output-shape (transform output-shape (collapsing-transformation output-shape))))
    (if (equalp input-shape output-shape)
        (identity-transformation (shape-rank input-shape))
        (invert-transformation (normalizing-transformation output-shape)))))

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
  (let* ((input-buffer (nth 0 input-buffers))
         (filter-buffer (nth 1 input-buffers))
         (output-buffer (nth 0 output-buffers))
         (input (buffer-storage input-buffer))
         (filter (buffer-storage filter-buffer))
         (output (buffer-storage output-buffer)))
    (petalisp-cuda.cudnn-handler::cudnn-convolution
      (transform-cuda-array input
                            (unnormalizing-transformation (buffer-shape input-buffer)
                                                          (lazy-array-shape (nth 0 (lazy-array-inputs custom-op)))))
      (transform-cuda-array filter
                            (unnormalizing-transformation (buffer-shape filter-buffer)
                                                          (lazy-array-shape (nth 1 (lazy-array-inputs custom-op)))))
      (transform-cuda-array output
                            (unnormalizing-transformation (buffer-shape output-buffer)
                                                          (lazy-array-shape custom-op)))
      (petalisp-cuda.backend::cudnn-handler backend))))

(defmethod lazy-custom-op-execute ((custom-op lazy-reduction)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (buffer-storage (nth 0 input-buffers))
                                                   (buffer-storage (nth 0 output-buffers))
                                                   (lazy-reduction-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))
