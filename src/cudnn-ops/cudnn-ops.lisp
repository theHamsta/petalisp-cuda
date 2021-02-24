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
  ((%convolution-algorithm 
    :initarg :algorithm
    :accessor lazy-convolution-algorithm
    :type (or symbol null))
   (%paddings
    :initarg :paddings
    :accessor lazy-convolution-paddings
    :type (or integer list null))
   (%strides 
    :initarg :strides
    :accessor lazy-convolution-strides
    :type (or integer list null))
   (%dilations 
    :initarg :dilations
    :accessor lazy-convolution-dilations
    :type (or integer list null))
   (%group-count 
    :initarg :group-count
    :accessor lazy-convolution-group-count
    :type (or integer null))))

(defclass lazy-reduction (lazy-custom-op)
  ((%reduction-operation 
    :initarg :reduction-operation
    :accessor lazy-reduction-reduction-operation
    :type (or function symbol))))

(defun lazy-convolution (input filter &key algorithm strides paddings dilations group-count)
  (let* ((input (lazy-array input))
         (filter (lazy-array filter))
         (input-shape (lazy-array-shape input))
         (filter-shape (lazy-array-shape filter))
         (paddings (or paddings (mapcar (lambda (s) (floor s 2)) (subseq (mapcar #'range-size (shape-ranges filter-shape)) 2))))
         (strides (or strides (make-list (- (rank input) 2) :initial-element 1)))
         (dilations (or dilations (make-list (- (rank input) 2) :initial-element 1)))
         ;; Fromm cudnn doc:
         ;; outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;
         (output-shape (make-shape
                         `(,(shape-range input-shape 0)
                           ,(shape-range filter-shape 1)
                           ,@(loop for i in (mapcar #'range-size (subseq (shape-ranges input-shape) 2))
                                   for f in (mapcar #'range-size (subseq (shape-ranges filter-shape) 2))
                                   for stride in strides
                                   for pad in paddings
                                   for dilation in dilations
                                   collect (range
                                             (1+ (/ (+ i (* 2 pad) (- (1+ (* (1- f) dilation)))) stride))))))))
  (make-instance 'lazy-convolution
                 :inputs (list input filter)
                 :shape output-shape
                 :ntype (element-ntype input)
                 :number-of-batches (range-size (nth 0 (shape-ranges input-shape)))
                 :input-channels (range-size (nth 1 (shape-ranges input-shape)))
                 :output-channels (range-size (nth 1 (shape-ranges filter-shape)))
                 :algorithm algorithm
                 :dilations dilations
                 :strides strides
                 :paddings paddings
                 :group-count group-count)))

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
      (petalisp-cuda.backend::cudnn-handler backend)
      :algorithm (lazy-convolution-algorithm custom-op)
      :group-count (lazy-convolution-group-count custom-op)
      :paddings (lazy-convolution-paddings custom-op)
      :dilations (lazy-convolution-dilations custom-op)
      :filter-strides (lazy-convolution-strides custom-op))))

(defmethod lazy-custom-op-execute ((custom-op lazy-reduction)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (buffer-storage (nth 0 input-buffers))
                                                   (buffer-storage (nth 0 output-buffers))
                                                   (lazy-reduction-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))
