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
    :type (or integer null))
   (%math-type
    :initarg :math-type
    :accessor lazy-convolution-math-type
    :type (or symbol null))
   (%convolution-mode
    :initarg :convolution-mode
    :accessor lazy-convolution-convolution-mode
    :type (or symbol null))))

(defclass lazy-convolution-backward-data (lazy-convolution)
  ())

(defclass lazy-convolution-backward-filter (lazy-convolution)
  ())

(defun get-convolution-output-shape (input-shape filter-shape strides paddings dilations &key transposedp)
  "Determines the shape of the output of a convolution

  From cudnn doc:
  outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride;"
  #| isympy
  In [3]: outputDim, inputDim, pad, filterDim, dilation, convolutionStride = symbols('outputDim, inputDim, pad, filterDim, dilation convolutionStride')

  In [4]: solve(outputDim - (1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride), inputDim)
  Out[4]: [convolutionStride⋅outputDim - convolutionStride + dilation⋅filterDim - dilation - 2⋅pad + 1]
  |#
  (unless (equalp (shape-range input-shape 1) (shape-range filter-shape (if transposedp 0 1)))
    (if transposedp
        (error "Input channels and filter input channels must be equal (first filter dimension must agree with second input dimension)!~%Input array: ~A,~%Filter array: ~A" input-shape filter-shape)
        (error "Input channels and filter input channels must be equal (both must agree in second dimension)!~%Input array: ~A,~%Filter array: ~A" input-shape filter-shape)))

  (unless (>= (rank filter-shape) 3)
    (error "Filter shape must be at least 3D (output channels, input channels, spatial dimensions...): ~A" filter-shape))

  (make-shape
    `(,(shape-range input-shape 0)
      ,(shape-range filter-shape (if transposedp 1 0))
      ,@(loop for i in (mapcar #'range-size (subseq (shape-ranges input-shape) 2))
              for f in (mapcar #'range-size (subseq (shape-ranges filter-shape) 2))
              for stride in strides
              for pad in paddings
              for dilation in dilations
              collect (range
                        (if transposedp
                            (+ (* stride i) (- stride) (* dilation f) (- dilation) (- (* 2 pad )) 1)
                            (1+ (/ (+ i (* 2 pad) (- (1+ (* (1- f) dilation)))) stride))))))))

(defun lazy-convolution (input
                          filter
                          &key
                          algorithm
                          strides
                          paddings
                          dilations
                          group-count
                          (math-type *cudnn-default-math-type*)
                          (convolution-mode :cudnn-convolution)
                          transposedp)
  (let* ((input (lazy-array input))
         (filter (lazy-array filter))
         (input-shape (lazy-array-shape input))
         (filter-shape (lazy-array-shape filter))
         (strides (or strides (make-list (- (rank input) 2) :initial-element 1)))
         (paddings (or paddings (mapcar (lambda (s) (floor s 2)) (mapcar (lambda (x stride) (* x (if transposedp 0 stride))) (mapcar #'range-size (subseq (shape-ranges filter-shape) 2)) strides))))
         (dilations (or dilations (make-list (- (rank input) 2) :initial-element 1)))
         (output-shape (get-convolution-output-shape input-shape filter-shape strides paddings dilations :transposedp transposedp)))
    (make-instance (if transposedp 'lazy-convolution-backward-data 'lazy-convolution)
                   :inputs (list input filter)
                   :shape output-shape
                   :ntype (element-ntype input)
                   :algorithm algorithm
                   :dilations dilations
                   :strides strides
                   :paddings paddings
                   :math-type math-type
                   :group-count group-count
                   :convolution-mode convolution-mode)))

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
    (assert (= 1 (length output-buffers)))
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
      :math-type (lazy-convolution-math-type custom-op)
      :paddings (lazy-convolution-paddings custom-op)
      :dilations (lazy-convolution-dilations custom-op)
      :filter-strides (lazy-convolution-strides custom-op)
      :mode (lazy-convolution-convolution-mode custom-op)
      :direction (if (typep custom-op 'lazy-convolution-backward-data)
                     :backward-data
                     :forward))))

(defmethod lazy-custom-op-execute ((custom-op lazy-convolution-backward-filter)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (let* ((input-buffer (nth 0 input-buffers))
         (filter-buffer (nth 0 output-buffers))
         (output-buffer (nth 1 input-buffers))
         (input (buffer-storage input-buffer))
         (filter (buffer-storage filter-buffer))
         (output (buffer-storage output-buffer)))
    (petalisp-cuda.cudnn-handler::cudnn-convolution
      (transform-cuda-array input
                            (unnormalizing-transformation (buffer-shape input-buffer)
                                                          (lazy-array-shape (nth 0 (lazy-array-inputs custom-op)))))
      (transform-cuda-array filter
                            (unnormalizing-transformation (buffer-shape filter-buffer)
                                                          (lazy-array-shape custom-op)))
      (transform-cuda-array output
                            (unnormalizing-transformation (buffer-shape output-buffer)
                                                          (lazy-array-shape (nth 1 (lazy-array-inputs custom-op)))))
      (petalisp-cuda.backend::cudnn-handler backend)
      :algorithm (lazy-convolution-algorithm custom-op)
      :group-count (lazy-convolution-group-count custom-op)
      :math-type (lazy-convolution-math-type custom-op)
      :paddings (lazy-convolution-paddings custom-op)
      :dilations (lazy-convolution-dilations custom-op)
      :filter-strides (lazy-convolution-strides custom-op)
      :mode (lazy-convolution-convolution-mode custom-op)
      :direction :backward-filter)))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution-backward-data) (output-gradient lazy-array) (index (eql 0)))
  (make-instance 'lazy-convolution
                 :inputs (list output-gradient (nth 1 (lazy-array-inputs lazy-convolution)))
                 :shape (lazy-array-shape (nth 0 (lazy-array-inputs lazy-convolution))) 
                 :strides (lazy-convolution-strides lazy-convolution)
                 :algorithm (lazy-convolution-algorithm lazy-convolution)
                 :math-type (lazy-convolution-math-type lazy-convolution)
                 :dilations (lazy-convolution-dilations lazy-convolution)
                 :group-count (lazy-convolution-group-count lazy-convolution)
                 :paddings (lazy-convolution-paddings lazy-convolution)
                 :convolution-mode (lazy-convolution-convolution-mode lazy-convolution)
                 :ntype (element-ntype (nth 0 (lazy-array-inputs lazy-convolution)))))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution-backward-data) (output-gradient lazy-array) (index (eql 1)))
  ;; Should do the same as bool caffe2::CudnnConvTransposeGradientOp<T>::RunOnDevice()
  (make-instance 'lazy-convolution-backward-filter
                 :shape (lazy-array-shape (nth 1 (lazy-array-inputs lazy-convolution)))
                 ;; Lol, inputs switched in comparison to normal lazy-convolution-backward-filter
                 :inputs (list (nth 0 (lazy-array-inputs lazy-convolution)) output-gradient)
                 :algorithm (lazy-convolution-algorithm lazy-convolution)
                 :strides (lazy-convolution-strides lazy-convolution)
                 :math-type (lazy-convolution-math-type lazy-convolution)
                 :dilations (lazy-convolution-dilations lazy-convolution)
                 :group-count (lazy-convolution-group-count lazy-convolution)
                 :paddings (lazy-convolution-paddings lazy-convolution)
                 :convolution-mode (lazy-convolution-convolution-mode lazy-convolution)
                 :ntype (element-ntype (nth 0 (lazy-array-inputs lazy-convolution)))))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution-backward-filter) output-gradient index)
  (error "Not implemented"))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution) (output-gradient lazy-array) (index (eql 0)))
  (make-instance 'lazy-convolution-backward-data
                 :inputs (list output-gradient (nth 1 (lazy-array-inputs lazy-convolution)))
                 :shape (lazy-array-shape (nth 0 (lazy-array-inputs lazy-convolution))) 
                 :strides (lazy-convolution-strides lazy-convolution)
                 :algorithm (lazy-convolution-algorithm lazy-convolution)
                 :math-type (lazy-convolution-math-type lazy-convolution)
                 :dilations (lazy-convolution-dilations lazy-convolution)
                 :group-count (lazy-convolution-group-count lazy-convolution)
                 :paddings (lazy-convolution-paddings lazy-convolution)
                 :convolution-mode (lazy-convolution-convolution-mode lazy-convolution)
                 :ntype (element-ntype (nth 0 (lazy-array-inputs lazy-convolution)))))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution) (output-gradient lazy-array) (index (eql 1)))
  (make-instance 'lazy-convolution-backward-filter 
                 :shape (lazy-array-shape (nth 1 (lazy-array-inputs lazy-convolution)))
                 :inputs (list output-gradient (nth 0 (lazy-array-inputs lazy-convolution)))
                 :algorithm (lazy-convolution-algorithm lazy-convolution)
                 :strides (lazy-convolution-strides lazy-convolution)
                 :math-type (lazy-convolution-math-type lazy-convolution)
                 :dilations (lazy-convolution-dilations lazy-convolution)
                 :group-count (lazy-convolution-group-count lazy-convolution)
                 :paddings (lazy-convolution-paddings lazy-convolution)
                 :convolution-mode (lazy-convolution-convolution-mode lazy-convolution)
                 :ntype (element-ntype (nth 0 (lazy-array-inputs lazy-convolution)))))

(defmethod substitute-array ((lazy-convolution lazy-convolution))
  (make-instance (class-of lazy-convolution)
                 :shape (lazy-array-shape lazy-convolution)
                 :ntype (element-ntype lazy-convolution)
                 :inputs (mapcar #'substitute-array (lazy-array-inputs lazy-convolution))
                 :algorithm (lazy-convolution-algorithm lazy-convolution)
                 :dilations (lazy-convolution-dilations lazy-convolution)
                 :strides (lazy-convolution-strides lazy-convolution)
                 :paddings (lazy-convolution-paddings lazy-convolution)
                 :math-type (lazy-convolution-math-type lazy-convolution)
                 :convolution-mode (lazy-convolution-convolution-mode lazy-convolution)
                 :group-count (lazy-convolution-group-count lazy-convolution)))
