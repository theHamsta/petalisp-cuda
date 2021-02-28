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
    :type (or symbol null))))

(defclass lazy-convolution-backward-data (lazy-convolution)
  ())

(defclass lazy-convolution-backward-filter (lazy-convolution)
  ())

(defun lazy-convolution (input filter &key algorithm
                                           strides
                                           paddings
                                           dilations
                                           group-count
                                           (math-type *cudnn-default-math-type*))
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
                 :algorithm algorithm
                 :dilations dilations
                 :strides strides
                 :paddings paddings
                 :math-type math-type
                 :group-count group-count)))

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
      :math-type (lazy-convolution-math-type custom-op)
      :paddings (lazy-convolution-paddings custom-op)
      :dilations (lazy-convolution-dilations custom-op)
      :filter-strides (lazy-convolution-strides custom-op)
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
      :direction :backward-filter)))

(defmethod petalisp.api::input-gradient ((lazy-convolution lazy-convolution-backward-data) output-gradient index)
  (error "Not implemented"))

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
                 :ntype (element-ntype (nth 0 (lazy-array-inputs lazy-convolution)))))

(defmethod substitute-array ((lazy-map lazy-convolution))
  (make-instance (class-of lazy-map)
                 :shape (lazy-array-shape lazy-map)
                 :ntype (element-ntype lazy-map)
                 :inputs (mapcar #'substitute-array (lazy-array-inputs lazy-map))
                 :algorithm (lazy-convolution-algorithm lazy-map)
                 :dilations (lazy-convolution-dilations lazy-map)
                 :strides (lazy-convolution-strides lazy-map)
                 :paddings (lazy-convolution-paddings lazy-map)
                 :math-type (lazy-convolution-math-type lazy-map)
                 :group-count (lazy-convolution-group-count lazy-map)))
