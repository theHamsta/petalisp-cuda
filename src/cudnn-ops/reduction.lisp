(in-package petalisp-cuda.cudnn-ops)

(defun lazy-reduction (input output-shape reduction-operation)
  (let* ((input (lazy-array input)))
    (make-instance 'lazy-reduction
                   :inputs (list input)
                   :shape output-shape
                   :ntype (element-ntype input)
                   :number-of-values 1
                   :operation reduction-operation)))

(defclass lazy-reduction (lazy-custom-op)
  ((%reduction-operation 
    :initarg :operation
    :accessor lazy-reduction-operation
    :type (or function symbol))))

(defmethod lazy-custom-op-execute ((custom-op lazy-reduction)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (transform-cuda-array
                                                     (buffer-storage (nth 0 input-buffers))
                                                     (unnormalizing-transformation
                                                       (buffer-shape (nth 0 input-buffers))
                                                       (lazy-array-shape (nth 0 (lazy-array-inputs custom-op)))))
                                                   (transform-cuda-array
                                                     (buffer-storage (nth 0 output-buffers))
                                                     (unnormalizing-transformation
                                                       (buffer-shape (nth 0 output-buffers))
                                                       (lazy-array-shape custom-op)))
                                                   (lazy-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))

(defun shape-to-list (shape)
  (mapcar #'range-size (shape-ranges shape)))

(defmethod petalisp.api::input-gradient ((lazy-reduction lazy-reduction) (output-gradient lazy-array) (index (eql 0)))
  (let* ((input-shape (lazy-array-shape (nth 0 (lazy-array-inputs lazy-reduction))))
         (output-shape (lazy-array-shape lazy-reduction))
         (shape-ratio (reduce #'* (mapcar (lambda (i o) (/ i o)) (shape-to-list input-shape) (shape-to-list output-shape)))))
    (alexandria:switch ((lazy-reduction-operation lazy-reduction) :test #'equalp)
      (#'+ (reshape output-gradient input-shape))
      (:avg (α #'* (reshape output-gradient input-shape) shape-ratio))
      ;; dy * 0.5/sqrt(dy) * 2x ??
      (:norm2 (α #'* (α #'sqrt (reshape output-gradient input-shape)) (nth 0 (lazy-array-inputs lazy-reduction))))
      (t (error "Not implemented!")))))

(defmethod substitute-array ((lazy-map lazy-reduction))
  (make-instance 'lazy-reduction
    :shape (lazy-array-shape lazy-map)
    :ntype (element-ntype lazy-map)
    :operation (lazy-reduction-operation lazy-map)
    :inputs (mapcar #'substitute-array (lazy-array-inputs lazy-map))))
