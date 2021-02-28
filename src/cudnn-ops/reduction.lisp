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
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (buffer-storage (nth 0 input-buffers))
                                                   (buffer-storage (nth 0 output-buffers))
                                                   (lazy-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))

(defmethod petalisp.api::input-gradient ((lazy-reduction lazy-reduction) (output-gradient lazy-array) (index (eql 0)))
  (let ((input-shape (lazy-array-shape (nth 0 (lazy-array-inputs lazy-reduction))))
        (output-shape (lazy-array-shape lazy-reduction)))
   (alexandria:switch ((lazy-reduction-operation lazy-reduction) :test #'equalp)
    (#'+ (reshape output-gradient input-shape))
    (#'avg (α #'* (reshape output-gradient input-shape) (mapc 'vector (lambda (i o) (/ i o)) input-shape output-shape)))
    (t (error "Not implemented!")))))

(defmethod substitute-array ((lazy-map lazy-reduction))
  (make-instance 'lazy-reduction
    :shape (lazy-array-shape lazy-map)
    :ntype (element-ntype lazy-map)
    :operation (lazy-reduction-operation lazy-map)
    :inputs (mapcar #'substitute-array (lazy-array-inputs lazy-map))))
