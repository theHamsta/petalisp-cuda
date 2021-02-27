(in-package petalisp-cuda.cudnn-ops)

(defun lazy-reduction (input output-shape reduction-operation)
  (let* ((input (lazy-array input)))
    (make-instance 'lazy-reduction
                   :inputs (list input)
                   :shape output-shape
                   :ntype (element-ntype input)
                   :number-of-values 1
                   :reduction-operation reduction-operation)))

(defclass lazy-reduction (lazy-custom-op)
  ((%reduction-operation 
    :initarg :reduction-operation
    :accessor lazy-reduction-reduction-operation
    :type (or function symbol))))

(defmethod lazy-custom-op-execute ((custom-op lazy-reduction)
                                   (backend cuda-backend)
                                   (input-buffers list)
                                   (output-buffers list))
  (petalisp-cuda.cudnn-handler::cudnn-reduce-array (buffer-storage (nth 0 input-buffers))
                                                   (buffer-storage (nth 0 output-buffers))
                                                   (lazy-reduction-reduction-operation custom-op)
                                                   (petalisp-cuda.backend::cudnn-handler backend)))

(defmethod petalisp.api::input-gradient ((lazy-reduction lazy-reduction) (output-gradient lazy-array) (index (eql 0)))
  (alexandria:switch ((lazy-reduction-reduction-operation lazy-reduction) :test #'equalp)
    (#'+ (reshape output-gradient (array-shape (nth 0 (lazy-array-inputs lazy-reduction)))))
    (t (error "Not implemented!"))))
