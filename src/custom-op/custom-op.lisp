(in-package petalisp-cuda.custom-op)

(defclass lazy-custom-op (non-empty-non-immediate)
  ((%number-of-values
    :initarg :number-of-values
    :reader lazy-map-number-of-values
    :type (integer 0 (#.multiple-values-limit)))))

(defmethod petalisp.ir::grow-dendrite
  ((dendrite dendrite)
   (lazy-custom-op lazy-custom-op))
  (with-accessors ((shape dendrite-shape)
                   (cons dendrite-cons)) dendrite
    (let* ((inputs (lazy-array-inputs lazy-custom-op))
           (input-conses (loop for input in inputs collect (cons 0 input))))
      (setf (cdr cons)
            (make-custom-op-instruction
              lazy-custom-op
              input-conses
              1))
      ;; If our function has zero inputs, we are done.  Otherwise we create
      ;; one dendrite for each input and continue growing.
      (loop for input in inputs
            for input-cons in input-conses do
            (let ((new-dendrite (copy-dendrite dendrite)))
              (setf (dendrite-cons new-dendrite) input-cons)
              (if (typep input 'non-empty-immediate)
                  (grow-dendrite new-dendrite input)
                  (push new-dendrite (cluster-dendrites (ensure-cluster input)))))))))

(defmethod petalisp.ir::grow-dendrite
  ((dendrite dendrite)
   (lazy-custom-op lazy-custom-op))
  (with-accessors ((shape dendrite-shape)
                   (transformation dendrite-transformation)
                   (stem dendrite-stem)
                   (cons dendrite-cons)) dendrite
    (let* ((kernel (stem-kernel stem))
           (custom-op-kernel (make-custom-op-kernel :iteration-space shape
                                                    :custom-op lazy-custom-op))
           (ntype (petalisp.type-inference:generalize-ntype
                    (element-ntype lazy-custom-op)))
           (buffer
             (alexandria:ensure-gethash
               lazy-custom-op
               (ir-converter-scalar-table *ir-converter*)
               (make-buffer
                 :shape shape
                 :ntype ntype
                 :storage nil)))
           (inputs (lazy-array-inputs lazy-custom-op)))
      (setf (cdr cons)
            (make-load-instruction kernel buffer transformation))
      (loop for lazy-array in inputs do
            (let* ((cluster (make-cluster lazy-array))
                   (shape (lazy-array-shape lazy-array))
                   (transformation (identity-transformation (shape-rank shape)))
                   (buffer (make-buffer
                             :shape shape
                             :ntype (petalisp.type-inference:generalize-ntype
                                      (element-ntype lazy-array))))
                   (new-dendrite (make-dendrite cluster shape (list buffer))))
              (if (typep lazy-array 'non-empty-immediate)
                  (grow-dendrite new-dendrite lazy-array)
                  (push new-dendrite (cluster-dendrites (ensure-cluster lazy-array))))
              (make-load-instruction custom-op-kernel buffer transformation)
              buffer))
      (let ((custom-op-instruction (make-custom-op-instruction
                                     lazy-custom-op
                                     nil
                                     0)))
        (make-store-instruction custom-op-kernel (cons 0 custom-op-instruction) buffer (identity-transformation (shape-rank (buffer-shape buffer))))))))

(defstruct (custom-op-instruction
            (:include instruction)
            (:predicate custom-op-instruction-p)
            (:constructor make-custom-op-instruction (custom-op inputs number-of-values)))
  (custom-op nil :type lazy-custom-op)
  (number-of-values nil :type (integer 0 (#.multiple-values-limit))))

(defstruct (custom-op-kernel
            (:predicate custom-op-kernel-p)
            (:include kernel)
            (:constructor make-custom-op-kernel))
  (custom-op nil :type lazy-custom-op))

(defgeneric lazy-custom-op-execute (custom-op backend input-buffers output-buffers))

(defun custom-op-kernel-execute (kernel backend)
  (lazy-custom-op-execute (custom-op-kernel-custom-op kernel)
                          backend
                          (kernel-inputs kernel)
                          (kernel-outputs kernel)))

(defun kernel-inputs (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-inputs
     (lambda (buffer) (push buffer buffers))
     kernel)
    buffers))

(defun kernel-outputs (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-outputs
     (lambda (buffer) (push buffer buffers))
     kernel)
    buffers))
