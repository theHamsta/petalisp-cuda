(in-package petalisp-cuda.iteration-scheme)


(defun slow-load-p (instruction fastest-dimension)
  (when fastest-dimension
    (let* ((transformation (instruction-transformation instruction))
           (output-rank (1- (length (transformation-output-mask transformation))))
           (output-mask (transformation-output-mask transformation)))
      (when (> output-rank 0)
        (/= (aref output-mask fastest-dimension) fastest-dimension)))))

(defun find-slow-loads (kernel fastest-dimension)
  (let ((slow-loads nil))
    (map-kernel-load-instructions (lambda (instruction)
                                    (when (slow-load-p instruction fastest-dimension)
                                      (push instruction slow-loads)))
                                  kernel)
    slow-loads))

`(ssa (sda ))

(defun select-iteration-scheme (kernel iteration-shape block-shape-as-list array-strides)
  (let* ((iteration-strides (loop for range in (shape-ranges iteration-shape)
                                  for stride in array-strides
                                  count t into i
                                  unless (range-empty-p range)
                                  collect (list (1- i) stride)))
         (fastest-dimensions (mapcar #'car (sort iteration-strides #'< :key #'second)))
         (fastest-dimension (first fastest-dimensions))
         (xyz (subseq fastest-dimensions 0 (min 3 (length fastest-dimensions))))
         (block-shape '())
         (iteration-shape (kernel-iteration-space kernel))
         (rank (shape-rank iteration-shape))
         (slow-loads (find-slow-loads kernel fastest-dimension)))
    (dotimes (i rank)
      (push (range 0) block-shape))
    (mapcar (lambda (idx range-size) (setf (nth idx block-shape) (range range-size))) xyz block-shape-as-list)
    (if (and (boundp cl-cuda:+hacked-cl-cuda+) (> (length slow-loads) 0))
      (make-instance 'slow-coordinate-transposed-scheme
                     :shape iteration-shape
                     :xyz-dimensions xyz
                     :block-shape block-shape
                     :slow-loads slow-loads
                     :load-strategy *slow-coordinate-load-strategy*)
      (make-instance (if *shape-independent-code*
                       'symbolic-block-iteration-scheme
                       'block-iteration-scheme)
                     :shape iteration-shape
                     :xyz-dimensions xyz
                     :block-shape block-shape)))

