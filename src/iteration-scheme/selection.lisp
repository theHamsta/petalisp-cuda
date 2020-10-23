(in-package petalisp-cuda.iteration-scheme)

(defun select-iteration-scheme (iteration-shape block-shape-as-list array-strides)
  (let* ((iteration-strides (loop for range in (shape-ranges iteration-shape)
                                  for stride in array-strides
                                  count t into i
                                  unless (range-empty-p range)
                                  collect (list (1- i) stride)))
         (fastest-dimensions (mapcar #'car (sort iteration-strides #'< :key #'second)))
         (xyz (subseq fastest-dimensions 0 (min 3 (length fastest-dimensions))))
         (block-shape '())
         (rank (shape-rank iteration-shape)))
    (dotimes (i rank)
      (push (range 0) block-shape))
    (mapcar (lambda (idx range-size) (setf (nth idx block-shape) (range range-size))) xyz block-shape-as-list)
    (make-instance (if *shape-independent-code*
                       'symbolic-block-iteration-scheme
                       'block-iteration-scheme)
                   :shape iteration-shape
                   :xyz-dimensions xyz
                   :block-shape block-shape)))

