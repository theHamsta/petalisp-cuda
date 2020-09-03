(in-package petalisp-cuda.iteration-scheme)

;; Helpers
(defun range-divup (range-a range-b)
  "How often fits range-b in range-a when considering in partial range-b at start and end"
  (let ((a range-a)
        (b range-b))
    (assert (equal (range-step a) (range-step b)))
    (assert (equal (range-start a) (range-start b))) ; for now. TODO: drop that restriction
    (ceiling (range-size a) (range-size b)))) ; so it's basically ceiling + assertions right now

(defun filter-xyz-dimensions (list xyz-dimensions)
  (trivia:match (mapcar (lambda (idx) (nth idx list)) xyz-dimensions)
    ((list)       (list 1 1 1))
    ((list x)     (list x 1 1))
    ((list x y)   (list x y 1))
    ((list x y z) (list x y z))
    (_ (error "CUDA allows maximum 3 dimensions"))))

(defun filtered-block-shape (iteration-scheme)
  (let ((xyz-size (mapcar #'range-size (block-shape iteration-scheme))))
    (filter-xyz-dimensions xyz-size
                           (xyz-dimensions iteration-scheme))))

(defun filtered-iteration-shape (iteration-scheme)
  (filter-xyz-dimensions (mapcar #'range-size (shape-ranges (iteration-shape iteration-scheme))) (xyz-dimensions iteration-scheme)))

(defun range-empty-p (range)
  (= (range-size range) 0))

(defun get-counter-symbol (dimension-index)
  (format-symbol t "idx~A" dimension-index))

(defun get-counter-vector (dim)
  (mapcar (lambda (dim) (format-symbol t "idx~A" dim)) (iota dim)))

(defun make-range-loop (iteration-ranges dim-idx xyz kernel-body)
  (let ((body (if (rest iteration-ranges) 
                  (make-range-loop (rest iteration-ranges) (1+ dim-idx) xyz kernel-body)
                  kernel-body))
        (dim-range (first iteration-ranges))
        (dim-symbol (get-counter-symbol dim-idx)))
    (if (or (or (null dim-range) (range-empty-p dim-range))
            (member dim-idx xyz))
        body
        `(do ((,dim-symbol ,(range-start dim-range) (+ ,dim-symbol ,(range-step dim-range))))
          ((>= ,dim-symbol ,(range-end dim-range)))
          ,body))))

(defun linearize-instruction-transformation (instruction &optional buffer)
  (let* ((transformation (instruction-transformation instruction))
         (input-rank (transformation-input-rank transformation))
         (strides (if buffer (cuda-array-strides (buffer-storage buffer)) (make-list input-rank :initial-element 1)))
         (index-space (get-counter-vector input-rank) )
         (transformed (transform index-space transformation)))
    (let ((rtn `(+ ,@(mapcar (lambda (a b) `(* ,a ,b)) transformed strides))))
      (if (= (length rtn) 1)
          0 ; '+ with zero arguments
          rtn))))
