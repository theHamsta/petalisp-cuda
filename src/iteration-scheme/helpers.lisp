(in-package petalisp-cuda.iteration-scheme)

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

(defun filtered-iteration-shape (iteration-scheme iteration-shape)
  (filter-xyz-dimensions (mapcar #'range-size (shape-ranges iteration-shape)) (xyz-dimensions iteration-scheme)))

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

(defun linearize-instruction-transformation (instruction &optional buffer kernel-parameter shape-independent-p)
  (let* ((transformation (instruction-transformation instruction))
         (input-rank (transformation-input-rank transformation))
         (strides (or (if buffer
                          (if shape-independent-p
                              (loop for i below (shape-rank (buffer-shape buffer))
                                    collect (format-symbol t "~A-stride-~A" kernel-parameter i))
                           (cuda-array-strides (buffer-storage buffer)))
                          (make-list input-rank :initial-element 1))
                      '(1)))
         (index-space (get-counter-vector input-rank) )
         (transformed (transform index-space transformation)))
    (let ((rtn `(+ ,@(mapcar (lambda (a b) `(* ,a ,b)) transformed strides))))
      (if (= (length rtn) 1)
          0 ; '+ with zero arguments
          rtn))))

(defmethod iteration-scheme-buffer-access ((iteration-scheme iteration-scheme) instruction buffer kernel-parameter)
  ;; We can always do a uncached memory access
  `(aref ,kernel-parameter ,(linearize-instruction-transformation instruction buffer kernel-parameter (shape-independent-p iteration-scheme))))

(defmethod shape-independent-p ((iteration-scheme iteration-scheme))
  )
(defmethod generic-offsets-p ((iteration-scheme iteration-scheme))
  )

(defun get-instruction-symbol (instruction &optional (suffix ""))
  (trivia:match instruction
    ;; weird multiple value instruction
    ((trivia:guard (cons a b ) (> a 0)) (format-symbol t "$~A_~A~A" (instruction-number b) a suffix))   
    ;; normal instruction
    (_
      (format-symbol t "$~A~A"
                     (etypecase instruction
                       (number instruction)
                       (cons (instruction-number (cdr instruction)))
                       (instruction (instruction-number instruction)))
                     suffix))))


(defun kernel-parameter-name (kernel-parameter)
  (first kernel-parameter))

(defun kernel-parameter-type (kernel-parameter)
  (second kernel-parameter))
