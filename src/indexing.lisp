(defpackage petalisp-cuda.indexing
  (:use :cl
        :petalisp)
  (:import-from :alexandria :iota :format-symbol)
  (:import-from :cl-cuda :block-dim-x :block-dim-y :block-dim-z
                          :block-idx-x :block-idx-y :block-idx-z
                          :thread-idx-x :thread-idx-y :thread-idx-z)
  (:export :block-iteration-scheme
           :make-block-iteration-scheme
           :call-parameters
           :iteration-code))

(in-package petalisp-cuda.indexing)

(defclass iteration-scheme ()
  ((%shape :initarg shape
           :accessor iteration-shape
           :type petalisp.core:shape)
   (%xyz-dimensions :initarg xyz-dimensions
                    :accessor xyz-dimensions
                    :type list)))

(defgeneric call-parameters (iteration-scheme))
(defgeneric call-parameters (iteration-scheme))

;; block-iteration-scheme
(defclass block-iteration-scheme (iteration-scheme)
  ((%block-shape :initarg block-shape
                :accessor block-shape
                :type petalisp.core:shape)))

(defun make-block-iteration-scheme (iteration-shape block-shape-as-list array-strides)
  (let* ((iteration-strides (loop for range in iteration-shape
                                  for stride in array-strides
                                  count t into i
                                  unless (range-empty-p range)
                                  collect (list i stride)))
         (fastest-dimensions (mapcar #'car (sort iteration-strides #'< :key #'second)))
         (xyz (subseq fastest-dimensions 0 (max 3 (length fastest-dimensions))))
         (block-shape '())
         (rank (shape-rank iteration-shape)))
    (progn
      (dotimes (i rank)
        (push (~ 1) block-shape))
      (mapcar (lambda (idx range-size) (setf (nth idx block-shape) (~ range-size))) xyz block-shape-as-list)
      (make-instance 'block-iteration-scheme
                     :shape iteration-shape
                     :xyz-dimensions xyz
                     :block-shape block-shape))))

(defmethod call-parameters ((iteration-scheme block-iteration-scheme))
  (let ((filtered-iteration-shape (filtered-iteration-shape iteration-scheme))
        (filtered-block-shape (filtered-block-shape iteration-scheme)))
    `(:grid-dim  ,(mapcar #'range-divup
                        (shape-ranges filtered-iteration-shape)
                        (shape-ranges filtered-block-shape))
     :block-dim ,filtered-block-shape)))

(defmethod iteration-code ((iteration-scheme block-iteration-scheme) kernel-body)
  (let ((iteration-ranges (shape-ranges (iteration-shape iteration-scheme)))
        (xyz (xyz-dimensions iteration-scheme)))
    ;; define x,y,z dimensions
    `(let (,(loop for dim-idx in xyz
                  for letter in (list 'x 'y 'z)
                  collect (let ((current-range (nth dim-idx iteration-ranges)))
                            `(,(format-symbol nil "idx-~A" dim-idx) 
                               (+ ,(range-start current-range)
                                  (* ,(range-step current-range)
                                     ,(case letter 
                                        (x '(+ thread-idx-x (* block-idx-x block-dim-x)))
                                        (y '(+ thread-idx-y (* block-idx-y block-dim-y)))
                                        (z '(+ thread-idx-z (* block-idx-x block-dim-z))))))))))
       (progn
          ;; return out-of-bounds threads
          ,@(loop for dim-idx in xyz
                collect `(if (>= ,(format-symbol nil  "idx-~A" dim-idx) ,(range-end (nth dim-idx iteration-ranges)))
                             (return)))
          ;; iterate over remaining dimensions with for-loops (c++, do in cl-cuda)
          ;; and append kernel-body
          ,(make-range-loop iteration-ranges 0 xyz kernel-body)))))

;; Helpers
(defun range-divup (range-a range-b)
  "How often fits range-b in range-a when considering in partial range-b at start and end"
  (let ((a range-a)
        (b range-b))
    (progn
      (assert (equal (range-step a) (range-step b)))
      (assert (equal (range-start a) (range-start b))) ; for now. TODO: drop that restriction
      (ceiling (range-size a) (range-size b))))) ; so it's basically ceiling + assertions right now

(defun filter-xyz-dimensions (list xyz-dimensions)
  (trivia:match (mapcar (lambda (idx) (nth idx list)) xyz-dimensions)
    ((list x)     (list x 1 1))
    ((list x y)   (list x y 1))
    ((list x y z) (list x y z))
    (_ (error "CUDA allows maximum 3 dimensions"))))

(defun filtered-block-shape (iteration-scheme)
  (filter-xyz-dimensions (block-shape iteration-scheme) (xyz-dimensions iteration-scheme)))

(defun filtered-iteration-shape (iteration-scheme)
  (filter-xyz-dimensions (iteration-shape iteration-scheme) (xyz-dimensions iteration-scheme)))

(defun range-empty-p (range)
  (= (range-size range) 0))

(defun make-range-loop (iteration-ranges dim-idx xyz kernel-body)
  (let ((body (if (rest iteration-ranges) 
                  (make-range-loop (rest iteration-ranges) (1+ dim-idx) xyz kernel-body)
                  kernel-body))
        (dim-range (first iteration-ranges))
        (dim-symbol (format-symbol nil "idx-~A" dim-idx)))
    (if (or (range-empty-p dim-range)
            (member dim-idx xyz))
        body
        `(do ((,dim-symbol ,(range-start dim-range) (+ ,dim-symbol ,(range-step dim-range))))
             ((< ,dim-symbol ,(range-end dim-range)))
           ,body))))