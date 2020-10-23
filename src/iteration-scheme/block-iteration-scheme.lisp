(in-package petalisp-cuda.iteration-scheme)

;; block-iteration-scheme (one thread per datum)
(defclass block-iteration-scheme (iteration-scheme)
  ((%block-shape :initarg :block-shape
                 :accessor block-shape
                 :type petalisp.core:shape)))

(defmethod call-parameters ((iteration-scheme block-iteration-scheme) (iteration-shape shape))
  (let ((filtered-iteration-shape (filtered-iteration-shape iteration-scheme iteration-shape))
        (filtered-block-shape (filtered-block-shape iteration-scheme)))
    `(:grid-dim  ,(mapcar #'ceiling
                          filtered-iteration-shape
                          filtered-block-shape)
      :block-dim ,filtered-block-shape)))

(defmethod iteration-code ((iteration-scheme block-iteration-scheme) kernel-body)
  (let ((iteration-ranges (or (shape-ranges (iteration-space iteration-scheme)) (list (range 1))))
        (xyz (or (xyz-dimensions iteration-scheme) '(0))))
    ;; define x,y,z dimensions
    `(let ,(loop for dim-idx in xyz
                 for letter in (list 'x 'y 'z)
                 collect (let ((current-range (nth dim-idx iteration-ranges)))
                           `(, (get-counter-symbol dim-idx) 
                               (+ ,(range-start current-range)
                                  (* ,(range-step current-range)
                                     ,(case letter 
                                        (x '(+ thread-idx-x (* block-idx-x block-dim-x)))
                                        (y '(+ thread-idx-y (* block-idx-y block-dim-y)))
                                        (z '(+ thread-idx-z (* block-idx-z block-dim-z)))))))))
       ;; return out-of-bounds threads
       ,@(loop for dim-idx in xyz
               collect `(if (>= ,(get-counter-symbol dim-idx) ,(range-end (nth dim-idx iteration-ranges)))
                            (return)))
       ;; iterate over remaining dimensions with for-loops (c++, do in cl-cuda)
       ;; and append kernel-body
       ,(make-range-loop iteration-ranges 0 xyz kernel-body))))
