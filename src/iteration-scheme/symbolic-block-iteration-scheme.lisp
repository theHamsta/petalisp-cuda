(in-package petalisp-cuda.iteration-scheme)

;; block-iteration-scheme (one thread per datum)
(defclass symbolic-block-iteration-scheme (block-iteration-scheme)
  ((%generic-offsets :initform *generic-offsets*
                     :accessor %generic-offsets-p)))

(defmethod iteration-code ((iteration-scheme symbolic-block-iteration-scheme) kernel-body)
  (let* ((ranges-from-ir (shape-ranges (iteration-space iteration-scheme)))
        (iteration-ranges (or ranges-from-ir (list (range 1))))
        (xyz (or (xyz-dimensions iteration-scheme) '(0))))
    ;; define x,y,z dimensions
    `(let ,(loop for dim-idx in xyz
                 for letter in (list 'x 'y 'z)
                 collect (let ((current-range (nth dim-idx iteration-ranges)))
                           `(,(get-counter-symbol dim-idx) 
                               (+ ,(if ranges-from-ir (format-symbol t "iteration-start-~A" dim-idx) 0)
                                  (* ,(range-step current-range)
                                     ,(case letter 
                                        (x '(+ thread-idx-x (* block-idx-x block-dim-x)))
                                        (y '(+ thread-idx-y (* block-idx-y block-dim-y)))
                                        (z '(+ thread-idx-z (* block-idx-z block-dim-z)))))))))
       ;; return out-of-bounds threads
       ,@(loop for dim-idx in xyz
               collect `(if (>= ,(get-counter-symbol dim-idx) ,(if ranges-from-ir (format-symbol t "iteration-end-~A" dim-idx) 1))
                            (return)))
       ;; iterate over remaining dimensions with for-loops (c++, do in cl-cuda)
       ;; and append kernel-body
       ,(make-range-loop iteration-ranges 0 xyz kernel-body))))

(defmethod shape-independent-p ((iteration-scheme symbolic-block-iteration-scheme))
  t)
(defmethod generic-offsets-p ((iteration-scheme symbolic-block-iteration-scheme))
  (%generic-offsets-p iteration-scheme))
