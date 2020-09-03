(in-package petalisp-cuda.iteration-scheme)

;; same as block-iteration-scheme, but uses a trick to cache slow coordinate accesses
;; https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
(defclass slow-coordinate-transposed-scheme (block-iteration-scheme)
  ((%slow-loads :initarg :slow-loads
                :accessor slow-loads
                :type list)))

(defmethod iteration-code :around ((iteration-scheme slow-coordinate-transposed-scheme) kernel-body)
  (let* ((first-slow-load (first (slow-loads iteration-scheme)))
         (cached-buffer (load-instruction-buffer first-slow-load))
         (block-shape (filtered-block-shape iteration-scheme)))
   (call-next-method iteration-scheme 
                    `(with-shared-memory ((shared-mem ,(cl-cuda-type-from-buffer cached-buffer) ,(first block-shape) ,(1+ (second block-shape))))
                       ,kernel-body))))

(defmethod iteration-scheme-buffer-access ((iteration-scheme slow-coordinate-transposed-scheme) instruction buffer kernel-parameter)
  ;; else unchached
  (call-next-method iteration-scheme instruction buffer kernel-parameter))
