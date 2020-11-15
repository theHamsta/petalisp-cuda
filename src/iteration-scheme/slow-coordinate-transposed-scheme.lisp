(in-package petalisp-cuda.iteration-scheme)

(when (boundp cl-cuda:+hacked-cl-cuda+)
;; same as block-iteration-scheme, but uses a trick to cache slow coordinate accesses
;; https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
(defclass slow-coordinate-transposed-scheme (symbolic-block-iteration-scheme)
  ((%slow-loads :initarg :slow-loads
                :accessor slow-loads
                :type list)
   (%load-strategy :initarg :load-strategy
                   :accessor load-strategy
                   :type string)))

  (defmethod iteration-code :around ((iteration-scheme slow-coordinate-transposed-scheme) &rest kernel-body)
    (let* ((slow-loads (first (slow-loads iteration-scheme)))
           (cached-buffer (load-instruction-buffer first-slow-load))
           (block-shape (filtered-block-shape iteration-scheme))
           (load-strategy (load-strategy iteration-scheme)))
      ,(call-next-method iteration-scheme 
                         `((:inline-c ,(format t "typedef cub::BlockLoad<~A, ~A, 1, ~A, ~A, ~A> BlockLoad;
                                              __shared__ typename BlockLoad::TempStorage temp_storage;" ((first slow-loads)) load-strategy (first block-shape) (second block-shape) (third block-shape)))
                                      (loop for l in slow-loads
                                            collect `(:inline-c "BlockLoad(temp_storage).Load("
                                                                ,(+ ,(buffer->kernel-parameter (first (instruction-inputs instruction))) ,(linearize-instruction-transformation instruction))
                                                                " , "
                                                                (get-instruction-symbol l)
                                                                ");"))
                                      ,@kernel-body))))

(defmethod iteration-scheme-buffer-access ((iteration-scheme slow-coordinate-transposed-scheme) instruction buffer kernel-parameter)
  ;; else unchached
  (call-next-method iteration-scheme instruction buffer kernel-parameter))


)


