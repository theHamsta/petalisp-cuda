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

  (defun instruction-cuda-type (instruction)
    (cl-cuda-type-from-ntype (buffer-ntype (load-instruction-buffer instruction))))

  (defun cuda-type-string (instruction)
    (cl-cuda.lang.type:cuda-type (instruction-cuda-type instruction)))

  (defmethod iteration-code :around ((iteration-scheme slow-coordinate-transposed-scheme) kernel-body buffer->kernel-parameter)
    (let* ((slow-loads (slow-loads iteration-scheme))
           (block-shape (filtered-block-shape iteration-scheme))
           (load-strategy (load-strategy iteration-scheme)))
      (call-next-method iteration-scheme 
                        ;; Declare cached variables: (let (($0_ (coerce-float 0)) ($1_ ...) ...) ...)
                        `(let ,(loop for instruction in slow-loads
                                     collect `(,(get-instruction-symbol instruction "_cached")
                                                (,(alexandria:format-symbol :keyword (string-upcase (format nil "coerce-~A" (cuda-type-string instruction)))) 0)))
                           ;; Declare cache
                           (:inline-c ,(format nil
"
typedef cub::BlockLoad<~A, ~A, ~A, ~A, ~A, ~A> BlockLoad;
__shared__ typename BlockLoad::TempStorage temp_storage;
"
                                               (string-downcase (instruction-cuda-type (first slow-loads)))
                                               (first block-shape) ; block-shape-x
                                               1 ; items-per-thread
                                               load-strategy ; cub::BlockLoadAlgorithm
                                               (second block-shape) ; block-shape-y
                                               (third block-shape))) ; block-shape-z
                           ;; Load slow loads over cache
                           ,@(loop for instruction in slow-loads
                                   collect (let* ((type (instruction-cuda-type instruction))
                                                  (type-string (cl-cuda.lang.type:cuda-type type))
                                                  (array-sym (symbol-name (get-instruction-symbol instruction "_array"))))
                                             `(:inline-c
                                               ,type-string " " ,array-sym "[1] = {0};
"
                                               "BlockLoad(temp_storage).Load("
                                               ,(kernel-parameter-name (funcall buffer->kernel-parameter (load-instruction-buffer instruction)))
                                               " + "
                                               ,(linearize-instruction-transformation instruction)
                                               " , "
                                               ,array-sym
                                               ");
__syncthreads();
"
                                               ,(get-instruction-symbol instruction "_cached") " = " ,array-sym "[0];
"
                                               )))
                           ;; Progamm body using cached variables
                           (,@kernel-body)))))

  (defmethod iteration-scheme-buffer-access ((iteration-scheme slow-coordinate-transposed-scheme) instruction buffer kernel-parameter)
    (if (find instruction (slow-loads iteration-scheme))
        (get-instruction-symbol instruction "_cached")
        ;; else unchached
      (call-next-method iteration-scheme instruction buffer kernel-parameter)))
  )


