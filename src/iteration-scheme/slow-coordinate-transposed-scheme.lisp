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

  (defun transposed-instruction (instruction)
    )

  (defmethod iteration-code :around ((iteration-scheme slow-coordinate-transposed-scheme) kernel-body buffer->kernel-parameter)
    (let* ((slow-loads (slow-loads iteration-scheme))
           (block-shape (filtered-block-shape iteration-scheme))
           (load-strategy (load-strategy iteration-scheme)))
      (call-next-method iteration-scheme 
                        ;; Declare cached variables: (let (($0_ (coerce-float 0)) ($1_ ...) ...) ...)
                        `(let ,(loop for instruction in slow-loads
                                     collect `(,(get-instruction-symbol instruction "_cached")
                                                (,(alexandria:format-symbol :keyword (string-upcase (format nil "coerce-~A" (cuda-type-string instruction)))) 0)))
                           ;; Load slow loads over cache
                           ,@(loop for instruction in slow-loads
                                   collect (let* ((buffer (load-instruction-buffer instruction))
                                                  (kernel-parameter (buffer->kernel-parameter buffer))
                                                  (transposed-instruction instruction))
                                              `(let ((oob-me? (get-oob-check transposed-instruction thread-idx-x thread-idx-y thread-idx-z))
                                                     (oob-transposed? (get-oob-check transposed-instruction thread-idx-y thread-idx-x thread-idx-z)))
                                                 (set (aref shared-mem thread-idx-x thread-idx-y)
                                                      (if ,oob-me?
                                                          0
                                                          (aref ,(linearize-instruction-transformation transposed-instruction
                                                                                                   buffer
                                                                                                   kernel-parameter
                                                                                                   (shape-independent-p iteration-scheme))))
                                                 (:inline-c "__syncthreads();")
                                                 (set ,(get-instruction-symbol instruction "_cached")
                                                      (if ,oob-transposed?
                                                          ,(linearize-instruction-transformation instruction
                                                                                                 buffer
                                                                                                 kernel-parameter
                                                                                                 (shape-independent-p iteration-scheme)))
                                                          (aref shared-mem thread-idx-y thread-idx-x))))
                           ;; Progamm body using cached variables
                           ,@kernel-body))))

  (defmethod iteration-scheme-buffer-access ((iteration-scheme slow-coordinate-transposed-scheme) instruction buffer kernel-parameter)
    (if (find instruction (slow-loads iteration-scheme))
        (get-instruction-symbol instruction "_cached")
        ;; else unchached
      (call-next-method iteration-scheme instruction buffer kernel-parameter)))

(defmethod iteration-scheme-shared-mem ((iteration-scheme slow-coordinate-transposed-scheme))
  (values (concatentate 'list (butlast (block-shape iteration-scheme)) (list (1+ (last (block-shape iteration-scheme)))))
          (instruction-cuda-type (first (slow-loads iteration-scheme)))))
    )
