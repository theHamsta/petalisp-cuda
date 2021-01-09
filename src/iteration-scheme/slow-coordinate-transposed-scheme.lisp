(in-package petalisp-cuda.iteration-scheme)

(when (boundp cl-cuda:+hacked-cl-cuda+)
  ;; same as block-iteration-scheme, but uses a trick to cache slow coordinate accesses
  ;; https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
  (defclass slow-coordinate-transposed-scheme (symbolic-block-iteration-scheme)
    ((%slow-loads :initarg :slow-loads
                  :accessor slow-loads
                  :type list)
     (%fastest-dimension :initarg :fastest-dimension
                         :accessor fastest-dimension)))

  (defun instruction-cuda-type (instruction)
    (cl-cuda-type-from-ntype (buffer-ntype (load-instruction-buffer instruction))))

  (defun cuda-type-string (instruction)
    (cl-cuda.lang.type:cuda-type (instruction-cuda-type instruction)))

  (defun transposed-load-instruction (instruction slow-idx fast-idx)
    (let* ((transformation (instruction-transformation instruction))
           (new-output-mask (alexandria:copy-array (transformation-output-mask transformation))))
      (rotatef (aref new-output-mask fast-idx) (aref new-output-mask slow-idx))
      (petalisp.ir::make-load-instruction
        (load-instruction-buffer instruction)
        (make-transformation 
                          :input-rank (transformation-input-rank transformation)
                          :output-rank (transformation-output-rank transformation)
                          :input-mask (transformation-input-mask transformation)
                          :output-mask new-output-mask
                          :scalings (transformation-scalings transformation)
                          :offsets (transformation-offsets transformation)))))

  (defmethod caching-code ((iteration-scheme slow-coordinate-transposed-scheme) kernel-body buffer->kernel-parameter)
    (let* ((slow-loads (slow-loads iteration-scheme))
           (load-dimension (aref (transformation-output-mask (instruction-transformation (first slow-loads))) fastest-dimension)))
      ;; Declare cached variables: (let (($0_ (coerce-float 0)) ($1_ ...) ...) ...)
      (shared-mem-code iteration-scheme
                       `((let ,(loop for instruction in slow-loads
                                     when (and (equalp (instruction-cuda-type instruction) (instruction-cuda-type (first slow-loads)))
                                               (equalp (aref (transformation-output-mask (instruction-transformation instruction)) fastest-dimension)
                                                       load-dimension))
                                     collect `(,(get-instruction-symbol instruction "_cached")
                                                (,(alexandria:format-symbol :keyword (string-upcase (format nil "coerce-~A" (cuda-type-string instruction)))) 0)))
                           ;; Load slow loads over cache
                           ,@(loop for instruction in slow-loads
                                   when (and (equalp (instruction-cuda-type instruction) (instruction-cuda-type (first slow-loads)))
                                             (equalp (aref (transformation-output-mask (instruction-transformation instruction)) fastest-dimension)
                                                     load-dimension))
                                   collect (let* ((buffer (load-instruction-buffer instruction))
                                                  (kernel-parameter (funcall buffer->kernel-parameter buffer))
                                                  (fastest-dimension (fastest-dimension iteration-scheme))
                                                  (transposed-instruction (transposed-load-instruction instruction load-dimension fastest-dimension)))
                                             `(let ((out-of-bounds? ,(get-oob-check transposed-instruction)))
                                                (set (aref shared-mem thread-idx-x thread-idx-y)
                                                     (if out-of-bounds?
                                                         0
                                                         (aref ,(linearize-instruction-transformation transposed-instruction
                                                                                                      buffer
                                                                                                      kernel-parameter
                                                                                                      (shape-independent-p iteration-scheme))))
                                                     (:inline-c "__syncthreads();")
                                                     (set ,(get-instruction-symbol instruction "_cached")
                                                          ,(linearize-instruction-transformation instruction
                                                                                                 buffer
                                                                                                 kernel-parameter
                                                                                                 (shape-independent-p iteration-scheme))
                                                          (aref shared-mem thread-idx-y thread-idx-x))))))
                           ;; Progamm body using cached variables
                           ,@kernel-body))
                       buffer->kernel-parameter)))

  (defmethod iteration-scheme-buffer-access ((iteration-scheme slow-coordinate-transposed-scheme) instruction buffer kernel-parameter)
    (if (find instruction (slow-loads iteration-scheme))
        (get-instruction-symbol instruction "_cached")
        ;; else unchached
      (call-next-method iteration-scheme instruction buffer kernel-parameter)))

  (defmethod iteration-scheme-shared-mem ((iteration-scheme slow-coordinate-transposed-scheme))
    (let ((shape (mapcar #'range-end (block-shape iteration-scheme))))
      (values (concatenate 'list (butlast shape) (mapcar #'1+ (last shape)))
              (instruction-cuda-type (first (slow-loads iteration-scheme))))))
  )
