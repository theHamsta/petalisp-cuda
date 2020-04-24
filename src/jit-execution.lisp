(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :cl
        :cl-cuda)
  (:import-from :cl-cuda.api.kernel-manager :*kernel-manager*
                                            :kernel-manager-define-function)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:import-from :petalisp-cuda.indexing :call-parameters :iteration-code :make-block-iteration-scheme)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array-strides)
  (:export :compile-kernel))

(in-package petalisp-cuda.jitexecution)


(defstruct (jit-function)
  kernel-symbol iteration-scheme)

(defgeneric generate-iteration-scheme (kernel backend))

(defmethod generate-iteration-scheme (kernel backend)
  (make-block-iteration-scheme (petalisp.ir:kernel-iteration-space kernel)
                               (preferred-block-size backend)
                               (cuda-array-strides (storage (first (petalisp.ir:map-kernel-outputs #'identity kernel))))))

(defun generate-kernel-arguments (buffers)
  (mapcar (lambda (buffer idx) (list (format-symbol "buffer-~A" idx) (cl-cuda-type-from-buffer buffer)))
          buffers
          (iota (length buffers))))

(defun compile-kernel (kernel backend)
  (with-hash-table-memoization (kernel)
    (compile-cache backend)
    (let* ((buffers (petalisp.ir:kernel-buffers kernel))
          (kernel-arguments (generate-kernel-arguments buffers))
          (iteration-scheme (generate-iteration-scheme kernel backend)))
      (with-gensyms (function-name)
        (progn 
          (kernel-manager-define-function *kernel-manager*
                                          function-name
                                          'void
                                          kernel-arguments
                                          (generate-kernel kernel kernel-arguments buffers iteration-scheme))
          (make-jit-function :kernel-symbol function-name
                             :iteration-scheme iteration-scheme))))))

(defmethod execute-kernel (kernel (backend cuda-backend))
    (let ((compiled-function compile-kernel (kernel backend)))))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  ;; Loop over domain
  (iteration-code iteration-scheme
                  (let* ((instructions (petalisp.ir:map-instructions #'identity kernel))
                         (buffer->kernel-argument (make-buffer->kernel-argument buffers kernel-arguments))
                         (map-input (lambda (input) (map-input input buffer->kernel-argument))))
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'petalisp.ir:instruction-number)
                                          buffer->kernel-argument
                                          map-input))))

(defun generate-instructions (instructions map-input &optional (i 0))
  (let (instruction (pop instructions))
    ($i (format-symbol "$~A" i)))
  `(let (($i ,(etypecase instruction
                (petalisp.ir:call-instruction
                  `(set (#|c-type|#     ,(cl-cuda-type-from-buffer (petalisp.ir:load-instruction-buffer instruction))
                         #|identifier|# $i
                         #|value|#      
                         `(,(map-call-operator (petalisp.ir:call-instruction-operator instruction))
                            ,@(petalisp.ir:map-instruction-inputs map-input instruction)))
                        (petalisp.ir:iref-instruction
                          (format nil "iref ~S~%"
                                  (petalisp.ir:instruction-transformation instruction)))
                        (petalisp.ir:load-instruction
                          `(set (#|c-type|#     ,(cl-cuda-type-from-buffer (petalisp.ir:load-instruction-buffer instruction))
                                 #|identifier|# $i
                                 #|value|#      (petalisp.ir:instruction-transformation instruction))))
                        (petalisp.ir:store-instruction
                          (format nil "store ~S ~S ~S~%"
                                  (petalisp.type-inference:type-specifier
                                    (petalisp.ir:buffer-ntype
                                      (petalisp.ir:store-instruction-buffer instruction)))
                                  (petalisp.ir:instruction-transformation instruction)
                                  (simplify-input
                                    (first
                                      (petalisp.ir:instruction-inputs instruction))))))))
             ,(generate-instructions instruction map-input (1+ i))))))

(defun map-call-operator (operator)
  operator)

(defun map-input (input buffer->kernel-argument)
  input)

(defun make-buffer->kernel-argument (buffers kernel-arguments)
    (lambda (buffer) (nth (position buffer buffers) kernel-arguments)))
