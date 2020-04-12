(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :cl)
  (:import-from :cl-cuda :void :set)
  (:import-from :cl-cuda.api.kernel-manager :*kernel-manager*
                                            :kernel-manager-define-function)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:import-from :petalisp-cuda.indexing :call-parameters :iteration-code :make-block-iteration-scheme)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array-strides)
  (:export :compile-kernel))

(in-package petalisp-cuda.jitexecution)


(defstruct jit-function ()
  (kernel-symbol iteration-scheme))

(defgeneric generate-iteration-scheme ())

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

(defun execute-kernel (kernel (backend cuda-backend))
    (let ((compiled-function compile-kernel (kernel backend)))))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  ;; Loop over domain
  (iteration-code iteration-scheme
                  ;; kernel body
                  (let* ((instructions (petalisp.ir:map-instructions #'identity kernel))
                        (buffer->kernel-argument (make-buffer->kernel-argument buffers kernel-arguments))
                        (map-input (lambda (input) (map-input input buffer->kernel-argument))))
                    (loop for instruction in (sort instructions #'< :key #'petalisp.ir:instruction-number)
                          count t into i
                          collect
                          (let (($i (format-symbol "$~A" i)))
                           (etypecase instruction
                            (petalisp.ir:call-instruction
                              `(,(map-call-operator (petalisp.ir:call-instruction-operator instruction))
                                 ,@(petalisp.ir:map-instruction-inputs map-input instruction)))
                            (petalisp.ir:iref-instruction
                              (format nil "iref ~S~%"
                                      (petalisp.ir:instruction-transformation instruction)))
                            (petalisp.ir:load-instruction
                              `(set 
                                      (petalisp.type-inference:type-specifier
                                        (buffer->kernel-argument (petalisp.ir:load-instruction-buffer instruction)))
                                      (petalisp.ir:instruction-transformation instruction)))
                            (petalisp.ir:store-instruction
                              (format nil "store ~S ~S ~S~%"
                                      (petalisp.type-inference:type-specifier
                                        (petalisp.ir:buffer-ntype
                                          (petalisp.ir:store-instruction-buffer instruction)))
                                      (petalisp.ir:instruction-transformation instruction)
                                      (simplify-input
                                        (first
                                          (petalisp.ir:instruction-inputs instruction)))))))))))

(defun map-call-operator (operator)
  operator)

(defun map-input (input buffer->kernel-argument)
  input)

(defun make-buffer->kernel-argument (buffers kernel-arguments)
    (lambda (buffer) (nth (position buffer buffers) kernel-arguments)))
