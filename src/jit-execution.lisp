(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :petalisp.core
        :petalisp
        :cl
        :let-plus
        :cl-cuda)
  (:import-from :cl-cuda.api.kernel-manager :*kernel-manager*
                                            :kernel-manager-define-function
                                            :ensure-kernel-function-loaded)
  (:import-from :cl-cuda.driver-api :cu-device-ptr :cu-launch-kernel)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:import-from :petalisp-cuda.iteration-scheme :call-parameters :iteration-code :make-block-iteration-scheme :get-counter-vector)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array-strides :device-ptr)
  (:export :compile-kernel
           :execute-kernel))

(in-package petalisp-cuda.jitexecution)

(defun kernel-inputs (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-inputs
     (lambda (buffer) (push buffer buffers))
     kernel)
    buffers))

(defun kernel-outputs (kernel)
  (let ((buffers '()))
    (petalisp.ir:map-kernel-outputs
     (lambda (buffer) (push buffer buffers))
     kernel)
    buffers))

(defun kernel-instructions (kernel)
  (let ((rtn '()))
    (petalisp.ir:map-instructions
     (lambda (i) (push i rtn))
     kernel)
    rtn))

(defstruct (jit-function)
  kernel-symbol iteration-scheme shared-mem-bytes)

(defgeneric generate-iteration-scheme (kernel backend))

(defmethod generate-iteration-scheme (kernel backend)
  (make-block-iteration-scheme (kernel-iteration-space kernel)
                               (preferred-block-size backend)
                               (cuda-array-strides (buffer-storage (first (kernel-outputs kernel))))))

(defun generate-kernel-arguments (buffers)
  (mapcar (lambda (buffer idx) (list (format-symbol t "buffer-~A" idx) (cl-cuda.lang.type:array-type (cl-cuda-type-from-buffer buffer) 1)))
          buffers
          (iota (length buffers))))

(defun compile-kernel (kernel backend)
  (with-hash-table-memoization (kernel)
    (compile-cache backend)
    (let* ((buffers (kernel-buffers kernel))
          (kernel-arguments (generate-kernel-arguments buffers))
          (iteration-scheme (generate-iteration-scheme kernel backend)))
      (with-gensyms (function-name)
        (progn 
          (format t "~A~%" `(,(generate-kernel kernel kernel-arguments buffers iteration-scheme)))
          (kernel-manager-define-function *kernel-manager*
                                          (format-symbol t "~A" function-name) ;cl-cuda wants symbol with a package for the function name
                                          'void
                                          kernel-arguments
                                          `(,(generate-kernel kernel kernel-arguments buffers iteration-scheme)))
          (make-jit-function :kernel-symbol function-name
                             :iteration-scheme iteration-scheme
                             :shared-mem-bytes 0))))))

(defun fill-with-device-ptrs (ptr-array kernel-arguments)
  (loop for i from 0 to (1- (length kernel-arguments)) do
        (setf (cffi:mem-aref ptr-array 'cu-device-ptr i) (device-ptr (nth i kernel-arguments)))))

(defun run-compiled-function (compiled-function kernel-arguments)
  (let+ (((&slots kernel-symbol iteration-scheme shared-mem-bytes) compiled-function))
    (let ((parameters (call-parameters iteration-scheme)))
      (let ((hfunc (ensure-kernel-function-loaded *kernel-manager* kernel-symbol)))
        (cffi:with-foreign-object (kargs 'cu-device-ptr (length kernel-arguments))
          (progn
            (fill-with-device-ptrs kargs kernel-arguments)
            (destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) (getf parameters :grid-dim)
              (destructuring-bind (block-dim-x block-dim-y block-dim-z) (getf parameters :block-dim)
                (cu-launch-kernel hfunc
                                  grid-dim-x  grid-dim-y  grid-dim-z
                                  block-dim-x block-dim-y block-dim-z
                                  shared-mem-bytes cl-cuda.api.context:*cuda-stream*
                                  kargs (cffi:null-pointer))))))))))

(defmethod petalisp-cuda.backend:execute-kernel (kernel (backend cuda-backend))
  (let* ((buffers (kernel-buffers kernel))
         (arrays (mapcar #'buffer-storage buffers)))
    (run-compiled-function (compile-kernel kernel backend) arrays)))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  ;; Loop over domain
  (iteration-code iteration-scheme
                  (let* ((instructions (kernel-instructions kernel))
                         (buffer->kernel-argument (make-buffer->kernel-argument buffers kernel-arguments)))
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'instruction-number)
                                          buffer->kernel-argument))))

(defun linearize-instruction-transformation (instruction &optional buffer)
  (let* ((transformation (instruction-transformation instruction))
         (input-rank (transformation-input-rank transformation))
         (output-rank (transformation-output-rank transformation))
         (input-mask (transformation-input-mask transformation))
         (output-mask (transformation-output-mask transformation))
         (scalings (transformation-scalings transformation))
         (offsets (transformation-offsets transformation)))
    (let ((input (map 'list (lambda (a b) (or a b)) input-mask (get-counter-vector input-rank)))
          (strides (if buffer (cuda-array-strides (buffer-storage buffer)) (iota output-rank)))
          (starts (if buffer (cuda-array-strides (buffer-storage buffer)) (iota output-rank))))
      `(+ ,@(map 'list (lambda (a b) (or a b)) output-mask
                   (map 'list
                     (lambda (i o start s1 s2) `(* (+ ,i ,o ,start) ,s1 ,s2))
                     input offsets starts scalings strides))))))

(defun get-instruction-symbol (instruction)
  (format-symbol t "$~A"
                 (etypecase instruction
                     (number instruction)
                     (cons (instruction-number (cdr instruction)))
                     (instruction (instruction-number instruction)))))

(defun generate-instructions (instructions buffer->kernel-argument)
  (if instructions
      (let* ((instruction (pop instructions))
             ($i (get-instruction-symbol instruction)))
        (if (store-instruction-p instruction)
            (let ((buffer (store-instruction-buffer instruction)))
              `
              (progn
                (set
                 (aref ,(car (funcall buffer->kernel-argument buffer))
                       ,(linearize-instruction-transformation instruction buffer))
                 ,(get-instruction-symbol (first (instruction-inputs instruction))))
                ,(generate-instructions instructions buffer->kernel-argument)))
            `(let ((,$i ,(etypecase instruction
                           (call-instruction
                             `(,(map-call-operator (call-instruction-operator instruction))
                                ,@(mapcar #'get-instruction-symbol (instruction-inputs instruction))))
                           (iref-instruction
                             (linearize-instruction-transformation
                               (instruction-transformation instruction)))
                           (load-instruction
                             `(aref ,(car (funcall buffer->kernel-argument (load-instruction-buffer instruction)))
                                    ,(linearize-instruction-transformation instruction)))
                           ;(store-instruction
                             ;(let ((buffer (store-instruction-buffer instruction)))
                               ;`(set
                                  ;(aref ,(funcall buffer->kernel-argument buffer)
                                        ;,(linearize-instruction-transformation instruction buffer))
                                  ;,(get-instruction-symbol (first (instruction-inputs instruction))))))
                           )))
               ,(generate-instructions instructions buffer->kernel-argument))))
      '(return)))

(defun make-buffer->kernel-argument (buffers kernel-arguments)
    (lambda (buffer) (nth (position buffer buffers) kernel-arguments)))

(defun map-call-operator (operator)
  ;; LHS: Petalisp/code/type-inference/package.lisp
  ;; RHS: cl-cuda/src/lang/built-in.lisp
  (case operator 
    ((petalisp.type-inference:double-float+) '+)
    ((petalisp.type-inference:single-float+) '+)
    ((petalisp.type-inference:short-float+) '+)
    ((petalisp.type-inference:long-float+) '+)

    ((petalisp.type-inference:double-float-) '-)
    ((petalisp.type-inference:single-float-) '-)
    ((petalisp.type-inference:short-float-) '-)
    ((petalisp.type-inference:long-float-) '-)

    ((petalisp.type-inference:double-float*) '*)
    ((petalisp.type-inference:single-float*) '*)
    ((petalisp.type-inference:short-float*) '*)
    ((petalisp.type-inference:long-float*) '*)

    ((petalisp.type-inference:double-float/) '/)
    ((petalisp.type-inference:single-float/) '/)
    ((petalisp.type-inference:short-float/) '/)
    ((petalisp.type-inference:long-float/) '/)

    ((petalisp.type-inference:double-float=) '==)
    ((petalisp.type-inference:single-float=) '==)
    ((petalisp.type-inference:short-float=) '==)
    ((petalisp.type-inference:long-float=) '==)

    ((petalisp.type-inference:double-float>) '>)
    ((petalisp.type-inference:single-float>) '>)
    ((petalisp.type-inference:short-float>) '>)
    ((petalisp.type-inference:long-float>) '>)

    ((petalisp.type-inference:double-float<) '<)
    ((petalisp.type-inference:single-float<) '<)
    ((petalisp.type-inference:short-float<) '<)
    ((petalisp.type-inference:long-float<) '<)

    ((petalisp.type-inference:double-float<=) '<=)
    ((petalisp.type-inference:single-float<=) '<=)
    ((petalisp.type-inference:short-float<=) '<=)
    ((petalisp.type-inference:long-float<=) '<=)

    ((petalisp.type-inference:double-float>=) '>=)
    ((petalisp.type-inference:single-float>=) '>=)
    ((petalisp.type-inference:short-float>=) '>=)
    ((petalisp.type-inference:long-float>=) '>=)

    ((petalisp.type-inference:double-float-min) 'min)
    ((petalisp.type-inference:single-float-min) 'min)
    ((petalisp.type-inference:short-float-min) 'min)
    ((petalisp.type-inference:long-float-min) 'min)

    ((petalisp.type-inference:double-float-max) 'max)
    ((petalisp.type-inference:single-float-max) 'max)
    ((petalisp.type-inference:short-float-max) 'max)
    ((petalisp.type-inference:long-float-max) 'max)

    ((petalisp.type-inference:double-float-abs) 'abs)
    ((petalisp.type-inference:single-float-abs) 'abs)
    ((petalisp.type-inference:short-float-abs) 'abs)
    ((petalisp.type-inference:long-float-abs) 'abs)

    ((petalisp.type-inference:double-float-cos) 'cos)
    ((petalisp.type-inference:single-float-cos) 'cos)
    ((petalisp.type-inference:short-float-cos) 'cos)
    ((petalisp.type-inference:long-float-cos) 'cos)

    ((petalisp.type-inference:double-float-sin) 'sin)
    ((petalisp.type-inference:single-float-sin) 'sin)
    ((petalisp.type-inference:short-float-sin) 'sin)
    ((petalisp.type-inference:long-float-sin) 'sin)

    ((petalisp.type-inference:double-float-tan) 'tan)
    ((petalisp.type-inference:single-float-tan) 'tan)
    ((petalisp.type-inference:short-float-tan) 'tan)
    ((petalisp.type-inference:long-float-tan) 'tan)

    ;; Petalisp has no exp
    ;((petalisp.type-inference:double-float-exp) 'exp)
    ;((petalisp.type-inference:single-float-exp) 'exp)
    ;((petalisp.type-inference:short-float-exp) 'exp)
    ;((petalisp.type-inference:long-float-exp) 'exp)

    (t (error "Cannot convert Petalisp instruction ~A to cl-cuda instruction.
More copy paste required here!" operator))))

