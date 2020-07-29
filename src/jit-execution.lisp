(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :petalisp.core
        :petalisp
        :cl
        :let-plus
        :cl-cuda)
  (:import-from :cl-cuda.api.kernel-manager :make-kernel-manager
                                            :kernel-manager-define-function
                                            :ensure-kernel-function-loaded
                                            :kernel-manager-module-handle)
  (:import-from :cl-cuda.driver-api :cu-device-ptr :cu-launch-kernel)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:import-from :petalisp-cuda.iteration-scheme :call-parameters :iteration-code :make-block-iteration-scheme :get-counter-vector)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array-strides :device-ptr :make-cuda-array :cuda-array-p)
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
  kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-manager)

(defgeneric generate-iteration-scheme (kernel backend))

(defmethod generate-iteration-scheme (kernel backend)
  (make-block-iteration-scheme (kernel-iteration-space kernel)
                               (preferred-block-size backend)
                               (cuda-array-strides (buffer-storage (first (kernel-outputs kernel))))))

(defun generate-kernel-arguments (buffers)
  (mapcar (lambda (buffer idx) (list (format-symbol t "buffer-~A" idx) (cl-cuda.lang.type:array-type (cl-cuda-type-from-buffer buffer) 1)))
          buffers
          (iota (length buffers))))

(defun upload-buffer-to-gpu (buffer)
  (let ((storage (buffer-storage buffer)))
    ; TODO: optimization for scalars, not upload via global memory
    (unless (cuda-array-p storage)
     (setf (buffer-storage buffer) (make-cuda-array storage (cl-cuda-type-from-buffer buffer))))))

(defun upload-buffers-to-gpu (buffers)
  (mapcar #'upload-buffer-to-gpu buffers))

(defun compile-kernel (kernel backend)
  (petalisp.utilities:with-hash-table-memoization (kernel) ; TODO find out what I need to hash from kernel
      (compile-cache backend)
    (let* ((buffers (kernel-buffers kernel))
           (kernel-arguments (generate-kernel-arguments buffers))
           (iteration-scheme (generate-iteration-scheme kernel backend)))
      (with-gensyms (function-name)
        (let* ((kernel-symbol (format-symbol (make-package function-name) "~A" function-name)) ;cl-cuda wants symbol with a package for the function name
               (generated-kernel `(,(generate-kernel kernel kernel-arguments buffers iteration-scheme)))
               (kernel-manager (make-kernel-manager)))  
            (when cl-cuda:*show-messages*
              (format t "Generated kernel ~A:~%Arguments: ~A~%~A~%" function-name kernel-arguments generated-kernel))
            (kernel-manager-define-function kernel-manager
                                            kernel-symbol
                                            'void
                                            kernel-arguments
                                            generated-kernel)
            (make-jit-function :kernel-symbol kernel-symbol
                               :iteration-scheme iteration-scheme
                               :dynamic-shared-mem-bytes 0
                               :kernel-manager kernel-manager))))))

(defun fill-with-device-ptrs (ptrs-to-device-ptrs device-ptrs kernel-arguments)
  (loop for i from 0 to (1- (length kernel-arguments)) do
        (setf (cffi:mem-aref device-ptrs 'cu-device-ptr i) (device-ptr (nth i kernel-arguments)))
        (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) i) (cffi:mem-aptr device-ptrs 'cu-device-ptr i))))

(defun run-compiled-function (compiled-function kernel-arguments)
  (let+ (((&slots kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-manager) compiled-function))
    (let ((parameters (call-parameters iteration-scheme)))
      (let ((hfunc (ensure-kernel-function-loaded kernel-manager kernel-symbol))
            (nargs (length kernel-arguments))
            (extra-arguments (cffi:null-pointer))) ; has to be NULL since we use the kernel-args parameter
        (cffi:with-foreign-objects ((ptrs-to-device-ptrs '(:pointer :pointer) nargs) (device-ptrs 'cu-device-ptr nargs))
          (fill-with-device-ptrs ptrs-to-device-ptrs device-ptrs kernel-arguments)
          (destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) (getf parameters :grid-dim)
            (destructuring-bind (block-dim-x block-dim-y block-dim-z) (getf parameters :block-dim)
              (when cl-cuda:*show-messages*
                (format t "Calling kernel ~A with call parameters ~A~%" kernel-symbol parameters))
              (cu-launch-kernel hfunc
                                grid-dim-x  grid-dim-y  grid-dim-z
                                block-dim-x block-dim-y block-dim-z
                                dynamic-shared-mem-bytes
                                cl-cuda.api.context:*cuda-stream*
                                ptrs-to-device-ptrs
                                extra-arguments))))))))

(defmethod petalisp-cuda.backend:execute-kernel (kernel (backend cuda-backend))
  (let ((buffers (kernel-buffers kernel)))
    (upload-buffers-to-gpu buffers)
    (let ((arrays (mapcar #'buffer-storage buffers)))
      (run-compiled-function (compile-kernel kernel backend) arrays))))

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
         (strides (if buffer (cuda-array-strides (buffer-storage buffer)) (make-list input-rank :initial-element 1)))
         (index-space (get-counter-vector input-rank) )
         (transformed (transform index-space transformation)))
    (let ((rtn `(+ ,@(mapcar (lambda (a b) `(* ,a ,b)) transformed strides))))
      (if (= (length rtn) 1) 0 rtn))))

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
              `(progn
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
                                    ,(linearize-instruction-transformation instruction (load-instruction-buffer instruction)))))))
               ,(generate-instructions instructions buffer->kernel-argument))))
      '(progn)))

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

