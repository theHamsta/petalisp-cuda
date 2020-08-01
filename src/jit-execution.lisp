(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :petalisp.core
        :petalisp
        :cl
        :let-plus
        :trivia
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
  (:import-from :petalisp-cuda.type-conversion :cl-cuda-type-from-buffer)
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
  kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-manager kernel-parameters kernel-body)

(defgeneric generate-iteration-scheme (kernel backend))

(defmethod generate-iteration-scheme (kernel backend)
  (make-block-iteration-scheme (kernel-iteration-space kernel)
                               (preferred-block-size backend)
                               (cuda-array-strides (buffer-storage (first (kernel-outputs kernel))))))

(defun pass-as-scalar-argument-p (buffer)
  (and (scalar-buffer-p buffer)
       (arrayp (buffer-storage buffer))
       ; we pass the arguments with a pointer array right now
       ; and all are types are 8 bytes or smaller
       (= 8 (cffi:foreign-type-size :pointer))))

(defun generate-kernel-parameters (buffers)
  (mapcar (lambda (buffer idx)
            (list (format-symbol t "buffer-~A" idx)
                  (let ((dtype (cl-cuda-type-from-buffer buffer)))
                   (if (pass-as-scalar-argument-p buffer)
                      dtype 
                      (cl-cuda.lang.type:array-type dtype 1)))))
          buffers
          (iota (length buffers))))

(defun scalar-buffer-p (buffer)
  (= 0 (shape-rank (buffer-shape buffer))))

(defun upload-buffer-to-gpu (buffer)
  (let ((storage (buffer-storage buffer)))
    ; Do not upload cuda arrays or sclars 
    (unless (or (cuda-array-p storage)
                (pass-as-scalar-argument-p buffer))
      (setf (buffer-storage buffer) (make-cuda-array storage (cl-cuda-type-from-buffer buffer))))))

;; Remove stuff that cl-cuda does not like
(defun remove-lispy-stuff (tree)
  (match tree
    ((cons (cons 'declare _) b) (remove-lispy-stuff b))
    ; unary +
    ((list '+ b) `(+ 0 ,(remove-lispy-stuff b)))
    ; 1+/1-
    ((cons '1+ b) `(+ 1 ,@(remove-lispy-stuff b)))
    ((cons '1- b) `(+ (- 1) ,@(remove-lispy-stuff b)))
    ; ratios
    ((list '* (guard r (and (rationalp r) (not (integerp r)))) s) `(/ (* ,(numerator r)
                                                                         ,(remove-lispy-stuff s))
                                                                      ,(denominator r)))
    ; rest
    ((guard a (atom a)) a)
    ((cons a b) (cons (remove-lispy-stuff a) (remove-lispy-stuff b)))))

(defun upload-buffers-to-gpu (buffers)
  (mapcar #'upload-buffer-to-gpu buffers))

(defun compile-kernel (kernel backend)
  (let ((blueprint (kernel-blueprint kernel)))
    ; TODO: compile we do not compile iteration-space independent
    (petalisp.utilities:with-hash-table-memoization (kernel)
      ;((format nil "~S~S" blueprint (kernel-iteration-space kernel)))
      (compile-cache backend)
      (let* ((buffers (kernel-buffers kernel))
             (kernel-parameters (generate-kernel-parameters buffers))
             (iteration-scheme (generate-iteration-scheme kernel backend)))
        (with-gensyms (function-name)
          (let* ((kernel-symbol (format-symbol t "kernel-function")) ;cl-cuda wants symbol with a package for the function name
                 (kernel-manager (make-kernel-manager))
                 (generated-kernel `(,(remove-lispy-stuff (generate-kernel kernel
                                                                           kernel-parameters
                                                                           buffers
                                                                           iteration-scheme)))))  
            (when cl-cuda:*show-messages*
              (format t "Generated kernel ~A:~%Arguments: ~A~%~A~%" function-name kernel-parameters generated-kernel))

            ; TODO(seitz): probably faster compilation with all kernels of a run in one single kernel-manager
            (kernel-manager-define-function kernel-manager
                                            kernel-symbol
                                            'void
                                            kernel-parameters
                                            generated-kernel)
            (make-jit-function :kernel-symbol kernel-symbol
                               :iteration-scheme iteration-scheme
                               :dynamic-shared-mem-bytes 0
                               :kernel-manager kernel-manager
                               :kernel-parameters kernel-parameters
                               :kernel-body generated-kernel)))))))

(defun fill-with-device-ptrs (ptrs-to-device-ptrs device-ptrs kernel-arguments kernel-parameters)
  (loop for i from 0 to (1- (length kernel-arguments)) do
        (let ((argument (nth i kernel-arguments)))
          (if (arrayp argument)
              (let ((ffi-type (intern (symbol-name (kernel-parameter-type (nth i kernel-parameters))) "KEYWORD")))
                (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr i) ffi-type)
                      (cffi:convert-to-foreign (aref argument) ffi-type)))
              (setf (cffi:mem-aref device-ptrs 'cu-device-ptr i) (device-ptr argument))))
        (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) i) (cffi:mem-aptr device-ptrs 'cu-device-ptr i))))

(defun run-compiled-function (compiled-function kernel-arguments)
  (let+ (((&slots kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-manager kernel-parameters) compiled-function))
    (let ((parameters (call-parameters iteration-scheme)))
      (let ((hfunc (ensure-kernel-function-loaded kernel-manager kernel-symbol))
            (nargs (length kernel-arguments))
            (extra-arguments (cffi:null-pointer))) ; has to be NULL since we use the kernel-args parameter
        (cffi:with-foreign-objects ((ptrs-to-device-ptrs '(:pointer :pointer) nargs) (device-ptrs 'cu-device-ptr nargs))
          (fill-with-device-ptrs ptrs-to-device-ptrs device-ptrs kernel-arguments kernel-parameters)
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
                         (buffer->kernel-parameter (make-buffer->kernel-parameter buffers kernel-arguments)))
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'instruction-number)
                                           buffer->kernel-parameter))))

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

(defun kernel-parameter-name (kernel-parameter)
  (car kernel-parameter))

(defun kernel-parameter-type (kernel-parameter)
  (cadr kernel-parameter))

(defun buffer-access (buffer buffer->kernel-parameter instruction)
  (assert (functionp buffer->kernel-parameter))
  (let ((kernel-parameter (kernel-parameter-name (funcall buffer->kernel-parameter buffer))))
    (if (pass-as-scalar-argument-p buffer)
        kernel-parameter
        `(aref ,kernel-parameter ,(linearize-instruction-transformation instruction buffer)))))

(defun generate-instructions (instructions buffer->kernel-parameter)
  (if instructions
      (let* ((instruction (pop instructions))
             ($i (get-instruction-symbol instruction)))
        (if (store-instruction-p instruction)
            ;; Instructions that produce no result in C code
          (let ((buffer (store-instruction-buffer instruction)))
            `(progn
               ;; Store instructions
               (set
                 ,(buffer-access buffer
                                 buffer->kernel-parameter
                                 instruction)
                 ,(get-instruction-symbol (first (instruction-inputs instruction))))
               ;; Rest
               ,(generate-instructions instructions buffer->kernel-parameter)))
          ;; Instructions that produce a result in C code
          `(let ((,$i ,(etypecase instruction
                         ;; Call instructions
                         (call-instruction
                           (let ((arguments (mapcar #'get-instruction-symbol (instruction-inputs instruction))))
                             (multiple-value-bind (cuda-function inlined-expression) (map-call-operator (call-instruction-operator instruction) arguments)
                               (if cuda-function
                                   `(,cuda-function ,@arguments)
                                   inlined-expression))))
                         ;; Iref instructions
                         (iref-instruction
                           (linearize-instruction-transformation instruction))
                         ;; Load instructions
                         (load-instruction
                           (buffer-access (load-instruction-buffer instruction)
                                          buffer->kernel-parameter
                                          instruction)))))
             ;; Rest
             ,(generate-instructions instructions buffer->kernel-parameter))))
      '(progn)))

(defun make-buffer->kernel-parameter (buffers kernel-parameters)
  (lambda (buffer) (nth (position buffer buffers) kernel-parameters)))

(defun map-call-operator (operator arguments)
  ;; LHS: Petalisp/code/type-inference/package.lisp
  ;; RHS: cl-cuda/src/lang/built-in.lisp
  (case operator 
    (+ '+)
    ((petalisp.type-inference:double-float+) '+)
    ((petalisp.type-inference:single-float+) '+)
    ((petalisp.type-inference:short-float+) '+)
    ((petalisp.type-inference:long-float+) '+)

    (- '+)
    ((petalisp.type-inference:double-float-) '-)
    ((petalisp.type-inference:single-float-) '-)
    ((petalisp.type-inference:short-float-) '-)
    ((petalisp.type-inference:long-float-) '-)

    (* '*)
    ((petalisp.type-inference:double-float*) '*)
    ((petalisp.type-inference:single-float*) '*)
    ((petalisp.type-inference:short-float*) '*)
    ((petalisp.type-inference:long-float*) '*)

    (/ '/)
    ((petalisp.type-inference:double-float/) '/)
    ((petalisp.type-inference:single-float/) '/)
    ((petalisp.type-inference:short-float/) '/)
    ((petalisp.type-inference:long-float/) '/)

    (= '==)
    (CMPEQ '==)
    ((petalisp.type-inference:double-float=) '==)
    ((petalisp.type-inference:single-float=) '==)
    ((petalisp.type-inference:short-float=) '==)
    ((petalisp.type-inference:long-float=) '==)

    (> '>)
    ((petalisp.type-inference:double-float>) '>)
    ((petalisp.type-inference:single-float>) '>)
    ((petalisp.type-inference:short-float>) '>)
    ((petalisp.type-inference:long-float>) '>)

    (< '<)
    ((petalisp.type-inference:double-float<) '<)
    ((petalisp.type-inference:single-float<) '<)
    ((petalisp.type-inference:short-float<) '<)
    ((petalisp.type-inference:long-float<) '<)

    (<= '<=)
    ((petalisp.type-inference:double-float<=) '<=)
    ((petalisp.type-inference:single-float<=) '<=)
    ((petalisp.type-inference:short-float<=) '<=)
    ((petalisp.type-inference:long-float<=) '<=)

    (>= '>=)
    ((petalisp.type-inference:double-float>=) '>=)
    ((petalisp.type-inference:single-float>=) '>=)
    ((petalisp.type-inference:short-float>=) '>=)
    ((petalisp.type-inference:long-float>=) '>=)

    (min 'min)
    ((petalisp.type-inference:double-float-min) 'min)
    ((petalisp.type-inference:single-float-min) 'min)
    ((petalisp.type-inference:short-float-min) 'min)
    ((petalisp.type-inference:long-float-min) 'min)

    (max 'max)
    ((petalisp.type-inference:double-float-max) 'max)
    ((petalisp.type-inference:single-float-max) 'max)
    ((petalisp.type-inference:short-float-max) 'max)
    ((petalisp.type-inference:long-float-max) 'max)

    (abs 'abs)
    ((petalisp.type-inference:double-float-abs) 'abs)
    ((petalisp.type-inference:single-float-abs) 'abs)
    ((petalisp.type-inference:short-float-abs) 'abs)
    ((petalisp.type-inference:long-float-abs) 'abs)

    (cos 'cos)
    ((petalisp.type-inference:double-float-cos) 'cos)
    ((petalisp.type-inference:single-float-cos) 'cos)
    ((petalisp.type-inference:short-float-cos) 'cos)
    ((petalisp.type-inference:long-float-cos) 'cos)

    (sin 'sin)
    ((petalisp.type-inference:double-float-sin) 'sin)
    ((petalisp.type-inference:single-float-sin) 'sin)
    ((petalisp.type-inference:short-float-sin) 'sin)
    ((petalisp.type-inference:long-float-sin) 'sin)

    (tan 'tan)
    ((petalisp.type-inference:double-float-tan) 'tan)
    ((petalisp.type-inference:single-float-tan) 'tan)
    ((petalisp.type-inference:short-float-tan) 'tan)
    ((petalisp.type-inference:long-float-tan) 'tan)

    (exp 'exp)
    ((petalisp.type-inference::double-float-exp) 'exp)
    ((petalisp.type-inference::single-float-exp) 'exp)
    ((petalisp.type-inference::short-float-exp) 'exp)
    ((petalisp.type-inference::long-float-exp) 'exp)

    (t (let ((source-form (function-lambda-expression operator)))
         (if source-form
             (let* ((lambda-arguments (nth 1 source-form))
                    (lambda-body? (last source-form))
                    (lambda-body (if (equal 'BLOCK (caar lambda-body?)) (car (last (first lambda-body?))) (car lambda-body?))))
               (when cl-cuda:*show-messages*
                 (format t "Creating kernel lambda: ~A~%" lambda-body))
               (loop for ir-arg in arguments
                     for arg in lambda-arguments do
                     (setf lambda-body (subst ir-arg arg lambda-body)))
               (values nil lambda-body))
         (error "Cannot convert Petalisp instruction ~A to cl-cuda instruction.
More copy paste required here!~%
You may also try to compile ~A with (debug 3) so that petalisp-cuda can retrieve its source from." operator operator))))))

