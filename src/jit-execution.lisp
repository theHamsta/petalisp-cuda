(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :petalisp.core
        :petalisp
        :cl
        :let-plus
        :trivia
        :cl-cuda
        :petalisp-cuda.memory.memory-pool)
  (:import-from :cl-cuda.api.kernel-manager :make-kernel-manager
                :kernel-manager-define-function
                :ensure-kernel-function-loaded
                :kernel-manager-module-handle)
  (:import-from :cl-cuda.driver-api
                :cu-device-ptr
                :cu-launch-kernel)
  (:import-from :petalisp.utilities
                :with-hash-table-memoization)
  (:import-from :alexandria
                :when-let
                :format-symbol
                :iota
                :with-gensyms)
  (:import-from :petalisp-cuda.iteration-scheme
                :call-parameters
                :iteration-code
                :select-iteration-scheme
                :get-counter-vector
                :linearize-instruction-transformation
                :iteration-scheme-buffer-access)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-strides
                :device-ptr
                :make-cuda-array
                :cuda-array-p)
  (:import-from :petalisp-cuda.type-conversion
                :cl-cuda-type-from-buffer)
  (:import-from :petalisp-cuda.utils.cl-cuda
                :record-corresponding-event
                :wait-for-correspoding-event)
  (:import-from :petalisp-cuda.utils.petalisp
                :pass-as-scalar-argument-p)
  (:export :compile-kernel
           :execute-kernel))

(in-package petalisp-cuda.jitexecution)

(defun weird-rational-p (r)
  (and (rationalp r) (not (integerp r))))

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

(defun generate-iteration-scheme (kernel backend)
  (select-iteration-scheme (kernel-iteration-space kernel)
                           (preferred-block-size backend)
                           (cuda-array-strides (buffer-storage (first (kernel-outputs kernel))))))

(defun generate-kernel-parameters (buffers)
  (mapcar (lambda (buffer idx)
            (list (format-symbol t "buffer-~A" idx)
                  (let ((dtype (cl-cuda-type-from-buffer buffer)))
                   (if (pass-as-scalar-argument-p buffer)
                      dtype 
                      (cl-cuda.lang.type:array-type dtype 1)))))
          buffers
          (iota (length buffers))))

(defun upload-buffers-to-gpu (buffers backend)
  (mapcar (lambda (b) (upload-buffer-to-gpu b backend)) buffers))

;; TODO: make thread-safe it's possible that multiple threads try to upload the same buffer
(defun upload-buffer-to-gpu (buffer backend)
  (let ((storage (buffer-storage buffer)))
    ; Do not upload cuda arrays or scalars 
    (unless (or (cuda-array-p storage)
                (pass-as-scalar-argument-p buffer))
      (setf (buffer-storage buffer) (make-cuda-array storage
                                                     (cl-cuda-type-from-buffer buffer)
                                                     nil
                                                     (lambda (type size)
                                                       (memory-pool-allocate (cuda-memory-pool backend)
                                                                             type
                                                                             size))))
      (record-corresponding-event buffer (cuda-backend-event-map backend)))))

(defun make-multiple-value-let (lhs rhs more-rhs)
  `((,lhs ,rhs)
    ,@(loop for r in more-rhs
            for i from 1
            collect (list (format-symbol t "~A_~A" lhs i) r))))

;; Remove stuff that cl-cuda does not like
(defun remove-lispy-stuff (tree)
  (match tree
    ((cons (cons 'declare _) b) (remove-lispy-stuff b))
    ; unary +
    ((list '+ b) `(+ 0 ,(remove-lispy-stuff b)))
    ; binary floor/ceiling
    ((list 'floor a b) `(floor (/ ,(remove-lispy-stuff a) ,(remove-lispy-stuff b))))
    ((list 'ceiling a b) `(ceiling (/ ,(remove-lispy-stuff a) ,(remove-lispy-stuff b))))
    ; 1+/1-
    ((cons '1+ b) `(+ 1 ,@(remove-lispy-stuff b)))
    ((cons '1- b) `(+ (- 1) ,@(remove-lispy-stuff b)))
    ; ratios
    ((list '* (guard r (and (rationalp r) (not (integerp r)))) s) `(/ (* ,(numerator r)
                                                                         ,(remove-lispy-stuff s))
                                                                      ,(denominator r)))
    ; multiple values set
    ((cons 'let (cons (list (list lhs (list 'floor a b))) body)) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff `(floor ,a ,b)) (list (remove-lispy-stuff `(rem ,a ,b)))) ,@(remove-lispy-stuff body)))
    ((cons 'let (cons (list (list lhs (list 'ceiling a b))) body)) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff `(ceiling ,a ,b)) (list (remove-lispy-stuff `(rem ,a ,b)))) ,@(remove-lispy-stuff body)))
    ((guard (cons 'let (cons (list (cons lhs (cons rhs more-rhs))) body)) more-rhs) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff rhs) (remove-lispy-stuff more-rhs)) ,@(remove-lispy-stuff body)))
    ; rest
    ((guard a (atom a)) a)
    ((cons a b) (cons (remove-lispy-stuff a) (remove-lispy-stuff b)))))

(defun compile-kernel (kernel backend)
  (let* ((blueprint (kernel-blueprint kernel))
        (hash (list blueprint (kernel-iteration-space kernel) (mapcar #'buffer-shape (kernel-inputs kernel)))))
    ; TODO: compile we do not compile iteration-space independent
    (let* ((buffers (kernel-buffers kernel))
             (kernel-parameters (generate-kernel-parameters buffers))
             (iteration-scheme (generate-iteration-scheme kernel backend)))
        (let* ((kernel-symbol (format-symbol t "kernel-function")) ;cl-cuda wants symbol with a package for the function name
                 (kernel-manager (make-kernel-manager))
                 (generated-kernel `(,(remove-lispy-stuff (generate-kernel kernel
                                                                           kernel-parameters
                                                                           buffers
                                                                           iteration-scheme)))))  
            (when cl-cuda:*show-messages*
              (format t "Generated kernel:~%Arguments: ~A~%~A~%" kernel-parameters generated-kernel))

            ; TODO(seitz): probably faster compilation with all kernels of a run in one single kernel-manager
            (kernel-manager-define-function kernel-manager
                                            kernel-symbol
                                            'void
                                            kernel-parameters
                                            generated-kernel)
              ;(simple-error (e)
                ;(declare (ignore e))
                ;(remhash hash (compile-cache backend))
                ;(format t "failed to compile kernel:~%~A~%" generated-kernel)
                ;(error e)))
            (make-jit-function :kernel-symbol kernel-symbol
                               :iteration-scheme iteration-scheme
                               :dynamic-shared-mem-bytes 0
                               :kernel-manager kernel-manager
                               :kernel-parameters kernel-parameters
                               :kernel-body generated-kernel)))))

(defun fill-with-device-ptrs (ptrs-to-device-ptrs device-ptrs kernel-arguments kernel-parameters)
  (loop for i from 0 to (1- (length kernel-arguments)) do
        (let ((argument (nth i kernel-arguments)))
          (if (arrayp argument)
              (let ((ffi-type (intern (symbol-name (kernel-parameter-type (nth i kernel-parameters))) "KEYWORD")))
                (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr i) ffi-type)
                      (cffi:convert-to-foreign (if (weird-rational-p (aref argument))
                                                                     (coerce (aref argument) 'single-float)
                                                                     (aref argument)) ffi-type)))
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

(defun wait-for-buffer (buffer backend)
  (wait-for-correspoding-event buffer (cuda-backend-event-map backend))
  (map-buffer-inputs (lambda (kernel) (wait-for-correspoding-event kernel (cuda-backend-event-map backend))) buffer))

(defmethod petalisp-cuda.backend:execute-kernel (kernel (backend cuda-backend))
  (let ((buffers (kernel-buffers kernel)))
    (upload-buffers-to-gpu buffers backend)
    (map-kernel-inputs (lambda (buffer) (wait-for-buffer buffer backend)) kernel)
    (let ((arrays (mapcar #'buffer-storage buffers)))
      (run-compiled-function (compile-kernel kernel backend) arrays))
    (record-corresponding-event kernel (cuda-backend-event-map backend))))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  ;; Loop over domain
  (iteration-code iteration-scheme
                  (let* ((instructions (kernel-instructions kernel))
                         (buffer->kernel-parameter (make-buffer->kernel-parameter buffers kernel-arguments)))
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'instruction-number)
                                           buffer->kernel-parameter
                                           iteration-scheme))))

(defun get-instruction-symbol (instruction)
  (trivia:match instruction
    ;; weird multiple value instruction
    ((guard (cons a b ) (> a 0)) (format-symbol t "$~A_~A" (instruction-number b) a))   
    ;; normal instruction
    (_
      (format-symbol t "$~A"
                     (etypecase instruction
                       (number instruction)
                       (cons (instruction-number (cdr instruction)))
                       (instruction (instruction-number instruction)))))))

(defun kernel-parameter-name (kernel-parameter)
  (car kernel-parameter))

(defun kernel-parameter-type (kernel-parameter)
  (cadr kernel-parameter))

(defun buffer-access (buffer buffer->kernel-parameter instruction iteration-scheme)
  (assert (functionp buffer->kernel-parameter))
  (let ((kernel-parameter (kernel-parameter-name (funcall buffer->kernel-parameter buffer))))
    (if (pass-as-scalar-argument-p buffer)
        kernel-parameter
        (iteration-scheme-buffer-access iteration-scheme instruction buffer kernel-parameter))))

(defun generate-instructions (instructions buffer->kernel-parameter iteration-scheme)
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
                                 instruction
                                 iteration-scheme)
                 ,(get-instruction-symbol (first (instruction-inputs instruction))))
               ;; Rest
               ,(generate-instructions instructions buffer->kernel-parameter iteration-scheme)))
          ;; Instructions that produce a result in C code
          `(let ((,$i ,@(etypecase instruction
                         ;; Call instructions
                         (call-instruction
                           (let ((arguments (mapcar #'get-instruction-symbol (instruction-inputs instruction))))
                             (multiple-value-bind (cuda-function inlined-expression) (map-call-operator (call-instruction-operator instruction) arguments)
                               (if cuda-function
                                   `((,cuda-function ,@arguments))
                                   inlined-expression))))
                         ;; Iref instructions
                         (iref-instruction
                           (list (linearize-instruction-transformation instruction)))
                         ;; Load instructions
                         (load-instruction
                           (list (buffer-access (load-instruction-buffer instruction)
                                          buffer->kernel-parameter
                                          instruction
                                          iteration-scheme))))))
             ;; Rest
             ,(generate-instructions instructions buffer->kernel-parameter iteration-scheme))))
      '(progn)))

(defun make-buffer->kernel-parameter (buffers kernel-parameters)
  (lambda (buffer) (nth (position buffer buffers) kernel-parameters)))

