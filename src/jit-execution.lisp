(defpackage petalisp-cuda.jit-execution
 (:use :petalisp-cuda.backend
       :petalisp.ir
       :petalisp.core
        :petalisp
        :cl
        :let-plus
        :trivia
        :cl-cuda
        :petalisp-cuda.memory.memory-pool
        :petalisp-cuda.custom-op)
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
                :if-let
                :switch
                :format-symbol
                :iota
                :with-gensyms)
  (:import-from :petalisp-cuda.iteration-scheme
                :call-parameters
                :iteration-code
                :select-iteration-scheme
                :get-counter-vector
                :linearize-instruction-transformation
                :iteration-scheme-buffer-access
                :shape-independent-p
                :generic-offsets-p
                :iteration-space
                :get-instruction-symbol
                :kernel-parameter-name
                :kernel-parameter-type)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-strides
                :cuda-array-device
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
          :execute-kernel
          :*device-function-mapping*))

(in-package petalisp-cuda.jit-execution)

(defun weird-rational-p (r)
 (and (rationalp r) (not (integerp r))))

(defun get-offset-vector (kernel)
  (apply #'concatenate `(list ,@(loop for i in (kernel-instructions kernel)
                                     when (not (typep i 'call-instruction))
                                     collect (transformation-offsets (instruction-transformation i))))))

(defun filtered-offset-vector (kernel)
  (remove-if-not (lambda (a) (> (abs a) 2)) (remove-duplicates (get-offset-vector kernel))))

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
  kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-manager kernel-parameters kernel-body hfunc)

;; TODO: make this generic for possible C backend?
(defun generate-iteration-scheme (kernel backend)
  (declare (ignore backend))
  (select-iteration-scheme kernel
                           (kernel-iteration-space kernel)
                           (cuda-array-strides (buffer-storage (first (kernel-outputs kernel))))))

(defun generate-kernel-parameters (buffers iteration-scheme filtered-offset-vector)
  (concatenate 'list (mapcar (lambda (buffer idx)
                               (if (pass-as-scalar-argument-p buffer)
                                   (list (format-symbol t "buffer-~A" idx)
                                         (cl-cuda-type-from-buffer buffer))
                                   (list (format-symbol t "buffer-~A" idx)
                                         (cl-cuda.lang.type:array-type (cl-cuda-type-from-buffer buffer) 1)
                                         #| :restrict ;; <- TODO|#)))
                             buffers
                             (iota (length buffers)))
               (when (shape-independent-p iteration-scheme)
                 (loop for s in (shape-ranges (iteration-space iteration-scheme))
                       for i from 0
                       collect `(,(format-symbol t "iteration-start-~A" i) int)
                       collect `(,(format-symbol t "iteration-end-~A" i) int)))
               (when (shape-independent-p iteration-scheme)
                 (apply #'concatenate `(list ,@(loop for b in buffers
                                                     for i from 0
                                                     collect (loop for j below (shape-rank (buffer-shape b))
                                                                   collect `(,(format-symbol t "buffer-~A-stride-~A" i j) int))))))
               (when (generic-offsets-p iteration-scheme)
                 (loop for s in filtered-offset-vector
                       for i from 0
                       collect `(,(format-symbol t "offset-~A" i) int)))))

(defun upload-buffers-to-gpu (buffers backend)
  (mapcar (lambda (b) (upload-buffer-to-gpu b backend)) buffers))

;; TODO: make thread-safe it's possible that multiple threads try to upload the same buffer
(defun upload-buffer-to-gpu (buffer backend)
  (let ((storage (buffer-storage buffer)))
    ; Do not upload cuda arrays on same device or scalars 
    (unless (or (and (cuda-array-p storage)
                     (= cl-cuda:*cuda-device* (cuda-array-device storage)))
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
(defun remove-lispy-stuff (tree offset-vector)
  (match tree
    ((cons (cons 'declare _) b) (remove-lispy-stuff b offset-vector))
    ; unary +
    ((list '+ b) `(+ 0 ,(remove-lispy-stuff b offset-vector)))
    ((list (list 'and a b)) `(and ,(remove-lispy-stuff a offset-vector) ,(remove-lispy-stuff b offset-vector)))
    ((guard (list* 'and a b c) c) `(and ,(remove-lispy-stuff a offset-vector) (and ,(remove-lispy-stuff b offset-vector) ,(remove-lispy-stuff c offset-vector))))
    ; binary floor/ceiling
    ((list 'floor a b) `(floor (/ ,(remove-lispy-stuff a offset-vector) ,(remove-lispy-stuff b offset-vector))))
    ((list 'ceiling a b) `(ceiling (/ ,(remove-lispy-stuff a offset-vector) ,(remove-lispy-stuff b offset-vector))))
    ; 1+/1-
    ((cons '1+ b) `(+ 1 ,@(remove-lispy-stuff b offset-vector)))
    ((cons '1- b) `(+ (- 1) ,@(remove-lispy-stuff b offset-vector)))
    ; ratios
    ((list '* (guard r (and (rationalp r) (not (integerp r)))) s) `(/ (* ,(numerator r)
                                                                         ,(remove-lispy-stuff s offset-vector))
                                                                      ,(denominator r)))
    ; multiple values set
    ((cons 'let (cons (list (list lhs (list 'floor a b))) body)) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff `(floor ,a ,b) offset-vector) (list (remove-lispy-stuff `(rem ,a ,b) offset-vector) offset-vector)) ,@(remove-lispy-stuff body offset-vector)))
    ((cons 'let (cons (list (list lhs (list 'ceiling a b))) body)) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff `(ceiling ,a ,b) offset-vector) (list (remove-lispy-stuff `(rem ,a ,b) offset-vector))) ,@(remove-lispy-stuff body offset-vector)))
    ((guard (cons 'let (cons (list (cons lhs (cons rhs more-rhs))) body)) more-rhs) `(let ,(make-multiple-value-let lhs (remove-lispy-stuff rhs offset-vector) (remove-lispy-stuff more-rhs offset-vector)) ,@(remove-lispy-stuff body offset-vector)))
    ; replace numbers by symbols
    ((guard a (numberp a)) (or (loop for o in offset-vector
                                     for i from 0
                                     when (= o a)
                                     return (format-symbol t "offset-~A" i))  a))
    ; rest
    ((guard a (atom a)) a)
    ((cons a b) (cons (remove-lispy-stuff a offset-vector) (remove-lispy-stuff b offset-vector)))))

(defun compile-kernel (kernel backend)
  (let* ((blueprint (kernel-blueprint kernel))
         (hash (list blueprint
                     (kernel-iteration-space kernel)
                     (mapcar #'buffer-shape (kernel-buffers kernel))
                     (get-offset-vector kernel)
                     ;; call-instruction-operator is nil in blueprints when functionp
                     (mapcar (lambda (i) (when (call-instruction-p i) (call-instruction-operator i))) (kernel-instructions kernel)))))
    (when cl-cuda:*show-messages*
      (format t "~A~%" blueprint))
    (petalisp.utilities:with-hash-table-memoization 
      (hash)
      (if petalisp-cuda.options:*with-hash-table-memoization* (compile-cache backend) (make-hash-table))
      (let* ((iteration-scheme (generate-iteration-scheme kernel backend))
             (buffers (kernel-buffers kernel))
             (filtered-offset-vector (when (generic-offsets-p iteration-scheme) (filtered-offset-vector kernel)))
             (kernel-parameters (generate-kernel-parameters buffers iteration-scheme filtered-offset-vector))
             (kernel-symbol (format-symbol t "kernel-function")) ;cl-cuda wants symbol with a package for the function name
             (kernel-manager (make-kernel-manager))
             (generated-kernel `(,(remove-lispy-stuff (generate-kernel kernel
                                                                       kernel-parameters
                                                                       buffers
                                                                       iteration-scheme)
                                                      filtered-offset-vector))))  
        (when cl-cuda:*show-messages*
          (format t "Generated kernel:~%Arguments: ~A~%~A~%" kernel-parameters generated-kernel))
        (kernel-manager-define-function kernel-manager
                                        kernel-symbol
                                        'void
                                        kernel-parameters
                                        generated-kernel)
        ;; Load function here that only loadable function get into the compile cache
        (let ((hfunc (ensure-kernel-function-loaded kernel-manager kernel-symbol)))
          (make-jit-function :kernel-symbol kernel-symbol
                             :iteration-scheme iteration-scheme
                             :dynamic-shared-mem-bytes 0
                             :kernel-manager kernel-manager
                             :kernel-parameters kernel-parameters
                             :kernel-body generated-kernel
                             :hfunc hfunc))))))

(defun fill-with-device-ptrs (ptrs-to-device-ptrs device-ptrs kernel-arguments kernel-parameters iteration-scheme filtered-offsets)
  (loop for i from 0 below (length kernel-arguments) do
        (let ((argument (nth i kernel-arguments)))
          (if (arrayp argument)
              (let ((ffi-type (intern (symbol-name (kernel-parameter-type (nth i kernel-parameters))) "KEYWORD")))
                (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr i) ffi-type)
                      (cffi:convert-to-foreign
                        (cond ((and (weird-rational-p (aref argument)) (not petalisp-cuda.options:*strict-cast-mode*)) (coerce (aref argument) 'single-float))
                              ((symbolp (aref argument)) 0.0)
                              (t (aref argument)))
                        ffi-type)))
              (setf (cffi:mem-aref device-ptrs 'cu-device-ptr i) (device-ptr argument))))
        (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) i) (cffi:mem-aptr device-ptrs 'cu-device-ptr i)))
  (when (shape-independent-p iteration-scheme)
    (let ((offset (length kernel-arguments)))
      (loop for r in (shape-ranges (iteration-space iteration-scheme)) do
            (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr offset) :int)
                  (cffi:convert-to-foreign (range-start r) :int))
            (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) offset) (cffi:mem-aptr device-ptrs 'cu-device-ptr offset))
            (incf offset)
            (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr offset) :int)
                  (cffi:convert-to-foreign (range-end r) :int))
            (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) offset) (cffi:mem-aptr device-ptrs 'cu-device-ptr offset))
            (incf offset))
      (loop for p in kernel-arguments do
            (when (cuda-array-p p)
              (loop for s in (cuda-array-strides p) do
                    (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr offset) :int)
                          (cffi:convert-to-foreign s :int))
                    (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) offset) (cffi:mem-aptr device-ptrs 'cu-device-ptr offset))
                    (incf offset))))
    (when (generic-offsets-p iteration-scheme)
      (loop for o in filtered-offsets do
            (setf (cffi:mem-ref (cffi:mem-aptr device-ptrs 'cu-device-ptr offset) :int)
                  (cffi:convert-to-foreign o :int))
            (setf (cffi:mem-aref ptrs-to-device-ptrs '(:pointer :pointer) offset) (cffi:mem-aptr device-ptrs 'cu-device-ptr offset))
            (incf offset))))))

(defun run-compiled-function (compiled-function kernel-arguments iteration-space kernel)
  (let+ (((&slots kernel-symbol iteration-scheme dynamic-shared-mem-bytes kernel-parameters hfunc) compiled-function))
    (let ((parameters (call-parameters iteration-scheme iteration-space))
          (filtered-offsets (when (generic-offsets-p iteration-scheme)
                              (filtered-offset-vector kernel))))
      (let ((nargs (+ (length kernel-arguments)
                      (if (shape-independent-p iteration-scheme)
                              (+ (* 2 (shape-rank iteration-space)) (reduce #'+ (mapcar (lambda (b) (if (cuda-array-p b) (rank b) 0)) kernel-arguments)))
                              0)
                      (length filtered-offsets)))
            (extra-arguments (cffi:null-pointer))) ; has to be NULL since we use the kernel-args parameter
        (assert (= nargs (length kernel-parameters)))
        (cffi:with-foreign-objects ((ptrs-to-device-ptrs '(:pointer :pointer) nargs) (device-ptrs 'cu-device-ptr nargs))
          (fill-with-device-ptrs ptrs-to-device-ptrs device-ptrs kernel-arguments kernel-parameters iteration-scheme filtered-offsets)
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

(defun wait-for-previous-usages-deallocated (buffer backend)
  (map-buffer-outputs (lambda (kernel) (wait-for-correspoding-event kernel (cuda-backend-event-map backend))) buffer))

(defmethod petalisp-cuda.backend:execute-kernel (kernel (backend cuda-backend))
  (let ((buffers (kernel-buffers kernel)))
    (upload-buffers-to-gpu buffers backend)
    ;; All inputs must be uploaded to GPU and their inputs calculated
    (map-kernel-inputs (lambda (buffer) (wait-for-buffer buffer backend))
                       kernel)
    ;; All previous usages of the outputs must be freed
    (map-kernel-outputs (lambda (buffer)
                          (mapcar (lambda (predecessor)
                                    (wait-for-previous-usages-deallocated predecessor backend))
                                  (gethash (buffer-storage buffer) (cuda-backend-predecessor-map backend))))
                        kernel)
    (let ((arrays (mapcar #'buffer-storage buffers)))
      (if (custom-op-kernel-p kernel)
          (custom-op-kernel-execute kernel backend)
          (run-compiled-function (compile-kernel kernel backend) arrays (kernel-iteration-space kernel) kernel)))
    (record-corresponding-event kernel (cuda-backend-event-map backend))))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  (let* ((instructions (kernel-instructions kernel))
         (buffer->kernel-parameter (make-buffer->kernel-parameter buffers kernel-arguments)))
    ;; Loop over domain
    (iteration-code iteration-scheme
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'instruction-number)
                                           buffer->kernel-parameter
                                           iteration-scheme)
                    buffer->kernel-parameter)))


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

