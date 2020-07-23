(defpackage petalisp-cuda.jitexecution
  (:use :petalisp-cuda.backend
        :petalisp.ir
        :petalisp
        :cl
        :cl-cuda
        :let-plus)
  (:import-from :cl-cuda.api.kernel-manager :*kernel-manager*
                                            :kernel-manager-define-function
                                            :ensure-kernel-function-loaded)
  (:import-from :cl-cuda.driver-api :cu-device-ptr :cu-launch-kernel)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:import-from :petalisp-cuda.iteration-scheme :call-parameters :iteration-code :make-block-iteration-scheme :get-counter-vector)
  (:import-from :petalisp-cuda.memory.cuda-array :cuda-array-strides :device-ptr)
  (:export :compile-kernel))

(in-package petalisp-cuda.jitexecution)


(defstruct (jit-function)
  kernel-symbol iteration-scheme shared-mem-bytes)

(defgeneric generate-iteration-scheme (kernel backend))

(defmethod generate-iteration-scheme (kernel backend)
  (make-block-iteration-scheme (kernel-iteration-space kernel)
                               (preferred-block-size backend)
                               (cuda-array-strides (storage (first (map-kernel-outputs #'identity kernel))))))

(defun generate-kernel-arguments (buffers)
  (mapcar (lambda (buffer idx) (list (format-symbol "buffer-~A" idx) (cl-cuda-type-from-buffer buffer)))
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
          (kernel-manager-define-function *kernel-manager*
                                          function-name
                                          'void
                                          kernel-arguments
                                          (generate-kernel kernel kernel-arguments buffers iteration-scheme))
          (make-jit-function :kernel-symbol function-name
                             :iteration-scheme iteration-scheme
                             :shared-mem-bytes 0))))))

(defun fill-with-device-ptrs (ptr-array kernel-arguments)
  (loop for i from 0 to (1- (list-length kernel-arguments)) do
        (setf (cffi:mem-aref ptr-array 'cu-device-ptr i) (device-ptr (nth i kernel-arguments)))))

(defun run-compiled-function (compiled-function kernel-arguments)
  (let+ (((&slots kernel-symbol iteration-scheme shared-mem-bytes) compiled-function))
    (let ((parameters (call-parameters iteration-scheme)))
      (let ((hfunc (ensure-kernel-function-loaded *kernel-manager* kernel-symbol)))
        (cffi:with-foreign-object (kargs 'cu-device-ptr (list-length kernel-arguments))
          (progn
            (fill-with-device-ptrs kargs kernel-arguments)
            (destructuring-bind (grid-dim-x grid-dim-y grid-dim-z) (getf parameters :grid-dim)
              (destructuring-bind (block-dim-x block-dim-y block-dim-z) (getf parameters :block-dim)
                (cu-launch-kernel hfunc
                                  grid-dim-x  grid-dim-y  grid-dim-z
                                  block-dim-x block-dim-y block-dim-z
                                  shared-mem-bytes cl-cuda.api.context:*cuda-stream*
                                  kargs (cffi:null-pointer))))))))))

(defmethod execute-kernel (kernel (backend cuda-backend))
  (let* ((buffers (kernel-buffers kernel))
         (arrays (mapcar #'buffer-storage buffers)))
    (run-compiled-function (compile-kernel kernel backend) arrays)))

(defun generate-kernel (kernel kernel-arguments buffers iteration-scheme)
  ;; Loop over domain
  (iteration-code iteration-scheme
                  (let* ((instructions (map-instructions #'identity kernel))
                         (buffer->kernel-argument (make-buffer->kernel-argument buffers kernel-arguments)))
                    ;; kernel body
                    (generate-instructions (sort instructions #'< :key #'instruction-number)
                                          buffer->kernel-argument))))

(defun linearize-instruction-transformation (instruction &optional buffer)
  (let+ (((&slots input-rank output-rank input-mask output-mask scalings offsets inverse)
          (instruction-transformation instruction)))
        (let ((input (mapcar (lambda (a b) (or a b)) input-mask (get-counter-vector input-rank)))
              (strides (if buffer (slot-value 'strides (storage buffer)) (iota output-rank)))
              (starts (if buffer (slot-value 'strides (storage buffer)) (iota output-rank))))
          `(+ ,(mapcar (lambda (a b) (or a b)) output-mask
                      (mapcar
                        (lambda (i o start s1 s2) `(* (+ ,i ,o ,start) ,s1 ,s2))
                        input offsets starts scalings strides))))))

(defun get-instruction-symbol (instruction)
  (format-symbol nil "$~A"
                 (if (numberp instruction)
                     instruction
                     (instruction-number instruction))))

(defun generate-instructions (instructions buffer->kernel-argument)
  (let* ((instruction (pop instructions))
        ($i (get-instruction-symbol instruction)))
    `(let ((,$i ,(etypecase instruction
                  (call-instruction
                    `(,(map-call-operator (call-instruction-operator instruction))
                       ,@(map-instruction-inputs #'get-instruction-symbol instruction)))
                  (iref-instruction
                    (linearize-instruction-transformation
                      (instruction-transformation instruction)))
                  (load-instruction
                    `(aref ,(funcall buffer->kernel-argument (load-instruction-buffer instruction))
                           ,(linearize-instruction-transformation instruction)))
                  (store-instruction
                    (let ((buffer (store-instruction-buffer instruction)))
                      `(set
                         (aref ,(funcall buffer->kernel-argument buffer)
                               (linearize-instruction-transformation instruction buffer))
                         ,(first
                            (map-instruction-inputs #'get-instruction-symbol instruction))))))))
       ,(generate-instructions instruction buffer->kernel-argument))))

(defun map-call-operator (operator)
  (progn 
    (break)
    operator))

(defun make-buffer->kernel-argument (buffers kernel-arguments)
    (lambda (buffer) (nth (position buffer buffers) kernel-arguments)))
