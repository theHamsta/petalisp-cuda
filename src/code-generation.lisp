(defpackage petalisp-cuda.codegeneration
  (:use :petalisp-cuda.backend
        :cl)
  (:import-from :cl-cuda :void)
  (:import-from :cl-cuda.api.kernel-manager :*kernel-manager*
                                            :kernel-manager-define-function)
  (:import-from :petalisp.utilities :with-hash-table-memoization)
  (:import-from :alexandria :format-symbol :iota :with-gensyms)
  (:export :compile-kernel))

(in-package petalisp-cuda.codegeneration)

(defgeneric generate-iteration-scheme ())

(defun generate-iteration-scheme (kernel backend)
  )

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
                                          (generate-kernel-body kernel kernel-arguments buffers))
          function-name)))))
