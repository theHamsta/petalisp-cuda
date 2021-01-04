(defpackage petalisp-cuda.iteration-scheme
  (:use :cl
        :petalisp.core
        :petalisp.ir
        :petalisp-cuda.options)
  (:import-from :alexandria :iota :format-symbol)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-strides)
  (:import-from :cl-cuda :block-dim-x :block-dim-y :block-dim-z
                :block-idx-x :block-idx-y :block-idx-z
                :thread-idx-x :thread-idx-y :thread-idx-z)
  (:import-from :petalisp-cuda.type-conversion
                :cl-cuda-type-from-ntype)
  (:import-from :alexandria
                :format-symbol)
  (:export :select-iteration-scheme
           :call-parameters
           :iteration-code
           :get-counter-symbol
           :linearize-instruction-transformation
           :shape-independent-p
           :generic-offsets-p
           :iteration-space
           :get-instruction-symbol
           :kernel-parameter-name
           :kernel-parameter-type))

(in-package petalisp-cuda.iteration-scheme)

(defclass iteration-scheme ()
  ((%shape :initarg :shape
           :accessor iteration-space
           :type petalisp.core:shape)
   (%xyz-dimensions :initarg :xyz-dimensions
                    :accessor xyz-dimensions
                    :type list)))

(defgeneric call-parameters (iteration-scheme iteration-shape))
(defgeneric iteration-code (iteration-scheme kernel-body buffer->kernel-parameter))
(defgeneric iteration-scheme-buffer-access (iteration-scheme instruction buffer kernel-parameter))
(defgeneric shape-independent-p (iteration-scheme))
(defgeneric generic-offsets-p (iteration-scheme))
(defgeneric iteration-scheme-shared-mem (iteration-scheme))

(defmethod iteration-scheme-shared-mem ((iteration-scheme iteration-scheme))
  (values nil nil))

(defmethod iteration-code :before ((iteration-scheme iteration-scheme) kernel-body buffer->kernel-parameter)
  (multiple-value-bind (type shape) (iteration-scheme-shared-mem iteration-scheme)
    (if typpe
        `(with-shared-memory ((shared-mem ,type ,@shape))
           ,(call-next-method)))
    (call-next-method)))

