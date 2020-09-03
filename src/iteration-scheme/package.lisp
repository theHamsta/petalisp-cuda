(defpackage petalisp-cuda.iteration-scheme
  (:use :cl
        :petalisp)
  (:import-from :alexandria :iota :format-symbol)
  (:import-from :cl-cuda :block-dim-x :block-dim-y :block-dim-z
                :block-idx-x :block-idx-y :block-idx-z
                :thread-idx-x :thread-idx-y :thread-idx-z)
  (:export :select-iteration-scheme
           :call-parameters
           :iteration-code
           :get-counter-symbol))

(in-package petalisp-cuda.iteration-scheme)

(defclass iteration-scheme ()
  ((%shape :initarg :shape
           :accessor iteration-shape
           :type petalisp.core:shape)
   (%xyz-dimensions :initarg :xyz-dimensions
                    :accessor xyz-dimensions
                    :type list)))

(defgeneric call-parameters (iteration-scheme))
(defgeneric iteration-code (iteration-scheme kernel-body))
