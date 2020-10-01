(defpackage petalisp-cuda.iteration-scheme
  (:use :cl
        :petalisp.core
        :petalisp.ir
        :cl-cuda)
  (:import-from :alexandria :iota :format-symbol)
  (:import-from :petalisp-cuda.memory.type-conversion
                :cl-cuda-type-from-buffer)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-strides)
  (:import-from :cl-cuda :block-dim-x :block-dim-y :block-dim-z
                :block-idx-x :block-idx-y :block-idx-z
                :thread-idx-x :thread-idx-y :thread-idx-z)
  (:export :select-iteration-scheme
           :call-parameters
           :iteration-code
           :get-counter-symbol
           :linearize-instruction-transformation
           :iteration-scheme-prepare-instruction ))

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
(defgeneric iteration-scheme-buffer-access (iteration-scheme instruction buffer kernel-parameter))
(defgeneric iteration-scheme-prepare-instruction (iteration-scheme instruction buffer->kernel-parameter))
(defgeneric iterminate-of-bounds-threads-p (iteration-scheme))

(defun linearize-instruction-transformation (instruction &optional buffer)
  (let* ((transformation (instruction-transformation instruction))
         (input-rank (transformation-input-rank transformation))
         (strides (if buffer (cuda-array-strides (buffer-storage buffer)) (make-list input-rank :initial-element 1)))
         (index-space (get-counter-vector input-rank) )
         (transformed (transform index-space transformation)))
    (let ((rtn `(+ ,@(mapcar (lambda (a b) `(* ,a ,b)) transformed strides))))
      (if (= (length rtn) 1)
          0 ; '+ with zero arguments
          rtn))))

(defmethod iteration-scheme-buffer-access ((iteration-scheme iteration-scheme) instruction buffer kernel-parameter)
  ;; We can always do a uncached memory access
  `(aref ,kernel-parameter ,(linearize-instruction-transformation instruction buffer)))

(defmethod terminate-of-bounds-threads-p ((iteration-scheme iteration-scheme))
  t)

(defmethod iteration-scheme-prepare-instruction ((iteration-scheme iteration-scheme) instruction buffer->kernel-parameter)
  )
