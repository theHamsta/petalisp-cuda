(defpackage :petalisp-cuda.cuda-immediate
  (:use :cl
        :petalisp.core)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array
                :cuda-array-p
                :cuda-array-type
                :type-from-cl-cuda-type)
  (:import-from :petalisp.ir
                :grow-dendrite
                :dendrite
                :dendrite-transformation
                :stem-kernel
                :dendrite-shape
                :dendrite-cons
                :dendrite-stem
                :ir-converter-scalar-table
                :make-load-instruction
                :buffer-readers
                :kernel-sources
                :ir-converter-array-table
                :*ir-converter*
                :make-buffer)
  (:import-from :petalisp-cuda.type-conversion
                :ntype-cuda-array)
  (:export :cuda-immediate
           :make-cuda-immediate
           :cuda-immediate-p
           :cuda-immediate-storage))
(in-package :petalisp-cuda.cuda-immediate)

(defclass cuda-immediate (non-empty-immediate)
  ((%storage :initarg :storage :accessor cuda-immediate-storage)))

;;; Construction
(defmethod lazy-array ((cuda-array cuda-array))
  (make-cuda-immediate cuda-array))

(defun make-cuda-immediate (cu-array)
    (check-type cu-array cuda-array)
    (make-instance 'cuda-immediate
                   :shape (shape cu-array)
                   :storage cu-array
                   :ntype (ntype-cuda-array cu-array)))

;;; Predicates
(defgeneric cuda-immediate-p (immediate))

(defmethod cuda-immediate-p ((immediate t)))

(defmethod cuda-immediate-p ((immediate cuda-immediate))
  t)

;;; Conversion and Interop with Array
(defgeneric cuda-immediate-storage (immediate))

(defmethod cuda-immediate-storage ((immediate array-immediate))
  (array-immediate-storage immediate))

(defmethod array-from-immediate ((cuda-array cuda-array))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp cuda-array)) 

(defmethod array-from-immediate ((cuda-immediate cuda-immediate))
  (petalisp-cuda.memory.cuda-array:copy-cuda-array-to-lisp (cuda-immediate-storage cuda-immediate))) 

(defmethod replace-lazy-array ((instance lazy-array) (replacement cuda-immediate))
  (change-class instance (class-of replacement)
    :storage (cuda-immediate-storage replacement)
    :ntype (element-ntype replacement)
    :shape (array-shape replacement)))

(defmethod grow-dendrite
    ((dendrite dendrite)
     (cuda-immediate cuda-immediate))
  (with-accessors ((shape dendrite-shape)
                   (transformation dendrite-transformation)
                   (stem dendrite-stem)
                   (cons dendrite-cons)) dendrite
    (let* ((kernel (stem-kernel stem))
           (shape (lazy-array-shape cuda-immediate))
           (storage (cuda-immediate-storage cuda-immediate))
           (ntype (petalisp.type-inference:generalize-ntype
                   (element-ntype cuda-immediate)))
           (buffer
             (if (zerop (shape-rank shape))
                 (alexandria:ensure-gethash
                  (aref (cuda-immediate-storage cuda-immediate))
                  (ir-converter-scalar-table *ir-converter*)
                  (make-buffer
                   :shape shape
                   :ntype ntype
                   :storage storage))
                 (alexandria:ensure-gethash
                  (cuda-immediate-storage cuda-immediate)
                  (ir-converter-array-table *ir-converter*)
                  (make-buffer
                   :shape shape
                   :ntype ntype
                   :storage storage))))
           (load-instruction (make-load-instruction kernel buffer transformation)))
      (push load-instruction (alexandria:assoc-value (kernel-sources kernel) buffer))
      (push load-instruction (alexandria:assoc-value (buffer-readers buffer) kernel))
      (setf (cdr cons) load-instruction))))

(defmethod total-size ((cuda-array cuda-array))
  (total-size (lazy-array cuda-array)))
