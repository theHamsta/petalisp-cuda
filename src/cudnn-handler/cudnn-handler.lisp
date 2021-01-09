(defpackage petalisp-cuda.cudnn-handler
  (:use :petalisp-cuda.cudalibs
        :cl)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-type
                :device-ptr)
  (:export :make-cudnn-handler
           :finalize-cudnn-handler
           :cudnn-reduce-array))
(in-package petalisp-cuda.cudnn-handler)

(defclass cudnn-handler ()
  ((cudnn-handle :accessor cudnn-handle
                 :initform (cudnn-init))
   (tensor-descriptors :accessor tensor-descriptors
                       :initform (make-hash-table :test #'equalp))
   (reduce-descriptors :accessor reduce-descriptors
                       :initform (make-hash-table :test #'equalp))
   (workspace :accessor workspace
              :initform nil)
   (workspace-size :accessor workspace-size
                   :initform 0)))

(defun make-cudnn-handler ()
  (make-instance 'cudnn-handler))

; TODO: garbage finalization probably not good for handling GPU resources
;(defun make-cudnn-handler ()
  ;(let ((object (make-instance 'cudnn-handler) ))
    ;(trivial-garbage:finalize object #'finalize-cudnn-handler)))

;; Helper
(defun fill-foreign-array (list foreign-array foreign-array-length fill-element)
  (progn
    (dotimes (i foreign-array-length)
      (setf (cffi:mem-aref foreign-array :int i) (cffi:convert-to-foreign fill-element :int)))
    (dotimes (i (length list))
                (setf (cffi:mem-aref foreign-array :int
                                        (+ i (max 0 (- foreign-array-length (length list)))))
                         (cffi:convert-to-foreign (nth i list) :int)))))

(defun clear-foreign-hashtable (hash-table destroy-function)
  (progn
    (mapcar (lambda (x) (progn
                                (funcall destroy-function x)
                                (cffi:foreign-free x)))
               (alexandria:hash-table-values hash-table))
    (clrhash hash-table)))

;; CUDNN abstractions
(defun cudnn-init ()
  (cffi:with-foreign-object (cudnn-handle-ptr '(:pointer :pointer))
    (progn
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnncreate cudnn-handle-ptr)))
      (cffi:mem-ref cudnn-handle-ptr :pointer))))

(defun finalize-cudnn-handler (cudnn-handler)
  (when cudnn-handler
    (clear-foreign-hashtable (tensor-descriptors cudnn-handler) #'cudnnDestroyTensorDescriptor)
    (clear-foreign-hashtable (reduce-descriptors cudnn-handler) #'cudnnDestroyReduceTensorDescriptor)
    (alexandria:when-let ((workspace-mem (workspace cudnn-handler)))
      (cl-cuda:free-device-memory workspace-mem))
    (setf (workspace cudnn-handler) nil)
    (setf (workspace-size cudnn-handler) 0)
    (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnDestroy (cudnn-handle cudnn-handler))))))

(defun cudnn-type (type)
  (trivia:match type
    (:int         :cudnn-data-int32)
    (:short-float :cudnn-data-half)
    (:float       :cudnn-data-float)
    (:double      :cudnn-data-double)
    (:bool        :cudnn-data-int8)
    (:int8        :cudnn-data-int8)
    (:uint8       :cudnn-data-uint8)
    (:float3      :cudnn-data-floatx3)
    (:float4      :cudnn-data-floatx4)
    (:double3     :cudnn-data-doublex3)
    (:double4     :cudnn-data-doublex4)
    (t (error "The value ~S is invalid here." type))))

(defun cudnn-reduce-op (reduce-op)
  (cond ((equalp reduce-op #'+)   :cudnn-reduce-tensor-add)
           ((equalp reduce-op #'*)   :cudnn-reduce-tensor-mul)
           ((equalp reduce-op #'max) :cudnn-reduce-tensor-max)
           ((equalp reduce-op #'min) :cudnn-reduce-tensor-min)
           (t (error "The value ~S is invalid here." reduce-op))))

(defun cudnn-create-tensor-descriptor (array cudnn-handler)
  (let* ((shape (slot-value array 'petalisp-cuda.memory.cuda-array::shape))
            (strides (slot-value array 'petalisp-cuda.memory.cuda-array::strides))
            (element-type (petalisp-cuda.memory.cuda-array::cuda-array-type array))
            (hash-key (values shape strides element-type))
            (min-shape (max (length shape) 4))) ; cudnn wants tensors of dim 4 to 8
           (or (gethash hash-key (tensor-descriptors cudnn-handler))
               (cffi:with-foreign-object (new-descriptor '(:pointer :pointer))
                 (cffi:with-foreign-object (stride-array :int min-shape)
                   (cffi:with-foreign-object (shape-array :int min-shape)
                     (progn
                       (assert (<= (length shape) 8)) ; cudnn requirement
                       (fill-foreign-array shape shape-array min-shape 1)
                       (fill-foreign-array strides stride-array min-shape 1)
                       (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateTensorDescriptor new-descriptor))) 
                       (assert (equalp :CUDNN-STATUS-SUCCESS
                                       (cudnnSetTensorNdDescriptor
                                         (cffi:mem-ref new-descriptor :pointer)
                                         (cudnn-type element-type)
                                         min-shape
                                         shape-array
                                         stride-array)))
                       (setf (gethash hash-key (tensor-descriptors cudnn-handler))
                             (cffi:mem-ref new-descriptor :pointer)))))))))

(defun cudnn-create-reduction-descriptor (reduce-op element-type cudnn-handle)
  (petalisp.utilities:with-hash-table-memoization
    ((values reduce-op element-type))
    (reduce-descriptors cudnn-handle)
    (cffi:with-foreign-object (new-descriptor '(:pointer :pointer))
      (progn
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnCreateReduceTensorDescriptor new-descriptor)))
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnSetReduceTensorDescriptor (cffi:mem-ref new-descriptor :pointer)
                                                        (cudnn-reduce-op reduce-op)
                                                        (cudnn-type element-type)
                                                        :cudnn-propagate-nan
                                                        :cudnn-reduce-tensor-no-indices ; ignored except for min/max
                                                        :cudnn-32bit-indices)))           ; only 32bit currently supported
        (cffi:mem-ref new-descriptor :pointer)))))

(defun allocate-workspace (min-size cudnn-handler)
  (progn
    (when (<= (workspace-size cudnn-handler) min-size)
      (progn
        (when (workspace cudnn-handler)
          (cl-cuda:free-device-memory (workspace cudnn-handler)))
        (setf (workspace-size cudnn-handler) min-size)
        (setf (workspace cudnn-handler)
              (cl-cuda:alloc-device-memory 'cl-cuda:float
                                           (/ (workspace-size cudnn-handler) 4)))))  ; allocate more than necessary ?
    (values (workspace cudnn-handler)
            (workspace-size cudnn-handler))))


(defun cudnn-reduce-array (input-array output-array reduce-op cudnn-handler)
  (let ((input-descriptor (cudnn-create-tensor-descriptor input-array cudnn-handler))
          (output-descriptor (cudnn-create-tensor-descriptor output-array cudnn-handler))
          (reduction-descriptor (cudnn-create-reduction-descriptor reduce-op (cuda-array-type input-array) cudnn-handler))
          (double-or-float (if (equalp (cuda-array-type input-array) :double) :double :float)))
      (cffi:with-foreign-object (workspace-min-size '(:pointer :int))
        (cffi:with-foreign-object (indices-min-size '(:pointer :int))
          (cffi:with-foreign-object (alpha '(:pointer :double))
              (cffi:with-foreign-object (beta '(:pointer :double))
                (progn
                  ; this routine supports mixing data types, then alpha, beta are float or else everything is double
                  (setf (cffi:mem-ref alpha double-or-float) (cffi:convert-to-foreign 1.0 double-or-float))
                  (setf (cffi:mem-ref beta double-or-float) (cffi:convert-to-foreign 0.0 double-or-float))
                  (assert (equalp :CUDNN-STATUS-SUCCESS
                                  (cudnnGetReductionWorkspaceSize (cudnn-handle cudnn-handler)
                                                                  reduction-descriptor
                                                                  input-descriptor
                                                                  output-descriptor
                                                                  workspace-min-size)))
                  (assert (equalp :CUDNN-STATUS-SUCCESS
                                  (cudnnGetReductionIndicesSize (cudnn-handle cudnn-handler)
                                                                reduction-descriptor
                                                                input-descriptor
                                                                output-descriptor
                                                                indices-min-size)))
                  (assert (equalp 0 (cffi:mem-ref indices-min-size :int))) ; not 0 if argmin indices requested
                  (multiple-value-bind (workspace workspace-size) (allocate-workspace (cffi:mem-ref workspace-min-size :int) cudnn-handler)
                    (progn
                      (assert (equalp :CUDNN-STATUS-SUCCESS
                                      (cudnnReduceTensor (cudnn-handle cudnn-handler)
                                                         reduction-descriptor
                                                         ; if indices requrested, find this out with cudnnGetReductionIndicesSize 
                                                         (cffi:make-pointer 0)
                                                         (cffi:mem-ref indices-min-size :int)
                                                         (cffi:make-pointer workspace)
                                                         workspace-size
                                                         alpha ; &alpha == (type) 1 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                                         input-descriptor
                                                         (cffi:make-pointer (device-ptr input-array))
                                                         beta ; &beta = (type) 0 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                                         output-descriptor
                                                         (cffi:make-pointer (device-ptr output-array)))))))
                  output-array)))))))
;The data types of the tensors A and C must match if of type double. In this case, alpha and beta and the computation enum of reduceTensorDesc are all assumed to be of type double.
;The HALF and INT8 data types may be mixed with the FLOAT data types. In these cases, the computation enum of reduceTensorDesc is required to be of type FLOAT. 
