(in-package petalisp-cuda.cudalibs)

; TODO wrap this globals in nice structure
(cl:defparameter *cudnn-handle* cl:nil)
(cl:defparameter *cudnn-tensor-descriptors* (cl:make-hash-table))
(cl:defparameter *cudnn-reduction-descriptors* (cl:make-hash-table))
(cl:defparameter *cudnn-workspace* nil)
(cl:defparameter *cudnn-workspace-size* 0)

;; Helper
(cl:defun fill-foreign-array (list foreign-array foreign-array-length fill-element)
  (cl:progn
    (cl:dotimes (i foreign-array-length)
      (cl:setf (cffi:mem-aref foreign-array :int i) (cffi:convert-to-foreign fill-element :int)))
    (cl:dotimes (i (cl:length list))
                (cl:setf (cffi:mem-aref foreign-array :int
                                        (cl:+ i (cl:max 0 (cl:- foreign-array-length (cl:length list)))))
                         (cffi:convert-to-foreign (cl:nth i list) :int)))))

(cl:defun clear-foreign-hashtable (hash-table destroy-function)
  (cl:progn
    (cl:mapcar (cl:lambda (x) (progn
                                (cl:funcall destroy-function x)
                                (cffi:foreign-free x)))
               (alexandria:hash-table-values hash-table))
    (cl:setf hash-table (cl:make-hash-table))))

;; CUDNN abstractions
(cl:defun cudnn-init ()
  (cl:unless *cudnn-handle*
    (cl:progn
      (cl:setq *cudnn-handle* (cffi:foreign-alloc '(:pointer :pointer)))
      (cl:assert (cl:equalp :CUDNN-STATUS-SUCCESS (cudnncreate *cudnn-handle*))))))

(cl:defun cudnn-destroy ()
  (cl:when *cudnn-handle*
    (cl:progn 
      (clear-foreign-hashtable *cudnn-tensor-descriptors* #'cudnnDestroyTensorDescriptor)
      (clear-foreign-hashtable *cudnn-reduction-descriptors* #'cudnnDestroyReduceTensorDescriptor)
      (cl:assert (cl:equalp :CUDNN-STATUS-SUCCESS (cudnndestroy *cudnn-handle*))))))

(cl:defun cudnn-type (type)
  (trivia:match type
    (:int  :cudnn-data-int32)
    (:float       :cudnn-data-float)
    (:double                :cudnn-data-double)
    ;(cl-cuda:int8      :cudnn-data-int8)
    (:float3     :cudnn-data-floatx3)
    (:float4     :cudnn-data-floatx4)
    (:double3    :cudnn-data-doublex3)
    (:double4    :cudnn-data-doublex4)
    (cl:t (cl:error "The value ~S is invalid here." type))))

;; what's the problem with this??
;(defun cudnn-reduce-op (reduce-op)
  ;(trivia:match reduce-op
    ;(#':+                    :cudnn-reduce-tensor-add)
    ;(#'cl:*                  :cudnn-reduce-tensor-mul)
    ;(#'cl:max                :cudnn-reduce-tensor-max)
    ;(#'cl:min                :cudnn-reduce-tensor-min)
    ;(cl:t (cl:error "The value ~S is invalid here." reduce-op))))

(defun cudnn-reduce-op (reduce-op)
  (cl:if (equalp reduce-op #'cl:+)
      :cudnn-reduce-tensor-add
      (cl:if (equalp reduce-op #'cl:*)
          :cudnn-reduce-tensor-mul
          (cl:if (equalp reduce-op #'cl:min)
              :cudnn-reduce-tensor-min
              (cl:if (equalp reduce-op #'cl:max)
                  :cudnn-reduce-tensor-max)))))

(cl:defun cudnn-create-tensor-descriptor (array)
  (cl:let* ((shape (cl:slot-value array 'petalisp-cuda.cuda-array::shape))
            (strides (cl:slot-value array 'petalisp-cuda.cuda-array::strides))
            (element-type (petalisp-cuda.cuda-array::element-type array))
            (hash-key '(device-ptr shape stride element-type))
            (min-shape (cl:max (cl:length shape) 4))) ; cudnn wants tensors of dim 4 to 8
    (or nil;(cl:gethash hash-key *cudnn-tensor-descriptors*)
        (let ((new-descriptor (cffi:foreign-alloc '(:pointer :pointer))))
          (cffi:with-foreign-object (stride-array :int min-shape)
            (cffi:with-foreign-object (shape-array :int min-shape)
              (cl:progn
                (cl:assert (cl:<= (cl:length shape) 8)) ; cudnn requirement
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
                (setf (gethash hash-key *cudnn-tensor-descriptors*) new-descriptor))))))))

(defun cudnn-create-reduction-descriptor (reduce-op element-type)
  (let ((hash-key '(reduce-op element-type)))
    (or nil; (gethash hash-key *cudnn-reduction-descriptors*)
      (let ((new-descriptor (cffi:foreign-alloc '(:pointer :pointer))))
        (progn
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateReduceTensorDescriptor new-descriptor)))
          (assert (equalp :CUDNN-STATUS-SUCCESS
                          (cudnnSetReduceTensorDescriptor (cffi:mem-ref new-descriptor :pointer)
                                                          (cudnn-reduce-op reduce-op)
                                                          (cudnn-type element-type)
                                                          :cudnn-propagate-nan
                                                          :cudnn-reduce-tensor-no-indices ; ignored except for min/max
                                                          :cudnn-32bit-indices)))           ; only 32bit currently supported
          (setf (gethash hash-key *cudnn-reduction-descriptors*) new-descriptor))))))

(defun allocate-workspace (min-size)
  (progn
    (when (cl:<= *cudnn-workspace-size* min-size)
      (cl:progn
        (when *cudnn-workspace*
          (cl-cuda:free-device-memory *cudnn-workspace*))
        (setq *cudnn-workspace-size* min-size)
        (setq *cudnn-workspace* (cl-cuda:alloc-device-memory 'cl-cuda:float (cl:/ *cudnn-workspace-size* 4)))))  ; allocate more than necessary ?
    (values *cudnn-workspace* *cudnn-workspace-size*)))


(defun cudnn-reduce-array (input-array output-array reduce-op)
  (progn
    (cl:assert *cudnn-handle*)
    (let ((input-descriptor (cudnn-create-tensor-descriptor input-array))
          (output-descriptor (cudnn-create-tensor-descriptor output-array))
          (reduction-descriptor (cudnn-create-reduction-descriptor reduce-op (element-type input-array)))
          (double-or-float (if (equalp (element-type input-array) cl-cuda:double) cl-cuda:double :float))) ; this routine supports mixing data types, then alpha, beta are float or else everything is double
      (cffi:with-foreign-object (workspace-min-size '(:pointer :int))
        (cffi:with-foreign-object (indices-min-size '(:pointer :int))
          (cffi:with-foreign-object (indices '(:pointer :int))
            (cffi:with-foreign-object (alpha '(:pointer :double))
              (cffi:with-foreign-object (beta '(:pointer :double))
                (progn
                  (setf (cffi:mem-ref alpha double-or-float) (cffi:convert-to-foreign 1.0 double-or-float))
                  (setf (cffi:mem-ref beta double-or-float) (cffi:convert-to-foreign 0.0 double-or-float)) ;does not work for integers
                  (assert (equalp :CUDNN-STATUS-SUCCESS
                                  (cudnnGetReductionWorkspaceSize (cffi:mem-ref *cudnn-handle* :pointer)
                                                                  (cffi:mem-ref reduction-descriptor :pointer)
                                                                  (cffi:mem-ref input-descriptor :pointer)
                                                                  (cffi:mem-ref output-descriptor :pointer)
                                                                  workspace-min-size)))
                  (assert (equalp :CUDNN-STATUS-SUCCESS
                                  (cudnnGetReductionIndicesSize (cffi:mem-ref *cudnn-handle* :pointer)
                                                                (cffi:mem-ref reduction-descriptor :pointer)
                                                                (cffi:mem-ref input-descriptor :pointer)
                                                                (cffi:mem-ref output-descriptor :pointer)
                                                                indices-min-size)))
                  (assert (equalp 0 (cffi:mem-ref indices-min-size :int))) ; not 0 if argmin indices requested
                  (cl:multiple-value-bind (workspace workspace-size) (allocate-workspace (cffi:mem-ref workspace-min-size :int))
                    (progn
                      (cl:print workspace-size)
                      (cl:print (cffi:mem-ref workspace-min-size :int))
                      (assert (equalp :CUDNN-STATUS-SUCCESS
                                      (cudnnReduceTensor (cffi:mem-ref *cudnn-handle* :pointer)
                                                         (cffi:mem-ref reduction-descriptor :pointer)
                                                         ; if indices requrested, find this out with cudnnGetReductionIndicesSize 
                                                         (cffi:make-pointer 0)
                                                         (cffi:mem-ref indices-min-size :int)
                                                         (cffi:make-pointer workspace)
                                                         workspace-size
                                                         alpha ; &alpha == (type) 1 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                                         (cffi:mem-ref input-descriptor :pointer)
                                                         (cffi:make-pointer (device-ptr input-array))
                                                         beta ; &beta = (type) 0 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                                         (cffi:mem-ref output-descriptor :pointer)
                                                         (cffi:make-pointer (device-ptr output-array))))))))))))))))
;The data types of the tensors A and C must match if of type double. In this case, alpha and beta and the computation enum of reduceTensorDesc are all assumed to be of type double.
;The HALF and INT8 data types may be mixed with the FLOAT data types. In these cases, the computation enum of reduceTensorDesc is required to be of type FLOAT. 
