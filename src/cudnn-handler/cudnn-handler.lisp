(defpackage petalisp-cuda.cudnn-handler
  (:use :petalisp-cuda.cudalibs
        :cl)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :petalisp.core :rank)
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
   (convolution-descriptors :accessor convolution-descriptors
                            :initform (make-hash-table :test #'equalp))
   (convolution-algorithms :accessor convolution-algorithms
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

(defun cudnn-create-convolution-descriptor (input-array paddings dilations filter-strides mode cudnn-handler)
  (let ((input-type (cudnn-type (petalisp-cuda.memory.cuda-array::cuda-array-type array)))
        (input-rank (length paddings)))
    (petalisp.utilities:with-hash-table-memoization
      ((values reduce-op element-type))
      (convolution-descriptors cudnn-handle)
      (cffi:with-foreign-object (descriptor '(cudnnConvolutionDescriptor-t :pointer))
        (cffi:with-foreign-objects (padA '(:int 3))
          (dilA '(:int 3))
          (strideA '(:int 3))
          (loop for i below input-rank
                for p in paddings
                for d in dilations
                for s in strideA
                (setf (mem-aref padA int i) p)
                (setf (mem-aref dilA int i) d)
                (setf (mem-aref strideA int i) s))
          (cudnnCreateConvolutionDescriptor descriptor)
          (cudnnSetConvolutionNdDescriptor descriptor
                                           input-rank
                                           padA
                                           dilA
                                           strideA
                                           mode
                                           input-type)
          (cffi:mem-aref descriptor cudnnConvolutionDescriptor-t))))))

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
                    (cffi:mem-ref new-descriptor :pointer))))))))

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


(defun cudnn-reduce-array (input-array
                            output-array
                            reduce-op
                            cudnn-handler
                            &key (input-factor 1.0) (accumulator-factor 0.0))
  "Reduces input-array A along non-singleton dimensions of output-array C using reduce-op

  The exact operation performed is 

    C = reduce op( input-factor * A ) + accumulator-factor * C
  "
  (let ((input-descriptor (cudnn-create-tensor-descriptor input-array cudnn-handler))
        (output-descriptor (cudnn-create-tensor-descriptor output-array cudnn-handler))
        (reduction-descriptor (cudnn-create-reduction-descriptor reduce-op (cuda-array-type input-array) cudnn-handler))
          (double-or-float (if (equalp (cuda-array-type input-array) :double) :double :float)))
      (cffi:with-foreign-object (workspace-min-size '(:pointer :int))
        (cffi:with-foreign-object (indices-min-size '(:pointer :int))
          (cffi:with-foreign-object (alpha '(:pointer :double))
              (cffi:with-foreign-object (beta '(:pointer :double))
                  ; this routine supports mixing data types, then alpha, beta are float or else everything is double
                  (setf (cffi:mem-ref alpha double-or-float) (cffi:convert-to-foreign input-factor double-or-float))
                  (setf (cffi:mem-ref beta double-or-float) (cffi:convert-to-foreign accumulator-factor double-or-float))
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
                                                         (cffi:make-pointer (device-ptr output-array))))))
                  output-array))))))
;The data types of the tensors A and C must match if of type double. In this case, alpha and beta and the computation enum of reduceTensorDesc are all assumed to be of type double.
;The HALF and INT8 data types may be mixed with the FLOAT data types. In these cases, the computation enum of reduceTensorDesc is required to be of type FLOAT. 


(defun get-convolution-forward-algorithm (input-descriptor filter-descriptor convolution-descriptor output-descriptor cudnn-handler)
  (petalisp.utilities:with-hash-table-memoization
    ((values input-descriptor filter-descriptor convolution-descriptor output-descriptor *cudnn-autotune*))
    (convolution-algorithms cudnn-handler)
    (cffi:with-foreign-object (algo-count '(:pointer :int))
      (assert (= 0 (cudnnGetConvolutionForwardAlgorithmMaxCount algo-count (cudnn-handle cudnn-handler))))
      (cffi:with-foreign-object (perf-results '(cudnnConvolutionFwdAlgoPerf_t (mem-aref algo-count :int)))
        (let (( ())))
        (if *cudnn-autotune*
            ;cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
            ;cudnnHandle_t                      handle,
            ;const cudnnTensorDescriptor_t      xDesc,
            ;const cudnnFilterDescriptor_t      wDesc,
            ;const cudnnConvolutionDescriptor_t convDesc,
            ;const cudnnTensorDescriptor_t      yDesc,
            ;const int                          requestedAlgoCount,
            ;int                               *returnedAlgoCount,
            ;cudnnConvolutionFwdAlgoPerf_t     *perfResults)
          (assert (= 0 (cudnnFindConvolutionForwardAlgorithm (cudnn-handle cudnn-handler)
                                                             input-descriptor
                                                             filter-descriptor
                                                             convolution-descriptor
                                                             output-descriptor
                                                             algo-count
                                                             returned-algo-count
                                                             perf-results)))
          ;cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
          ;cudnnHandle_t                       handle,
          ;const cudnnTensorDescriptor_t       xDesc,
          ;const cudnnFilterDescriptor_t       wDesc,
          ;const cudnnConvolutionDescriptor_t  convDesc,
          ;const cudnnTensorDescriptor_t       yDesc,
          ;const int                           requestedAlgoCount,
          ;int                                *returnedAlgoCount,
          ;cudnnConvolutionFwdAlgoPerf_t      *perfResults)
          (assert (= 0 (cudnnGetConvolutionForwardAlgorithm_v7 (cudnn-handle cudnn-handler)
                                                               input-descriptor
                                                               filter-descriptor
                                                               convolution-descriptor
                                                               output-descriptor
                                                               algo-count
                                                               returned-algo-count
                                                               perf-results)))
        (foreign-slot-value perf-results cudnnConvolutionFwdAlgoPerf_t 'algo))))))

(defun cudnn-convolution (input-array
                           filter-array
                           output-array
                           cudnn-handler
                           &key
                           algorithm
                           (input-factor 1.0)
                           (accumulator-factor 0.0)
                           (mode :cudnn-convolution)
                           bias
                           activation-function
                           (paddings (make-list (rank input-array) :initial-element 0))
                           (filter-strides (make-list (rank input-array) :initial-element 1))
                           (dilations (make-list (rank input-array) :initial-element 1)))
  (let ((input-descriptor (cudnn-create-tensor-descriptor input-array cudnn-handler))
        (output-descriptor (cudnn-create-tensor-descriptor output-array cudnn-handler))
        (convolution-descriptor (cudnn-create-convolution-descriptor input-array paddings dilations filter-strides mode cudnn-handler cudnn-handler))
        (filter-descriptor (cudnn-create-filter-descriptor (cuda-array-type input-array) cudnn-handler))
        (double-or-float (if (equalp (cuda-array-type input-array) :double) :double :float))
        (convolution-algorithm (get-convolution-algorithm input-descriptor filter-descriptor convolution-descriptor output-descriptor cudnn-handler)))

    (cffi:with-foreign-objects ((workspace-min-size '(:pointer :int))
                                (alpha '(:pointer :double))
                                (beta '(:pointer :double)))
          ; this routine supports mixing data types, then alpha, beta are float or else everything is double
          (setf (cffi:mem-ref alpha double-or-float) (cffi:convert-to-foreign input-factor double-or-float))
          (setf (cffi:mem-ref beta double-or-float) (cffi:convert-to-foreign accumulator-factor double-or-float))
          (assert (equalp :CUDNN-STATUS-SUCCESS
                          (cudnnGetConvolutionForwardWorkspaceSize (cudnn-handle cudnn-handler)
                                                                   input-descriptor
                                                                   filter-descriptor
                                                                   convolution-descriptor
                                                                   output-descriptor
                                                                   workspace-min-size)))
          (multiple-value-bind (workspace workspace-size) (allocate-workspace (cffi:mem-ref workspace-min-size :int) cudnn-handler)
            (assert (equalp :CUDNN-STATUS-SUCCESS
                            (cudnnConvolutionForward (cudnn-handle cudnn-handler)
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
        ;cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
        ;cudnnHandle_t   handle,
        ;const   cudnnTensorDescriptor_t         xDesc,
        ;const   cudnnFilterDescriptor_t         wDesc,
        ;const   cudnnConvolutionDescriptor_t    convDesc,
        ;const   cudnnTensorDescriptor_t         yDesc,
        ;cudnnConvolutionFwdAlgo_t               algo,
        ;size_t                                 *sizeInBytes)
        ))

"
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionFwdAlgo_t           algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       yDesc,
    void                               *y)
 "
