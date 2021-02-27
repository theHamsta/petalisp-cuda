(defpackage petalisp-cuda.cudnn-handler
  (:use :petalisp-cuda.cudalibs
        :cl
        :cffi
        :make-hash)
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :petalisp.core :rank)
  (:import-from :petalisp-cuda.options :*cudnn-autotune*)
  (:import-from :petalisp-cuda.memory.cuda-array
                :cuda-array-type
                :cuda-array-shape
                :cuda-array-strides
                :device-ptr
                :c-layout-p)
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
   (filter-descriptors :accessor filter-descriptors
                            :initform (make-hash-table :test #'equalp))
   (convolution-descriptors :accessor convolution-descriptors
                            :initform (make-hash-table :test #'equalp))
   (convolution-algorithms :accessor convolution-algorithms
                           :initform (make-hash-table :test #'equalp))
   (activation-desciptors :accessor activation-desciptors
                          :initform (make-hash-table :test #'equalp))
   (workspace :accessor workspace
              :initform 0)
   (workspace-size :accessor workspace-size
                   :initform 0)))

(defun make-cudnn-handler ()
  (make-instance 'cudnn-handler))

;; Helper
(defun fill-foreign-array (list foreign-array foreign-array-length fill-element)
  (progn
    (dotimes (i foreign-array-length)
      (setf (mem-aref foreign-array :int i) (convert-to-foreign fill-element :int)))
    (dotimes (i (length list))
                (setf (mem-aref foreign-array :int
                                        (+ i (max 0 (- foreign-array-length (length list)))))
                         (convert-to-foreign (nth i list) :int)))))

(defun clear-foreign-hashtable (hash-table destroy-function)
  (mapcar (lambda (x) 
            (funcall destroy-function x)
            (foreign-free x))
          (alexandria:hash-table-values hash-table))
  (clrhash hash-table))

;; CUDNN abstractions
(defun cudnn-init ()
  (with-foreign-object (cudnn-handle-ptr '(:pointer :pointer))
    (assert (equalp :CUDNN-STATUS-SUCCESS (cudnncreate cudnn-handle-ptr)))
    (mem-ref cudnn-handle-ptr :pointer)))

(defun finalize-cudnn-handler (cudnn-handler)
  (when cudnn-handler
    (clear-foreign-hashtable (tensor-descriptors cudnn-handler) #'cudnnDestroyTensorDescriptor)
    (clear-foreign-hashtable (reduce-descriptors cudnn-handler) #'cudnnDestroyReduceTensorDescriptor)
    (clear-foreign-hashtable (convolution-descriptors cudnn-handler) #'cudnnDestroyConvolutionDescriptor)
    (clear-foreign-hashtable (filter-descriptors cudnn-handler) #'cudnnDestroyFilterDescriptor)
    (clear-foreign-hashtable (activation-desciptors cudnn-handler) #'cudnnDestroyActivationDescriptor)
    (alexandria:when-let ((workspace-mem (workspace cudnn-handler)))
      (cl-cuda:free-device-memory workspace-mem))
    (setf (workspace cudnn-handler) 0)
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
  (alexandria:switch (reduce-op :test #'equalp)
    (#'+            :cudnn-reduce-tensor-add)
    (#'*            :cudnn-reduce-tensor-mul)
    (#'max          :cudnn-reduce-tensor-max)
    (#'min          :cudnn-reduce-tensor-min)
    (:argmin        :cudnn-reduce-tensor-amin)
    (:avg           :cudnn-reduce-tensor-avg)
    (:norm1         :cudnn-reduce-tensor-norm1)
    (:norm2         :cudnn-reduce-tensor-norm2)
    (:mul-no-zeros  :cudnn-reduce-tensor-mul-no-zeros)
    (t (error "The value ~S is invalid here." reduce-op))))

(defun cudnn-create-convolution-descriptor (input-array paddings dilations filter-strides mode cudnn-handler)
  (let ((input-type (cudnn-type (petalisp-cuda.memory.cuda-array::cuda-array-type input-array)))
        (array-length (length paddings)))
    (assert (= (+ 2 array-length) (rank input-array)))
    (assert (= array-length (length dilations)))
    (assert (= array-length (length filter-strides)))
    (petalisp.utilities:with-hash-table-memoization
      ((list input-type paddings dilations filter-strides mode))
      (convolution-descriptors cudnn-handler)
      (with-foreign-objects ((descriptor '(:pointer cudnnConvolutionDescriptor-t))
                             (padA :int array-length)
                             (dilA :int array-length)
                             (strideA :int array-length))
        (loop for i from 0 below array-length
              for p in paddings
              for d in dilations
              for s in filter-strides
              do
              (progn
                (setf (mem-aref padA :int i) p)
                (setf (mem-aref dilA :int i) d)
                (setf (mem-aref strideA :int i) s)))
        (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateConvolutionDescriptor descriptor)))
        (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnSetConvolutionNdDescriptor (mem-aref descriptor 'cudnnConvolutionDescriptor-t)
                                                                               array-length
                                                                               padA
                                                                               strideA
                                                                               dilA
                                                                               mode
                                                                               input-type)))
        (mem-aref descriptor 'cudnnConvolutionDescriptor-t)))))

  ;(:cudnn-tensor-nchw 0)
  ;(:cudnn-tensor-nhwc 1)
  ;(:cudnn-tensor-nchw-vect-c 2)

;format

    ;Input.Type of the filter layout format. If this input is set to CUDNN_TENSOR_NCHW, which is one of the enumerant values allowed by cudnnTensorFormat_t descriptor, then the layout of the filter is as follows:

        ;For N=4, a 4D filter descriptor, the filter layout is in the form of KCRS:
            ;K represents the number of output feature maps
            ;C is the number of input feature maps
            ;R is the number of rows per filter
            ;S is the number of columns per filter
        ;For N=3, a 3D filter descriptor, the number S (number of columns per filter) is omitted.
        ;For N=5 and greater, the layout of the higher dimensions immediately follows RS.

    ;On the other hand, if this input is set to CUDNN_TENSOR_NHWC, then the layout of the filter is as follows:

;For N=4, a 4D filter descriptor, the filter layout is in the form of KRSC.
;For N=3, a 3D filter descriptor, the number S (number of columns per filter) is omitted and the layout of C immediately follows R.
;For N=5 and greater, the layout of the higher dimensions are inserted between S and C. For more information, see cudnnTensorFormat_t.
(defun cudnn-create-filter-descriptor (filter-array filter-format cudnn-handler &optional (min-dimensions 4))
  (assert (c-layout-p filter-array))
  (let ((input-type (cudnn-type (petalisp-cuda.memory.cuda-array::cuda-array-type filter-array)))
        (input-rank (max (rank filter-array) min-dimensions)))
    (petalisp.utilities:with-hash-table-memoization
      ((list input-rank input-type filter-format (cuda-array-shape filter-array)))
      (filter-descriptors cudnn-handler)
      (with-foreign-objects ((descriptor '(:pointer cudnnFilterDescriptor-t))
                             (filterDimA :int input-rank))
        (fill-foreign-array (cuda-array-shape filter-array) filterDimA input-rank 1)
        (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateFilterDescriptor descriptor)))
        (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnSetFilterNdDescriptor (mem-aref descriptor 'cudnnFilterDescriptor-t)
                                                                          input-type
                                                                          filter-format
                                                                          input-rank
                                                                          filterDimA)))
        (mem-aref descriptor 'cudnnFilterDescriptor-t)))))

(defun cudnn-create-tensor-descriptor (cuda-array cudnn-handler &optional (min-shape-dim 4))
  (let* ((shape (cuda-array-shape cuda-array))
         (strides (cuda-array-strides cuda-array))
         (element-type (cuda-array-type cuda-array))
         (min-shape-dim (max (length shape) min-shape-dim))) ; cudnn wants tensors of dim 4 to 8
    (petalisp.utilities:with-hash-table-memoization
      ((list shape strides element-type))
      (tensor-descriptors cudnn-handler)
      (with-foreign-objects ((new-descriptor '(:pointer :pointer))
                             (stride-array :int min-shape-dim)
                             (shape-array :int min-shape-dim))
        (assert (<= (length shape) 8)) ; cudnn requirement
        (fill-foreign-array shape shape-array min-shape-dim 1)
        (fill-foreign-array strides stride-array min-shape-dim 1)
        (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateTensorDescriptor new-descriptor))) 
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnSetTensorNdDescriptor
                          (mem-ref new-descriptor :pointer)
                          (cudnn-type element-type)
                          min-shape-dim
                          shape-array
                          stride-array)))
        (mem-ref new-descriptor :pointer)))))

(defun cudnn-create-reduction-descriptor (reduce-op element-type cudnn-handle)
  (petalisp.utilities:with-hash-table-memoization
    ((list reduce-op element-type))
    (reduce-descriptors cudnn-handle)
    (with-foreign-object (new-descriptor '(:pointer :pointer))
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnCreateReduceTensorDescriptor new-descriptor)))
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnSetReduceTensorDescriptor (mem-ref new-descriptor :pointer)
                                                        (cudnn-reduce-op reduce-op)
                                                        (cudnn-type element-type)
                                                        :cudnn-propagate-nan
                                                        :cudnn-reduce-tensor-no-indices ; ignored except for min/max
                                                        :cudnn-32bit-indices)))           ; only 32bit currently supported
        (mem-ref new-descriptor :pointer))))

(defun allocate-workspace (min-size cudnn-handler)
    (when (< (workspace-size cudnn-handler) min-size)
        (when (workspace cudnn-handler)
          ;; TODO: replace with malloc-async
          (cuCtxSynchronize)
          (cl-cuda:free-device-memory (workspace cudnn-handler)))
        (setf (workspace-size cudnn-handler) min-size)
        (setf (workspace cudnn-handler)
              (cl-cuda:alloc-device-memory 'cl-cuda:float
                                           (/ (workspace-size cudnn-handler) 4))))  ; allocate more than necessary ?
    (values (workspace cudnn-handler)
            (workspace-size cudnn-handler)))


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
    (with-foreign-objects ((workspace-min-size '(:pointer :int))
                           (indices-min-size '(:pointer :int))
                           (alpha '(:pointer :double))
                           (beta '(:pointer :double)))
      ; this routine supports mixing data types, then alpha, beta are float or else everything is double
      (setf (mem-ref alpha double-or-float) (convert-to-foreign input-factor double-or-float))
      (setf (mem-ref beta double-or-float) (convert-to-foreign accumulator-factor double-or-float))
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
      (assert (equalp 0 (mem-ref indices-min-size :int))) ; not 0 if argmin indices requested
      (multiple-value-bind (workspace workspace-size) (allocate-workspace (mem-ref workspace-min-size :int) cudnn-handler)
        (assert (equalp :CUDNN-STATUS-SUCCESS
                        (cudnnReduceTensor (cudnn-handle cudnn-handler)
                                           reduction-descriptor
                                           ; if indices requrested, find this out with cudnnGetReductionIndicesSize 
                                           (make-pointer 0)
                                           (mem-ref indices-min-size :int)
                                           (make-pointer workspace)
                                           workspace-size
                                           alpha ; &alpha == (type) 1 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                           input-descriptor
                                           (make-pointer (device-ptr input-array))
                                           beta ; &beta = (type) 0 <- /* Tensor operation : C = reduce op( alpha * A ) + beta * C */
                                           output-descriptor
                                           (make-pointer (device-ptr output-array))))))
      output-array)))
;The data types of the tensors A and C must match if of type double. In this case, alpha and beta and the computation enum of reduceTensorDesc are all assumed to be of type double.
;The HALF and INT8 data types may be mixed with the FLOAT data types. In these cases, the computation enum of reduceTensorDesc is required to be of type FLOAT. 

(defun get-convolution-backward-filter-algorithm (input-descriptor filter-descriptor convolution-descriptor output-descriptor cudnn-handler)
  (petalisp.utilities:with-hash-table-memoization
    ((list input-descriptor filter-descriptor convolution-descriptor output-descriptor *cudnn-autotune* :backward-data))
    (convolution-algorithms cudnn-handler)
    (with-foreign-object (algo-count '(:pointer :int))
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionBackwardFilterAlgorithmMaxCount (cudnn-handle cudnn-handler)
                                                                                                algo-count)))
      (with-foreign-object (perf-results 'cudnnConvolutionBwdFilterAlgoPerf-t (mem-aref algo-count :int))
        (if *cudnn-autotune*
            ;(cffi:defcfun ("cudnnGetConvolutionBackwardFilterAlgorithm_v7" cudnngetconvolutionbackwardfilteralgorithm-v7) cudnnStatus-t
               ;(handle cudnnHandle-t)
               ;(srcdesc cudnnTensorDescriptor-t)
               ;(diffdesc cudnnTensorDescriptor-t)
               ;(convdesc cudnnConvolutionDescriptor-t)
               ;(graddesc cudnnFilterDescriptor-t)
               ;(requestedalgocount :int)
               ;(returnedalgocount (:pointer :int))
               ;(perfresults (:pointer cudnnConvolutionBwdFilterAlgoPerf-t)))
            (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnFindConvolutionBackwardFilterAlgorithm (cudnn-handle cudnn-handler)
                                                                                               output-descriptor
                                                                                               input-descriptor
                                                                                               convolution-descriptor
                                                                                               filter-descriptor
                                                                                               (mem-aref algo-count :int)
                                                                                               algo-count
                                                                                               perf-results)))
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionBackwardFilterAlgorithm_v7 (cudnn-handle cudnn-handler)
                                                                                               output-descriptor
                                                                                               input-descriptor
                                                                                               convolution-descriptor
                                                                                               filter-descriptor
                                                                                               (mem-aref algo-count :int)
                                                                                               algo-count
                                                                                               perf-results))))
        (assert (> (mem-aref algo-count :int 0)))
        (foreign-slot-value (mem-aref perf-results 'cudnnConvolutionBwdFilterAlgoPerf-t) 'cudnnConvolutionBwdFilterAlgoPerf-t 'algo)))))

(defun get-convolution-backward-data-algorithm (input-descriptor filter-descriptor convolution-descriptor output-descriptor cudnn-handler)
  (petalisp.utilities:with-hash-table-memoization
    ((list input-descriptor filter-descriptor convolution-descriptor output-descriptor *cudnn-autotune* :backward-filter))
    (convolution-algorithms cudnn-handler)
    (with-foreign-object (algo-count '(:pointer :int))
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionBackwardDataAlgorithmMaxCount (cudnn-handle cudnn-handler)
                                                                                              algo-count)))
      (with-foreign-object (perf-results 'cudnnConvolutionBwdDataAlgoPerf-t (mem-aref algo-count :int))
        (if *cudnn-autotune*
            ;(handle cudnnHandle-t)
            ;(wdesc cudnnFilterDescriptor-t)
          ;(dydesc cudnnTensorDescriptor-t)
          ;(convdesc cudnnConvolutionDescriptor-t)
          ;(dxdesc cudnnTensorDescriptor-t)
          ;(requestedalgocount :int)
          ;(returnedalgocount (:pointer :int))
          ;(perfresults (:pointer cudnnConvolutionBwdDataAlgoPerf-t)))
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnFindConvolutionBackwardDataAlgorithm (cudnn-handle cudnn-handler)
                                                                                           filter-descriptor
                                                                                           input-descriptor
                                                                                           convolution-descriptor
                                                                                           output-descriptor
                                                                                           (mem-aref algo-count :int)
                                                                                           algo-count
                                                                                           perf-results)))
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionBackwardDataAlgorithm_v7 (cudnn-handle cudnn-handler)
                                                                                             filter-descriptor
                                                                                             input-descriptor
                                                                                             convolution-descriptor
                                                                                             output-descriptor
                                                                                             (mem-aref algo-count :int)
                                                                                             algo-count
                                                                                             perf-results))))
        (assert (> (mem-aref algo-count :int 0)))
        (foreign-slot-value (mem-aref perf-results 'cudnnConvolutionBwdDataAlgoPerf-t) 'cudnnConvolutionBwdDataAlgoPerf-t 'algo)))))

(defun get-convolution-forward-algorithm (input-descriptor filter-descriptor convolution-descriptor output-descriptor cudnn-handler)
  (petalisp.utilities:with-hash-table-memoization
    ((list input-descriptor filter-descriptor convolution-descriptor output-descriptor *cudnn-autotune*))
    (convolution-algorithms cudnn-handler)
    (with-foreign-object (algo-count '(:pointer :int))
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionForwardAlgorithmMaxCount (cudnn-handle cudnn-handler)
                                                                                         algo-count)))
      (with-foreign-object (perf-results 'cudnnConvolutionFwdAlgoPerf-t (mem-aref algo-count :int))
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
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnFindConvolutionForwardAlgorithm (cudnn-handle cudnn-handler)
                                                                                      input-descriptor
                                                                                      filter-descriptor
                                                                                      convolution-descriptor
                                                                                      output-descriptor
                                                                                      (mem-aref algo-count :int)
                                                                                      algo-count
                                                                                      perf-results)))
          ;;cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
          ;;cudnnHandle_t                       handle,
          ;;const cudnnTensorDescriptor_t       xDesc,
          ;;const cudnnFilterDescriptor_t       wDesc,
          ;;const cudnnConvolutionDescriptor_t  convDesc,
          ;;const cudnnTensorDescriptor_t       yDesc,
          ;;const int                           requestedAlgoCount,
          ;;int                                *returnedAlgoCount,
          ;;cudnnConvolutionFwdAlgoPerf_t      *perfResults)
          (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionForwardAlgorithm_v7 (cudnn-handle cudnn-handler)
                                                                                        input-descriptor
                                                                                        filter-descriptor
                                                                                        convolution-descriptor
                                                                                        output-descriptor
                                                                                        (mem-aref algo-count :int)
                                                                                        algo-count
                                                                                        perf-results))))
        (assert (> (mem-aref algo-count :int 0)))
        (foreign-slot-value (mem-aref perf-results 'cudnnConvolutionFwdAlgoPerf-t) 'cudnnConvolutionFwdAlgoPerf-t 'algo)))))

(defun check-output-dimensions (output-array input-descriptor convolution-descriptor filter-descriptor)
  (with-foreign-objects ((result :int (rank output-array)))
    (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnGetConvolutionNdForwardOutputDim
                                            convolution-descriptor
                                            input-descriptor
                                            filter-descriptor
                                            (rank output-array)
                                            result)))
    (assert
      (equalp
        (cuda-array-shape output-array)
        (loop for i from 0 below (rank output-array)
              for s in (cuda-array-shape output-array)
              collect (mem-aref result :int i))))))

(defun create-activation-descriptor (activation-mode activation-coefficient activation-nan-propagation cudnn-handler)
  (petalisp.utilities:with-hash-table-memoization
    ((list activation-mode activation-coefficient activation-nan-propagation))
    (activation-desciptors cudnn-handler)
    (with-foreign-object (activation-descriptor '(:pointer cudnnActivationDescriptor-t))
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnCreateActivationDescriptor activation-descriptor)))
      (assert (equalp :CUDNN-STATUS-SUCCESS (cudnnSetActivationDescriptor (mem-aref activation-descriptor 'cudnnActivationDescriptor-t)
                                                                          (or activation-mode :cudnn-activation-identity)
                                                                          activation-coefficient
                                                                          activation-nan-propagation)))
      (mem-aref activation-descriptor 'cudnnActivationDescriptor-t))))

(defvar *convolution-workspace-fun*
  (make-hash :test #'equal
             :initial-contents
             '(:forward #'cudnnGetConvolutionForwardWorkspaceSize
               :backward-data #'cudnnGetConvolutionBackwardDataWorkspaceSize)
               :backward-filter #'cudnnGetConvolutionBackwardFilterWorkspaceSize))

(defun cudnn-convolution (input-array
                           filter-array
                           output-array
                           cudnn-handler
                           &key
                           algorithm
                           (input-factor 1.0)
                           (accumulator-factor 0.0)
                           group-count
                           math-type
                           (mode :cudnn-convolution)
                           bias-array
                           pre-bias-accumulator-array
                           activation-mode
                           (activation-coefficient 0.0)
                           (activation-nan-propagation :cudnn-not-propagate-nan)
                           (paddings (mapcar (lambda (s) (floor s 2)) (subseq (cuda-array-shape filter-array) 2)))
                           (filter-strides (make-list (- (rank input-array) 2) :initial-element 1))
                           (dilations (make-list (- (rank input-array) 2) :initial-element 1))
                           (filter-format :cudnn-tensor-nchw)
                           (direction :forward))
  "
  Possible activation modes:

  (cffi:defcenum cudnnactivationmode-t-enum
    (:cudnn-activation-sigmoid 0)
    (:cudnn-activation-relu 1)
    (:cudnn-activation-tanh 2)
    (:cudnn-activation-clipped-relu 3)
    (:cudnn-activation-elu 4)
    (:cudnn-activation-identity 5))
  "
  (let* ((input-descriptor (cudnn-create-tensor-descriptor input-array cudnn-handler))
         (output-descriptor (cudnn-create-tensor-descriptor output-array cudnn-handler))
         (convolution-descriptor (cudnn-create-convolution-descriptor input-array paddings dilations filter-strides mode cudnn-handler))
         (filter-descriptor (cudnn-create-filter-descriptor filter-array filter-format cudnn-handler))
         (pre-bias-descriptor (when bias-array
                                (cudnn-create-tensor-descriptor pre-bias-accumulator-array cudnn-handler)))
         (bias-descriptor (when bias-array
                            (cudnn-create-tensor-descriptor bias-array cudnn-handler)))
         (double-or-float (if (equalp (cuda-array-type input-array) :double) :double :float))
         (convolution-algorithm (progn
                                  (when group-count
                                    (cudnnSetConvolutionGroupCount convolution-descriptor group-count))
                                  (when math-type
                                    (cudnnSetConvolutionMathType convolution-descriptor math-type))
                                  (check-output-dimensions output-array input-descriptor convolution-descriptor filter-descriptor)
                                  (or algorithm (alexandria:switch (direction :test #'equalp)
                                                  (:forward (get-convolution-forward-algorithm input-descriptor
                                                                                               filter-descriptor
                                                                                               convolution-descriptor
                                                                                               output-descriptor
                                                                                               cudnn-handler))
                                                  (:backward-data (get-convolution-backward-data-algorithm input-descriptor
                                                                                                           filter-descriptor
                                                                                                           convolution-descriptor
                                                                                                           output-descriptor
                                                                                                           cudnn-handler))
                                                  (:backward-filter (get-convolution-backward-filter-algorithm input-descriptor
                                                                                                               filter-descriptor
                                                                                                               convolution-descriptor
                                                                                                               output-descriptor
                                                                                                               cudnn-handler)))))))
    (with-foreign-objects ((workspace-min-size '(:pointer :int))
                           (alpha '(:pointer :double))
                           (beta '(:pointer :double)))
      ; this routine supports mixing data types, then alpha, beta are float or else everything is double
      (setf (mem-ref alpha double-or-float) (convert-to-foreign input-factor double-or-float))
      (setf (mem-ref beta double-or-float) (convert-to-foreign accumulator-factor double-or-float))
      (assert (equalp :CUDNN-STATUS-SUCCESS
                      (funcall (gethash direction *convolution-workspace-fun*)
                               (cudnn-handle cudnn-handler)
                               input-descriptor
                               filter-descriptor
                               convolution-descriptor
                               output-descriptor
                               convolution-algorithm
                               workspace-min-size)))
      (multiple-value-bind (workspace workspace-size) (allocate-workspace (mem-ref workspace-min-size :int) cudnn-handler)
        (if (and (or bias-array activation-mode) (equalp direction :forward))
            (let ((activation-desciptor (create-activation-descriptor activation-mode activation-coefficient activation-nan-propagation cudnn-handler)))
              (assert bias-array)
              (assert (equalp :CUDNN-STATUS-SUCCESS
                              (cudnnConvolutionBiasActivationForward (cudnn-handle cudnn-handler)
                                                                     alpha
                                                                     input-descriptor
                                                                     (make-pointer (device-ptr input-array))
                                                                     filter-descriptor
                                                                     (make-pointer (device-ptr filter-array))
                                                                     convolution-descriptor
                                                                     convolution-algorithm
                                                                     (make-pointer workspace)
                                                                     workspace-size
                                                                     beta
                                                                     (or pre-bias-descriptor output-descriptor)
                                                                     (make-pointer (device-ptr (or pre-bias-accumulator-array output-array)))
                                                                     bias-descriptor
                                                                     (make-pointer (device-ptr bias-array))
                                                                     (mem-aref activation-desciptor 'cudnnActivationDescriptor-t)
                                                                     output-descriptor
                                                                     (make-pointer (device-ptr output-array))))))
            (assert (equalp :CUDNN-STATUS-SUCCESS
                            (alexandria:switch (direction :test #'equalp)
                              ;; So funny to randomly swap those parameters! Thanks you cudnn...
                              (:forward (cudnnConvolutionForward (cudnn-handle cudnn-handler)
                                                                 alpha
                                                                 input-descriptor
                                                                 (make-pointer (device-ptr input-array))
                                                                 filter-descriptor
                                                                 (make-pointer (device-ptr filter-array))
                                                                 convolution-descriptor
                                                                 convolution-algorithm
                                                                 (make-pointer workspace)
                                                                 workspace-size
                                                                 beta
                                                                 output-descriptor
                                                                 (make-pointer (device-ptr output-array))))
                              (:backward-data (cudnnConvolutionBackwardData (cudnn-handle cudnn-handler)
                                                                            alpha
                                                                            filter-descriptor
                                                                            (make-pointer (device-ptr filter-array))
                                                                            input-descriptor
                                                                            (make-pointer (device-ptr input-array))
                                                                            convolution-descriptor
                                                                            convolution-algorithm
                                                                            (make-pointer workspace)
                                                                            workspace-size
                                                                            beta
                                                                            output-descriptor
                                                                            (make-pointer (device-ptr output-array))))
                              (:backward-filter (cudnnConvolutionBackwardFilter (cudnn-handle cudnn-handler)
                                                                                alpha
                                                                                output-descriptor
                                                                                (make-pointer (device-ptr output-array))
                                                                                input-descriptor
                                                                                (make-pointer (device-ptr input-array))
                                                                                convolution-descriptor
                                                                                convolution-algorithm
                                                                                (make-pointer workspace)
                                                                                workspace-size
                                                                                beta
                                                                                filter-descriptor
                                                                                (make-pointer (device-ptr filter-array))))))))))))

;cudnnStatus_t cudnnMultiHeadAttnForward(
;cudnnHandle_t handle,
;const cudnnAttnDescriptor_t attnDesc,
;int currIdx,
;const int loWinIdx[],
;const int hiWinIdx[],
;const int devSeqLengthsQO[],
;const int devSeqLengthsKV[],
;const cudnnSeqDataDescriptor_t qDesc,
;const void *queries,
;const void *residuals,
;const cudnnSeqDataDescriptor_t kDesc,
;const void *keys,
;const cudnnSeqDataDescriptor_t vDesc,
;const void *values,
;const cudnnSeqDataDescriptor_t oDesc,
;void *out,
;size_t weightSizeInBytes,
;const void *weights,
;size_t workSpaceSizeInBytes,
;void *workSpace,
;size_t reserveSpaceSizeInBytes,
;void *reserveSpace);
