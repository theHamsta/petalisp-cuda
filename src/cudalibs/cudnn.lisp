(in-package petalisp-cuda.cudalibs)

;;;
;;; DEFCUDNNFUN macro
;;;

; from cl-cuda defcufun
(cl:defmacro defcudnnfun ((name c-name &key disable-fp-traps) return-type
                       &rest arguments)
  (let ((%name (format-symbol (cl:symbol-package name) "%~A" name))
        (argument-vars (cl:mapcar #'cl:car arguments)))
    (if (cl:not *cudnn-not-found*)
        `(cl:progn
           (cl:defun ,name ,argument-vars
             (cl:assert (cl:equalp :CUDNN-STATUS-SUCCESS) (,%name ,@argument-vars)))
           (cffi:defcfun (,%name ,c-name) ,return-type ,@arguments))
        `(cl:defun ,name ,argument-vars
           (cl:error "CUDNN not available")))))

;; next section imported from file /usr/include/cudnn.h

#| MACRO_DEFINITION
(defconstant +cudnn-h-+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-major+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-minor+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-patchlevel+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-version+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnnwinapi+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-dim-max+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-lrn-min-n+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-lrn-max-n+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-lrn-min-k+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-lrn-min-beta+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-bn-min-epsilon+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-seqdata-dim-count+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-attn-querymap-all-to-one+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-attn-querymap-one-to-one+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-attn-disable-proj-biases+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-attn-enable-proj-biases+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-attn-wkind-count+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-sev-error-en+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-sev-warning-en+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudnn-sev-info-en+ ACTUAL_VALUE_HERE)
|#

;TODO make magic macro that checks if return value is :CUDNN-STATUS-SUCCESS

(cffi:defcenum cudnnStatus-t
	(:CUDNN_STATUS_SUCCESS #.0)
	(:CUDNN_STATUS_NOT_INITIALIZED #.1)
	(:CUDNN_STATUS_ALLOC_FAILED #.2)
	(:CUDNN_STATUS_BAD_PARAM #.3)
	(:CUDNN_STATUS_INTERNAL_ERROR #.4)
	(:CUDNN_STATUS_INVALID_VALUE #.5)
	(:CUDNN_STATUS_ARCH_MISMATCH #.6)
	(:CUDNN_STATUS_MAPPING_ERROR #.7)
	(:CUDNN_STATUS_EXECUTION_FAILED #.8)
	(:CUDNN_STATUS_NOT_SUPPORTED #.9)
	(:CUDNN_STATUS_LICENSE_ERROR #.10))

(cffi:defcstruct cudnnContext)

(cffi:defctype cudnnhandle-t (:pointer (:struct cudnnContext)))

(cffi:defcfun "cudnngetversion" :int)

(cffi:defcfun "cudnngetcudartversion" :int)

(cffi:defcenum cudnnstatus-t-enum
  (:cudnn-status-success 0)
  (:cudnn-status-not-initialized 1)
  (:cudnn-status-alloc-failed 2)
  (:cudnn-status-bad-param 3)
  (:cudnn-status-internal-error 4)
  (:cudnn-status-invalid-value 5)
  (:cudnn-status-arch-mismatch 6)
  (:cudnn-status-mapping-error 7)
  (:cudnn-status-execution-failed 8)
  (:cudnn-status-not-supported 9)
  (:cudnn-status-license-error 10)
  (:cudnn-status-runtime-prerequisite-missing 11)
  (:cudnn-status-runtime-in-progress 12)
  (:cudnn-status-runtime-fp-overflow 13))

(cffi:defctype cudnnstatus-t cudnnstatus-t-enum)

(cffi:defcfun "cudnngeterrorstring" (:pointer :char)
  (status cudnnStatus-t))

(cffi:defcstruct cudnnruntimetag-t)

(cffi:defctype cudnnruntimetag-t (:struct cudnnRuntimeTag-t))

(cffi:defcenum cudnnerrquerymode-t-enum
  (:cudnn-errquery-rawcode 0)
  (:cudnn-errquery-nonblocking 1)
  (:cudnn-errquery-blocking 2))

(cffi:defctype cudnnerrquerymode-t cudnnerrquerymode-t-enum)

(cffi:defcfun "cudnnqueryruntimeerror" cudnnStatus-t
  (handle cudnnHandle-t)
  (rstatus (:pointer cudnnStatus-t))
  (mode cudnnErrQueryMode-t)
  (tag (:pointer cudnnRuntimeTag-t)))

;(cffi:defcfun "cudnngetproperty" cudnnStatus-t
  ;(type libraryPropertyType)
  ;(value (:pointer :int)))

(cffi:defcfun "cudnnCreate" cudnnStatus-t
  (handle :pointer))

(cffi:defcfun "cudnnDestroy" cudnnStatus-t
  (handle cudnnHandle-t))

;(cffi:defcfun "cudnnsetstream" cudnnStatus-t
  ;(handle cudnnHandle-t)
  ;(streamid (:pointer)))

(cffi:defcfun "cudnngetstream" cudnnStatus-t
  (handle cudnnHandle-t)
  (streamid (:pointer)))

(cffi:defcstruct cudnntensorstruct)

(cffi:defctype cudnntensordescriptor-t (:pointer (:struct cudnnTensorStruct)))

(cffi:defcstruct cudnnconvolutionstruct)

(cffi:defctype cudnnconvolutiondescriptor-t (:pointer (:struct cudnnConvolutionStruct)))

(cffi:defcstruct cudnnpoolingstruct)

(cffi:defctype cudnnpoolingdescriptor-t (:pointer (:struct cudnnPoolingStruct)))

(cffi:defcstruct cudnnfilterstruct)

(cffi:defctype cudnnfilterdescriptor-t (:pointer (:struct cudnnFilterStruct)))

(cffi:defcstruct cudnnlrnstruct)

(cffi:defctype cudnnlrndescriptor-t (:pointer (:struct cudnnLRNStruct)))

(cffi:defcstruct cudnnactivationstruct)

(cffi:defctype cudnnactivationdescriptor-t (:pointer (:struct cudnnActivationStruct)))

(cffi:defcstruct cudnnspatialtransformerstruct)

(cffi:defctype cudnnspatialtransformerdescriptor-t (:pointer (:struct cudnnSpatialTransformerStruct)))

(cffi:defcstruct cudnnoptensorstruct)

(cffi:defctype cudnnoptensordescriptor-t (:pointer (:struct cudnnOpTensorStruct)))

(cffi:defcstruct cudnnreducetensorstruct)

(cffi:defctype cudnnreducetensordescriptor-t (:pointer (:struct cudnnReduceTensorStruct)))

(cffi:defcstruct cudnnctclossstruct)

(cffi:defctype cudnnctclossdescriptor-t (:pointer (:struct cudnnCTCLossStruct)))

(cffi:defcstruct cudnntensortransformstruct)

(cffi:defctype cudnntensortransformdescriptor-t (:pointer (:struct cudnnTensorTransformStruct)))

(cffi:defcenum cudnndatatype-t-enum
  (:cudnn-data-float 0)
  (:cudnn-data-double 1)
  (:cudnn-data-half 2)
  (:cudnn-data-int8 3)
  (:cudnn-data-int32 4)
  (:cudnn-data-int8x4 5)
  (:cudnn-data-uint8 6)
  (:cudnn-data-uint8x4 7)
  (:cudnn-data-int8x32 8))

(cffi:defctype cudnndatatype-t cudnndatatype-t-enum)

(cffi:defcenum cudnnmathtype-t-enum
  (:cudnn-default-math 0)
  (:cudnn-tensor-op-math 1)
  (:cudnn-tensor-op-math-allow-conversion 2))

(cffi:defctype cudnnmathtype-t cudnnmathtype-t-enum)

(cffi:defcenum cudnnnanpropagation-t-enum
  (:cudnn-not-propagate-nan 0)
  (:cudnn-propagate-nan 1))

(cffi:defctype cudnnnanpropagation-t cudnnnanpropagation-t-enum)

(cffi:defcenum cudnndeterminism-t-enum
  (:cudnn-non-deterministic 0)
  (:cudnn-deterministic 1))

(cffi:defctype cudnndeterminism-t cudnndeterminism-t-enum)

(cffi:defcenum cudnnreordertype-t-enum
  (:cudnn-default-reorder 0)
  (:cudnn-no-reorder 1))

(cffi:defctype cudnnreordertype-t cudnnreordertype-t-enum)

(cffi:defcfun "cudnnCreateTensorDescriptor" cudnnStatus-t
  (tensordesc (:pointer cudnnTensorDescriptor-t)))

(cffi:defcenum cudnntensorformat-t-enum
  (:cudnn-tensor-nchw 0)
  (:cudnn-tensor-nhwc 1)
  (:cudnn-tensor-nchw-vect-c 2))

(cffi:defctype cudnntensorformat-t cudnntensorformat-t-enum)

(cffi:defcfun "cudnnsettensor4ddescriptor" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (format cudnnTensorFormat-t)
  (datatype cudnnDataType-t)
  (n :int)
  (c :int)
  (h :int)
  (w :int))

(cffi:defcfun "cudnnsettensor4ddescriptorex" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (datatype cudnnDataType-t)
  (n :int)
  (c :int)
  (h :int)
  (w :int)
  (nstride :int)
  (cstride :int)
  (hstride :int)
  (wstride :int))

(cffi:defcfun "cudnngettensor4ddescriptor" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (datatype (:pointer cudnnDataType-t))
  (n (:pointer :int))
  (c (:pointer :int))
  (h (:pointer :int))
  (w (:pointer :int))
  (nstride (:pointer :int))
  (cstride (:pointer :int))
  (hstride (:pointer :int))
  (wstride (:pointer :int)))

(cffi:defcfun "cudnnSetTensorNdDescriptor" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (datatype cudnnDataType-t)
  (nbdims :int)
  (dima (:pointer :int) ; array 
)
  (stridea (:pointer :int) ; array 
))

(cffi:defcfun "cudnnsettensornddescriptorex" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (format cudnnTensorFormat-t)
  (datatype cudnnDataType-t)
  (nbdims :int)
  (dima (:pointer :int) ; array 
))

(cffi:defcfun "cudnngettensornddescriptor" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (nbdimsrequested :int)
  (datatype (:pointer cudnnDataType-t))
  (nbdims (:pointer :int))
  (dima (:pointer :int) ; array 
)
  (stridea (:pointer :int) ; array 
))

(cffi:defcfun "cudnngettensorsizeinbytes" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t)
  (size (:pointer :int)))

(cffi:defcfun "cudnndestroytensordescriptor" cudnnStatus-t
  (tensordesc cudnnTensorDescriptor-t))

(cffi:defcenum cudnnfoldingdirection-t-enum
  (:cudnn-transform-fold 0)
  (:cudnn-transform-unfold 1))

(cffi:defctype cudnnfoldingdirection-t cudnnfoldingdirection-t-enum)

(cffi:defcfun "cudnninittransformdest" cudnnStatus-t
  "Create a destination descriptor for cudnnTransformTensor"
  (transformdesc cudnnTensorTransformDescriptor-t)
  (srcdesc cudnnTensorDescriptor-t)
  (destdesc cudnnTensorDescriptor-t)
  (destsizeinbytes (:pointer :int)))

(cffi:defcfun "cudnncreatetensortransformdescriptor" cudnnStatus-t
  "Create an empty tensor transform descriptor"
  (transformdesc (:pointer cudnnTensorTransformDescriptor-t)))

(cffi:defcfun "cudnnsettensortransformdescriptor" cudnnStatus-t
  "Initialize a previously created tensor transform descriptor."
  (transformdesc cudnnTensorTransformDescriptor-t)
  (nbdims :uint32)
  (destformat cudnnTensorFormat-t)
  (padbeforea (:pointer :int32) ; array 
)
  (padaftera (:pointer :int32) ; array 
)
  (folda (:pointer :uint32) ; array 
)
  (direction cudnnFoldingDirection-t))

(cffi:defcfun "cudnngettensortransformdescriptor" cudnnStatus-t
  "Retrieves the values stored in a previously initialized tensor transform
  descriptor."
  (transformdesc cudnnTensorTransformDescriptor-t)
  (nbdimsrequested :uint32)
  (destformat (:pointer cudnnTensorFormat-t))
  (padbeforea (:pointer :int32) ; array 
)
  (padaftera (:pointer :int32) ; array 
)
  (folda (:pointer :uint32) ; array 
)
  (direction (:pointer cudnnFoldingDirection-t)))

(cffi:defcfun "cudnndestroytensortransformdescriptor" cudnnStatus-t
  "Destroys a previously created tensor transform descriptor."
  (transformdesc cudnnTensorTransformDescriptor-t))

(cffi:defcfun "cudnntransformtensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnntransformtensorex" cudnnStatus-t
  (handle cudnnHandle-t)
  (transdesc cudnnTensorTransformDescriptor-t)
  (alpha (:pointer :void))
  (srcdesc cudnnTensorDescriptor-t)
  (srcdata (:pointer :void))
  (beta (:pointer :void))
  (destdesc cudnnTensorDescriptor-t)
  (destdata (:pointer :void)))

(cffi:defcfun "cudnngetfoldedconvbackwarddatadescriptors" cudnnStatus-t
  (handle cudnnHandle-t)
  (filterdesc cudnnFilterDescriptor-t)
  (diffdesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (graddesc cudnnTensorDescriptor-t)
  (transformformat cudnnTensorFormat-t)
  (foldedfilterdesc cudnnFilterDescriptor-t)
  (paddeddiffdesc cudnnTensorDescriptor-t)
  (foldedconvdesc cudnnConvolutionDescriptor-t)
  (foldedgraddesc cudnnTensorDescriptor-t)
  (filterfoldtransdesc cudnnTensorTransformDescriptor-t)
  (diffpadtransdesc cudnnTensorTransformDescriptor-t)
  (gradfoldtransdesc cudnnTensorTransformDescriptor-t)
  (gradunfoldtransdesc cudnnTensorTransformDescriptor-t))

(cffi:defcfun "cudnnaddtensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (adesc cudnnTensorDescriptor-t)
  (a (:pointer :void))
  (beta (:pointer :void))
  (cdesc cudnnTensorDescriptor-t)
  (c (:pointer :void)))

(cffi:defcenum cudnnoptensorop-t-enum
  (:cudnn-op-tensor-add 0)
  (:cudnn-op-tensor-mul 1)
  (:cudnn-op-tensor-min 2)
  (:cudnn-op-tensor-max 3)
  (:cudnn-op-tensor-sqrt 4)
  (:cudnn-op-tensor-not 5))

(cffi:defctype cudnnoptensorop-t cudnnoptensorop-t-enum)

(cffi:defcfun "cudnncreateoptensordescriptor" cudnnStatus-t
  (optensordesc (:pointer cudnnOpTensorDescriptor-t)))

(cffi:defcfun "cudnnsetoptensordescriptor" cudnnStatus-t
  (optensordesc cudnnOpTensorDescriptor-t)
  (optensorop cudnnOpTensorOp-t)
  (optensorcomptype cudnnDataType-t)
  (optensornanopt cudnnNanPropagation-t))

(cffi:defcfun "cudnngetoptensordescriptor" cudnnStatus-t
  (optensordesc cudnnOpTensorDescriptor-t)
  (optensorop (:pointer cudnnOpTensorOp-t))
  (optensorcomptype (:pointer cudnnDataType-t))
  (optensornanopt (:pointer cudnnNanPropagation-t)))

(cffi:defcfun "cudnndestroyoptensordescriptor" cudnnStatus-t
  (optensordesc cudnnOpTensorDescriptor-t))

(cffi:defcfun "cudnnoptensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (optensordesc cudnnOpTensorDescriptor-t)
  (alpha1 (:pointer :void))
  (adesc cudnnTensorDescriptor-t)
  (a (:pointer :void))
  (alpha2 (:pointer :void))
  (bdesc cudnnTensorDescriptor-t)
  (b (:pointer :void))
  (beta (:pointer :void))
  (cdesc cudnnTensorDescriptor-t)
  (c (:pointer :void)))

(cffi:defcenum cudnnreducetensorop-t-enum
  (:cudnn-reduce-tensor-add 0)
  (:cudnn-reduce-tensor-mul 1)
  (:cudnn-reduce-tensor-min 2)
  (:cudnn-reduce-tensor-max 3)
  (:cudnn-reduce-tensor-amax 4)
  (:cudnn-reduce-tensor-avg 5)
  (:cudnn-reduce-tensor-norm1 6)
  (:cudnn-reduce-tensor-norm2 7)
  (:cudnn-reduce-tensor-mul-no-zeros 8))

(cffi:defctype cudnnreducetensorop-t cudnnreducetensorop-t-enum)

(cffi:defcenum cudnnreducetensorindices-t-enum
  (:cudnn-reduce-tensor-no-indices 0)
  (:cudnn-reduce-tensor-flattened-indices 1))

(cffi:defctype cudnnReduceTensorIndices-t cudnnreducetensorindices-t-enum)

(cffi:defcenum cudnnindicestype-t-enum
  (:cudnn-32bit-indices 0)
  (:cudnn-64bit-indices 1)
  (:cudnn-16bit-indices 2)
  (:cudnn-8bit-indices 3))

(cffi:defctype cudnnindicestype-t cudnnindicestype-t-enum)

(cffi:defcfun "cudnnCreateReduceTensorDescriptor" cudnnStatus-t
  (reducetensordesc (:pointer cudnnReduceTensorDescriptor-t)))

(cffi:defcfun "cudnnSetReduceTensorDescriptor" cudnnStatus-t
  (reducetensordesc cudnnReduceTensorDescriptor-t)
  (reducetensorop cudnnReduceTensorOp-t)
  (reducetensorcomptype cudnnDataType-t)
  (reducetensornanopt cudnnNanPropagation-t)
  (reducetensorindices cudnnReduceTensorIndices-t)
  (reducetensorindicestype cudnnIndicesType-t))

(cffi:defcfun "cudnnGetReduceTensorDescriptor" cudnnStatus-t
  (reducetensordesc cudnnReduceTensorDescriptor-t)
  (reducetensorop (:pointer cudnnReduceTensorOp-t))
  (reducetensorcomptype (:pointer cudnnDataType-t))
  (reducetensornanopt (:pointer cudnnNanPropagation-t))
  (reducetensorindices (:pointer cudnnReduceTensorIndices-t))
  (reducetensorindicestype (:pointer cudnnIndicesType-t)))

(cffi:defcfun "cudnnDestroyReduceTensorDescriptor" cudnnStatus-t
  (reducetensordesc cudnnReduceTensorDescriptor-t))

(cffi:defcfun "cudnnGetReductionIndicesSize" cudnnStatus-t
  (handle cudnnHandle-t)
  (reducetensordesc cudnnReduceTensorDescriptor-t)
  (adesc cudnnTensorDescriptor-t)
  (cdesc cudnnTensorDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnGetReductionWorkspaceSize" cudnnStatus-t
  (handle cudnnHandle-t)
  (reducetensordesc cudnnReduceTensorDescriptor-t)
  (adesc cudnnTensorDescriptor-t)
  (cdesc cudnnTensorDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnReduceTensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (reducetensordesc cudnnReduceTensorDescriptor-t)
  (indices (:pointer :void))
  (indicessizeinbytes :int)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (alpha (:pointer :void))
  (adesc cudnnTensorDescriptor-t)
  (a (:pointer :void))
  (beta (:pointer :void))
  (cdesc cudnnTensorDescriptor-t)
  (c (:pointer :void)))

(cffi:defcfun "cudnnsettensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (valueptr (:pointer :void)))

(cffi:defcfun "cudnnscaletensor" cudnnStatus-t
  (handle cudnnHandle-t)
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (alpha (:pointer :void)))

(cffi:defcenum cudnnconvolutionmode-t-enum
  (:cudnn-convolution 0)
  (:cudnn-cross-correlation 1))

(cffi:defctype cudnnconvolutionmode-t cudnnconvolutionmode-t-enum)

(cffi:defcfun "cudnncreatefilterdescriptor" cudnnStatus-t
  (filterdesc (:pointer cudnnFilterDescriptor-t)))

(cffi:defcfun "cudnnsetfilter4ddescriptor" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t)
  (datatype cudnnDataType-t)
  (format cudnnTensorFormat-t)
  (k :int)
  (c :int)
  (h :int)
  (w :int))

(cffi:defcfun "cudnngetfilter4ddescriptor" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t)
  (datatype (:pointer cudnnDataType-t))
  (format (:pointer cudnnTensorFormat-t))
  (k (:pointer :int))
  (c (:pointer :int))
  (h (:pointer :int))
  (w (:pointer :int)))

(cffi:defcfun "cudnnsetfilternddescriptor" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t)
  (datatype cudnnDataType-t)
  (format cudnnTensorFormat-t)
  (nbdims :int)
  (filterdima (:pointer :int) ; array 
))

(cffi:defcfun "cudnngetfilternddescriptor" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t)
  (nbdimsrequested :int)
  (datatype (:pointer cudnnDataType-t))
  (format (:pointer cudnnTensorFormat-t))
  (nbdims (:pointer :int))
  (filterdima (:pointer :int) ; array 
))

(cffi:defcfun "cudnngetfiltersizeinbytes" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t)
  (size (:pointer :int)))

(cffi:defcfun "cudnntransformfilter" cudnnStatus-t
  (handle cudnnHandle-t)
  (transdesc cudnnTensorTransformDescriptor-t)
  (alpha (:pointer :void))
  (srcdesc cudnnFilterDescriptor-t)
  (srcdata (:pointer :void))
  (beta (:pointer :void))
  (destdesc cudnnFilterDescriptor-t)
  (destdata (:pointer :void)))

(cffi:defcfun "cudnndestroyfilterdescriptor" cudnnStatus-t
  (filterdesc cudnnFilterDescriptor-t))

(cffi:defcfun "cudnnreorderfilterandbias" cudnnStatus-t
  (handle cudnnHandle-t)
  (filterdesc cudnnFilterDescriptor-t)
  (reordertype cudnnReorderType-t)
  (filterdata (:pointer :void))
  (reorderedfilterdata (:pointer :void))
  (reorderbias :int)
  (biasdata (:pointer :void))
  (reorderedbiasdata (:pointer :void)))

(cffi:defcfun "cudnncreateconvolutiondescriptor" cudnnStatus-t
  (convdesc (:pointer cudnnConvolutionDescriptor-t)))

(cffi:defcfun "cudnnsetconvolutionmathtype" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (mathtype cudnnMathType-t))

(cffi:defcfun "cudnngetconvolutionmathtype" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (mathtype (:pointer cudnnMathType-t)))

(cffi:defcfun "cudnnsetconvolutiongroupcount" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (groupcount :int))

(cffi:defcfun "cudnngetconvolutiongroupcount" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (groupcount (:pointer :int)))

(cffi:defcfun "cudnnsetconvolutionreordertype" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (reordertype cudnnReorderType-t))

(cffi:defcfun "cudnngetconvolutionreordertype" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (reordertype (:pointer cudnnReorderType-t)))

(cffi:defcfun "cudnnsetconvolution2ddescriptor" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (pad-h :int)
  (pad-w :int)
  (u :int)
  (v :int)
  (dilation-h :int)
  (dilation-w :int)
  (mode cudnnConvolutionMode-t)
  (computetype cudnnDataType-t))

(cffi:defcfun "cudnngetconvolution2ddescriptor" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (pad-h (:pointer :int))
  (pad-w (:pointer :int))
  (u (:pointer :int))
  (v (:pointer :int))
  (dilation-h (:pointer :int))
  (dilation-w (:pointer :int))
  (mode (:pointer cudnnConvolutionMode-t))
  (computetype (:pointer cudnnDataType-t)))

(cffi:defcfun "cudnngetconvolution2dforwardoutputdim" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (inputtensordesc cudnnTensorDescriptor-t)
  (filterdesc cudnnFilterDescriptor-t)
  (n (:pointer :int))
  (c (:pointer :int))
  (h (:pointer :int))
  (w (:pointer :int)))

(cffi:defcfun "cudnnsetconvolutionnddescriptor" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (arraylength :int)
  (pada (:pointer :int) ; array 
)
  (filterstridea (:pointer :int) ; array 
)
  (dilationa (:pointer :int) ; array 
)
  (mode cudnnConvolutionMode-t)
  (computetype cudnnDataType-t))

(cffi:defcfun "cudnngetconvolutionnddescriptor" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (arraylengthrequested :int)
  (arraylength (:pointer :int))
  (pada (:pointer :int) ; array 
)
  (stridea (:pointer :int) ; array 
)
  (dilationa (:pointer :int) ; array 
)
  (mode (:pointer cudnnConvolutionMode-t))
  (computetype (:pointer cudnnDataType-t)))

(cffi:defcfun "cudnngetconvolutionndforwardoutputdim" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t)
  (inputtensordesc cudnnTensorDescriptor-t)
  (filterdesc cudnnFilterDescriptor-t)
  (nbdims :int)
  (tensorouputdima (:pointer :int) ; array 
))

(cffi:defcfun "cudnndestroyconvolutiondescriptor" cudnnStatus-t
  (convdesc cudnnConvolutionDescriptor-t))

(cffi:defcenum cudnnconvolutionfwdpreference-t-enum
  (:cudnn-convolution-fwd-no-workspace 0)
  (:cudnn-convolution-fwd-prefer-fastest 1)
  (:cudnn-convolution-fwd-specify-workspace-limit 2))

(cffi:defctype cudnnconvolutionfwdpreference-t cudnnconvolutionfwdpreference-t-enum)

(cffi:defcenum cudnnconvolutionfwdalgo-t-enum
  (:cudnn-convolution-fwd-algo-implicit-gemm 0)
  (:cudnn-convolution-fwd-algo-implicit-precomp-gemm 1)
  (:cudnn-convolution-fwd-algo-gemm 2)
  (:cudnn-convolution-fwd-algo-direct 3)
  (:cudnn-convolution-fwd-algo-fft 4)
  (:cudnn-convolution-fwd-algo-fft-tiling 5)
  (:cudnn-convolution-fwd-algo-winograd 6)
  (:cudnn-convolution-fwd-algo-winograd-nonfused 7)
  (:cudnn-convolution-fwd-algo-count 8))

(cffi:defctype cudnnconvolutionfwdalgo-t cudnnconvolutionfwdalgo-t-enum)

(cffi:defcstruct cudnnconvolutionfwdalgoperf-t-record)

(cffi:defctype cudnnconvolutionfwdalgoperf-t (:struct cudnnconvolutionfwdalgoperf-t-record))

(cffi:defcfun "cudnngetconvolutionforwardalgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindconvolutionforwardalgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (wdesc cudnnFilterDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionFwdAlgoPerf-t)))

(cffi:defcfun "cudnnfindconvolutionforwardalgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionFwdAlgoPerf-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnngetconvolutionforwardalgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (wdesc cudnnFilterDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (preference cudnnConvolutionFwdPreference-t)
  (memorylimitinbytes :int)
  (algo (:pointer cudnnConvolutionFwdAlgo-t)))

(cffi:defcfun ("cudnngetconvolutionforwardalgorithm_v7" cudnngetconvolutionforwardalgorithm-v7) cudnnStatus-t
  (handle cudnnHandle-t)
  (srcdesc cudnnTensorDescriptor-t)
  (filterdesc cudnnFilterDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (destdesc cudnnTensorDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionFwdAlgoPerf-t)))

(cffi:defcfun "cudnngetconvolutionforwardworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (wdesc cudnnFilterDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (algo cudnnConvolutionFwdAlgo-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnconvolutionforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (algo cudnnConvolutionFwdAlgo-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnconvolutionbiasactivationforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha1 (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (algo cudnnConvolutionFwdAlgo-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (alpha2 (:pointer :void))
  (zdesc cudnnTensorDescriptor-t)
  (z (:pointer :void))
  (biasdesc cudnnTensorDescriptor-t)
  (bias (:pointer :void))
  (activationdesc cudnnActivationDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnconvolutionbackwardbias" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (beta (:pointer :void))
  (dbdesc cudnnTensorDescriptor-t)
  (db (:pointer :void)))

(cffi:defcenum cudnnconvolutionbwdfilterpreference-t-enum
  (:cudnn-convolution-bwd-filter-no-workspace 0)
  (:cudnn-convolution-bwd-filter-prefer-fastest 1)
  (:cudnn-convolution-bwd-filter-specify-workspace-limit 2))

(cffi:defctype cudnnconvolutionbwdfilterpreference-t cudnnconvolutionbwdfilterpreference-t-enum)

(cffi:defcenum cudnnconvolutionbwdfilteralgo-t-enum
  (:cudnn-convolution-bwd-filter-algo-0 0)
  (:cudnn-convolution-bwd-filter-algo-1 1)
  (:cudnn-convolution-bwd-filter-algo-fft 2)
  (:cudnn-convolution-bwd-filter-algo-3 3)
  (:cudnn-convolution-bwd-filter-algo-winograd 4)
  (:cudnn-convolution-bwd-filter-algo-winograd-nonfused 5)
  (:cudnn-convolution-bwd-filter-algo-fft-tiling 6)
  (:cudnn-convolution-bwd-filter-algo-count 7))

(cffi:defctype cudnnconvolutionbwdfilteralgo-t cudnnconvolutionbwdfilteralgo-t-enum)

(cffi:defcstruct cudnnconvolutionbwdfilteralgoperf-t-record)

(cffi:defctype cudnnconvolutionbwdfilteralgoperf-t (:struct cudnnconvolutionbwdfilteralgoperf-t-record))

(cffi:defcfun "cudnngetconvolutionbackwardfilteralgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindconvolutionbackwardfilteralgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (dwdesc cudnnFilterDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdFilterAlgoPerf-t)))

(cffi:defcfun "cudnnfindconvolutionbackwardfilteralgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (dwdesc cudnnFilterDescriptor-t)
  (dw (:pointer :void))
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdFilterAlgoPerf-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnngetconvolutionbackwardfilteralgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (dwdesc cudnnFilterDescriptor-t)
  (preference cudnnConvolutionBwdFilterPreference-t)
  (memorylimitinbytes :int)
  (algo (:pointer cudnnConvolutionBwdFilterAlgo-t)))

(cffi:defcfun ("cudnngetconvolutionbackwardfilteralgorithm_v7" cudnngetconvolutionbackwardfilteralgorithm-v7) cudnnStatus-t
  (handle cudnnHandle-t)
  (srcdesc cudnnTensorDescriptor-t)
  (diffdesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (graddesc cudnnFilterDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdFilterAlgoPerf-t)))

(cffi:defcfun "cudnngetconvolutionbackwardfilterworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (graddesc cudnnFilterDescriptor-t)
  (algo cudnnConvolutionBwdFilterAlgo-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnconvolutionbackwardfilter" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (algo cudnnConvolutionBwdFilterAlgo-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (beta (:pointer :void))
  (dwdesc cudnnFilterDescriptor-t)
  (dw (:pointer :void)))

(cffi:defcenum cudnnconvolutionbwddatapreference-t-enum
  ""
  (:cudnn-convolution-bwd-data-no-workspace 0)
  (:cudnn-convolution-bwd-data-prefer-fastest 1)
  (:cudnn-convolution-bwd-data-specify-workspace-limit 2))

(cffi:defctype cudnnconvolutionbwddatapreference-t cudnnconvolutionbwddatapreference-t-enum)

(cffi:defcenum cudnnconvolutionbwddataalgo-t-enum
  (:cudnn-convolution-bwd-data-algo-0 0)
  (:cudnn-convolution-bwd-data-algo-1 1)
  (:cudnn-convolution-bwd-data-algo-fft 2)
  (:cudnn-convolution-bwd-data-algo-fft-tiling 3)
  (:cudnn-convolution-bwd-data-algo-winograd 4)
  (:cudnn-convolution-bwd-data-algo-winograd-nonfused 5)
  (:cudnn-convolution-bwd-data-algo-count 6))

(cffi:defctype cudnnconvolutionbwddataalgo-t cudnnconvolutionbwddataalgo-t-enum)

(cffi:defcstruct cudnnconvolutionbwddataalgoperf-t-record)

(cffi:defctype cudnnconvolutionbwddataalgoperf-t (:struct cudnnconvolutionbwddataalgoperf-t-record))

(cffi:defcfun "cudnngetconvolutionbackwarddataalgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindconvolutionbackwarddataalgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (wdesc cudnnFilterDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (dxdesc cudnnTensorDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdDataAlgoPerf-t)))

(cffi:defcfun "cudnnfindconvolutionbackwarddataalgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void))
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdDataAlgoPerf-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnngetconvolutionbackwarddataalgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (wdesc cudnnFilterDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (dxdesc cudnnTensorDescriptor-t)
  (preference cudnnConvolutionBwdDataPreference-t)
  (memorylimitinbytes :int)
  (algo (:pointer cudnnConvolutionBwdDataAlgo-t)))

(cffi:defcfun ("cudnngetconvolutionbackwarddataalgorithm_v7" cudnngetconvolutionbackwarddataalgorithm-v7) cudnnStatus-t
  (handle cudnnHandle-t)
  (filterdesc cudnnFilterDescriptor-t)
  (diffdesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (graddesc cudnnTensorDescriptor-t)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnConvolutionBwdDataAlgoPerf-t)))

(cffi:defcfun "cudnngetconvolutionbackwarddataworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (wdesc cudnnFilterDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (dxdesc cudnnTensorDescriptor-t)
  (algo cudnnConvolutionBwdDataAlgo-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnconvolutionbackwarddata" cudnnStatus-t
  (handle cudnnHandle-t)
  (alpha (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (convdesc cudnnConvolutionDescriptor-t)
  (algo cudnnConvolutionBwdDataAlgo-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void)))

(cffi:defcfun "cudnnim2col" cudnnStatus-t
  (handle cudnnHandle-t)
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (convdesc cudnnConvolutionDescriptor-t)
  (colbuffer (:pointer :void)))

(cffi:defcenum cudnnsoftmaxalgorithm-t-enum
  (:cudnn-softmax-fast 0)
  (:cudnn-softmax-accurate 1)
  (:cudnn-softmax-log 2))

(cffi:defctype cudnnsoftmaxalgorithm-t cudnnsoftmaxalgorithm-t-enum)

(cffi:defcenum cudnnsoftmaxmode-t-enum
  (:cudnn-softmax-mode-instance 0)
  (:cudnn-softmax-mode-channel 1))

(cffi:defctype cudnnsoftmaxmode-t cudnnsoftmaxmode-t-enum)

(cffi:defcfun "cudnnsoftmaxforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (algo cudnnSoftmaxAlgorithm-t)
  (mode cudnnSoftmaxMode-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnsoftmaxbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (algo cudnnSoftmaxAlgorithm-t)
  (mode cudnnSoftmaxMode-t)
  (alpha (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void)))

(cffi:defcenum cudnnpoolingmode-t-enum
  (:cudnn-pooling-max 0)
  (:cudnn-pooling-average-count-include-padding 1)
  (:cudnn-pooling-average-count-exclude-padding 2)
  (:cudnn-pooling-max-deterministic 3))

(cffi:defctype cudnnpoolingmode-t cudnnpoolingmode-t-enum)

(cffi:defcfun "cudnncreatepoolingdescriptor" cudnnStatus-t
  (poolingdesc (:pointer cudnnPoolingDescriptor-t)))

(cffi:defcfun "cudnnsetpooling2ddescriptor" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (mode cudnnPoolingMode-t)
  (maxpoolingnanopt cudnnNanPropagation-t)
  (windowheight :int)
  (windowwidth :int)
  (verticalpadding :int)
  (horizontalpadding :int)
  (verticalstride :int)
  (horizontalstride :int))

(cffi:defcfun "cudnngetpooling2ddescriptor" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (mode (:pointer cudnnPoolingMode-t))
  (maxpoolingnanopt (:pointer cudnnNanPropagation-t))
  (windowheight (:pointer :int))
  (windowwidth (:pointer :int))
  (verticalpadding (:pointer :int))
  (horizontalpadding (:pointer :int))
  (verticalstride (:pointer :int))
  (horizontalstride (:pointer :int)))

(cffi:defcfun "cudnnsetpoolingnddescriptor" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (mode cudnnPoolingMode-t)
  (maxpoolingnanopt cudnnNanPropagation-t)
  (nbdims :int)
  (windowdima (:pointer :int) ; array 
)
  (paddinga (:pointer :int) ; array 
)
  (stridea (:pointer :int) ; array 
))

(cffi:defcfun "cudnngetpoolingnddescriptor" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (nbdimsrequested :int)
  (mode (:pointer cudnnPoolingMode-t))
  (maxpoolingnanopt (:pointer cudnnNanPropagation-t))
  (nbdims (:pointer :int))
  (windowdima (:pointer :int) ; array 
)
  (paddinga (:pointer :int) ; array 
)
  (stridea (:pointer :int) ; array 
))

(cffi:defcfun "cudnngetpoolingndforwardoutputdim" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (inputtensordesc cudnnTensorDescriptor-t)
  (nbdims :int)
  (outputtensordima (:pointer :int) ; array 
))

(cffi:defcfun "cudnngetpooling2dforwardoutputdim" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t)
  (inputtensordesc cudnnTensorDescriptor-t)
  (n (:pointer :int))
  (c (:pointer :int))
  (h (:pointer :int))
  (w (:pointer :int)))

(cffi:defcfun "cudnndestroypoolingdescriptor" cudnnStatus-t
  (poolingdesc cudnnPoolingDescriptor-t))

(cffi:defcfun "cudnnpoolingforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (poolingdesc cudnnPoolingDescriptor-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnpoolingbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (poolingdesc cudnnPoolingDescriptor-t)
  (alpha (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void)))

(cffi:defcenum cudnnactivationmode-t-enum
  (:cudnn-activation-sigmoid 0)
  (:cudnn-activation-relu 1)
  (:cudnn-activation-tanh 2)
  (:cudnn-activation-clipped-relu 3)
  (:cudnn-activation-elu 4)
  (:cudnn-activation-identity 5))

(cffi:defctype cudnnactivationmode-t cudnnactivationmode-t-enum)

(cffi:defcfun "cudnncreateactivationdescriptor" cudnnStatus-t
  (activationdesc (:pointer cudnnActivationDescriptor-t)))

(cffi:defcfun "cudnnsetactivationdescriptor" cudnnStatus-t
  (activationdesc cudnnActivationDescriptor-t)
  (mode cudnnActivationMode-t)
  (relunanopt cudnnNanPropagation-t)
  (coef :double))

(cffi:defcfun "cudnngetactivationdescriptor" cudnnStatus-t
  (activationdesc cudnnActivationDescriptor-t)
  (mode (:pointer cudnnActivationMode-t))
  (relunanopt (:pointer cudnnNanPropagation-t))
  (coef (:pointer :double)))

(cffi:defcfun "cudnndestroyactivationdescriptor" cudnnStatus-t
  (activationdesc cudnnActivationDescriptor-t))

(cffi:defcfun "cudnnactivationforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (activationdesc cudnnActivationDescriptor-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnactivationbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (activationdesc cudnnActivationDescriptor-t)
  (alpha (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void)))

(cffi:defcfun "cudnncreatelrndescriptor" cudnnStatus-t
  (normdesc (:pointer cudnnLRNDescriptor-t)))

(cffi:defcenum cudnnlrnmode-t-enum
  (:cudnn-lrn-cross-channel-dim1 0))

(cffi:defctype cudnnlrnmode-t cudnnlrnmode-t-enum)

(cffi:defcfun "cudnnsetlrndescriptor" cudnnStatus-t
  (normdesc cudnnLRNDescriptor-t)
  (lrnn :unsigned-int)
  (lrnalpha :double)
  (lrnbeta :double)
  (lrnk :double))

(cffi:defcfun "cudnngetlrndescriptor" cudnnStatus-t
  (normdesc cudnnLRNDescriptor-t)
  (lrnn (:pointer :unsigned-int))
  (lrnalpha (:pointer :double))
  (lrnbeta (:pointer :double))
  (lrnk (:pointer :double)))

(cffi:defcfun "cudnndestroylrndescriptor" cudnnStatus-t
  (lrndesc cudnnLRNDescriptor-t))

(cffi:defcfun "cudnnlrncrosschannelforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (normdesc cudnnLRNDescriptor-t)
  (lrnmode cudnnLRNMode-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnlrncrosschannelbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (normdesc cudnnLRNDescriptor-t)
  (lrnmode cudnnLRNMode-t)
  (alpha (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void)))

(cffi:defcenum cudnndivnormmode-t-enum
  (:cudnn-divnorm-precomputed-means 0))

(cffi:defctype cudnndivnormmode-t cudnndivnormmode-t-enum)

(cffi:defcfun "cudnndivisivenormalizationforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (normdesc cudnnLRNDescriptor-t)
  (mode cudnnDivNormMode-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (means (:pointer :void))
  (temp (:pointer :void))
  (temp2 (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnndivisivenormalizationbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (normdesc cudnnLRNDescriptor-t)
  (mode cudnnDivNormMode-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (means (:pointer :void))
  (dy (:pointer :void))
  (temp (:pointer :void))
  (temp2 (:pointer :void))
  (beta (:pointer :void))
  (dxdmeansdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void))
  (dmeans (:pointer :void)))

(cffi:defcenum cudnnbatchnormmode-t-enum
  (:cudnn-batchnorm-per-activation 0)
  (:cudnn-batchnorm-spatial 1)
  (:cudnn-batchnorm-spatial-persistent 2))

(cffi:defctype cudnnbatchnormmode-t cudnnbatchnormmode-t-enum)

(cffi:defcfun "cudnnderivebntensordescriptor" cudnnStatus-t
  (derivedbndesc cudnnTensorDescriptor-t)
  (xdesc cudnnTensorDescriptor-t)
  (mode cudnnBatchNormMode-t))

(cffi:defcenum cudnnbatchnormops-t-enum
  (:cudnn-batchnorm-ops-bn 0)
  (:cudnn-batchnorm-ops-bn-activation 1)
  (:cudnn-batchnorm-ops-bn-add-activation 2))

(cffi:defctype cudnnbatchnormops-t cudnnbatchnormops-t-enum)

(cffi:defcfun "cudnngetbatchnormalizationforwardtrainingexworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (bnops cudnnBatchNormOps-t)
  (xdesc cudnnTensorDescriptor-t)
  (zdesc cudnnTensorDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (bnscalebiasmeanvardesc cudnnTensorDescriptor-t)
  (activationdesc cudnnActivationDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnngetbatchnormalizationbackwardexworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (bnops cudnnBatchNormOps-t)
  (xdesc cudnnTensorDescriptor-t)
  (ydesc cudnnTensorDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (dzdesc cudnnTensorDescriptor-t)
  (dxdesc cudnnTensorDescriptor-t)
  (dbnscalebiasdesc cudnnTensorDescriptor-t)
  (activationdesc cudnnActivationDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnngetbatchnormalizationtrainingexreservespacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (bnops cudnnBatchNormOps-t)
  (activationdesc cudnnActivationDescriptor-t)
  (xdesc cudnnTensorDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnbatchnormalizationforwardtraining" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (alpha (:pointer :void))
  (beta (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (bnscalebiasmeanvardesc cudnnTensorDescriptor-t)
  (bnscale (:pointer :void))
  (bnbias (:pointer :void))
  (exponentialaveragefactor :double)
  (resultrunningmean (:pointer :void))
  (resultrunningvariance (:pointer :void))
  (epsilon :double)
  (resultsavemean (:pointer :void))
  (resultsaveinvvariance (:pointer :void)))

(cffi:defcfun "cudnnbatchnormalizationforwardtrainingex" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (bnops cudnnBatchNormOps-t)
  (alpha (:pointer :void))
  (beta (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (xdata (:pointer :void))
  (zdesc cudnnTensorDescriptor-t)
  (zdata (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (ydata (:pointer :void))
  (bnscalebiasmeanvardesc cudnnTensorDescriptor-t)
  (bnscale (:pointer :void))
  (bnbias (:pointer :void))
  (exponentialaveragefactor :double)
  (resultrunningmean (:pointer :void))
  (resultrunningvariance (:pointer :void))
  (epsilon :double)
  (resultsavemean (:pointer :void))
  (resultsaveinvvariance (:pointer :void))
  (activationdesc cudnnActivationDescriptor-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnbatchnormalizationforwardinference" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (alpha (:pointer :void))
  (beta (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (bnscalebiasmeanvardesc cudnnTensorDescriptor-t)
  (bnscale (:pointer :void))
  (bnbias (:pointer :void))
  (estimatedmean (:pointer :void))
  (estimatedvariance (:pointer :void))
  (epsilon :double))

(cffi:defcfun "cudnnbatchnormalizationbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (alphadatadiff (:pointer :void))
  (betadatadiff (:pointer :void))
  (alphaparamdiff (:pointer :void))
  (betaparamdiff (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void))
  (dbnscalebiasdesc cudnnTensorDescriptor-t)
  (bnscale (:pointer :void))
  (dbnscaleresult (:pointer :void))
  (dbnbiasresult (:pointer :void))
  (epsilon :double)
  (savedmean (:pointer :void))
  (savedinvvariance (:pointer :void)))

(cffi:defcfun "cudnnbatchnormalizationbackwardex" cudnnStatus-t
  (handle cudnnHandle-t)
  (mode cudnnBatchNormMode-t)
  (bnops cudnnBatchNormOps-t)
  (alphadatadiff (:pointer :void))
  (betadatadiff (:pointer :void))
  (alphaparamdiff (:pointer :void))
  (betaparamdiff (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (xdata (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (ydata (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dydata (:pointer :void))
  (dzdesc cudnnTensorDescriptor-t)
  (dzdata (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dxdata (:pointer :void))
  (dbnscalebiasdesc cudnnTensorDescriptor-t)
  (bnscaledata (:pointer :void))
  (bnbiasdata (:pointer :void))
  (dbnscaledata (:pointer :void))
  (dbnbiasdata (:pointer :void))
  (epsilon :double)
  (savedmean (:pointer :void))
  (savedinvvariance (:pointer :void))
  (activationdesc cudnnActivationDescriptor-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcenum cudnnsamplertype-t-enum
  (:cudnn-sampler-bilinear 0))

(cffi:defctype cudnnsamplertype-t cudnnsamplertype-t-enum)

(cffi:defcfun "cudnncreatespatialtransformerdescriptor" cudnnStatus-t
  (stdesc (:pointer cudnnSpatialTransformerDescriptor-t)))

(cffi:defcfun "cudnnsetspatialtransformernddescriptor" cudnnStatus-t
  (stdesc cudnnSpatialTransformerDescriptor-t)
  (samplertype cudnnSamplerType-t)
  (datatype cudnnDataType-t)
  (nbdims :int)
  (dima (:pointer :int) ; array 
))

(cffi:defcfun "cudnndestroyspatialtransformerdescriptor" cudnnStatus-t
  (stdesc cudnnSpatialTransformerDescriptor-t))

(cffi:defcfun "cudnnspatialtfgridgeneratorforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (stdesc cudnnSpatialTransformerDescriptor-t)
  (theta (:pointer :void))
  (grid (:pointer :void)))

(cffi:defcfun "cudnnspatialtfgridgeneratorbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (stdesc cudnnSpatialTransformerDescriptor-t)
  (dgrid (:pointer :void))
  (dtheta (:pointer :void)))

(cffi:defcfun "cudnnspatialtfsamplerforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (stdesc cudnnSpatialTransformerDescriptor-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (grid (:pointer :void))
  (beta (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void)))

(cffi:defcfun "cudnnspatialtfsamplerbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (stdesc cudnnSpatialTransformerDescriptor-t)
  (alpha (:pointer :void))
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (beta (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void))
  (alphadgrid (:pointer :void))
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (grid (:pointer :void))
  (betadgrid (:pointer :void))
  (dgrid (:pointer :void)))

(cffi:defcstruct cudnndropoutstruct)

(cffi:defctype cudnndropoutdescriptor-t (:pointer (:struct cudnnDropoutStruct)))

(cffi:defcfun "cudnncreatedropoutdescriptor" cudnnStatus-t
  (dropoutdesc (:pointer cudnnDropoutDescriptor-t)))

(cffi:defcfun "cudnndestroydropoutdescriptor" cudnnStatus-t
  (dropoutdesc cudnnDropoutDescriptor-t))

(cffi:defcfun "cudnndropoutgetstatessize" cudnnStatus-t
  (handle cudnnHandle-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnndropoutgetreservespacesize" cudnnStatus-t
  (xdesc cudnnTensorDescriptor-t)
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnsetdropoutdescriptor" cudnnStatus-t
  (dropoutdesc cudnnDropoutDescriptor-t)
  (handle cudnnHandle-t)
  (dropout :float)
  (states (:pointer :void))
  (statesizeinbytes :int)
  (seed :unsigned-long-long))

(cffi:defcfun "cudnnrestoredropoutdescriptor" cudnnStatus-t
  (dropoutdesc cudnnDropoutDescriptor-t)
  (handle cudnnHandle-t)
  (dropout :float)
  (states (:pointer :void))
  (statesizeinbytes :int)
  (seed :unsigned-long-long))

(cffi:defcfun "cudnngetdropoutdescriptor" cudnnStatus-t
  (dropoutdesc cudnnDropoutDescriptor-t)
  (handle cudnnHandle-t)
  (dropout (:pointer :float))
  (states (:pointer (:pointer :void)))
  (seed (:pointer :unsigned-long-long)))

(cffi:defcfun "cudnndropoutforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (dropoutdesc cudnnDropoutDescriptor-t)
  (xdesc cudnnTensorDescriptor-t)
  (x (:pointer :void))
  (ydesc cudnnTensorDescriptor-t)
  (y (:pointer :void))
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnndropoutbackward" cudnnStatus-t
  (handle cudnnHandle-t)
  (dropoutdesc cudnnDropoutDescriptor-t)
  (dydesc cudnnTensorDescriptor-t)
  (dy (:pointer :void))
  (dxdesc cudnnTensorDescriptor-t)
  (dx (:pointer :void))
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcenum cudnnrnnalgo-t-enum
  (:cudnn-rnn-algo-standard 0)
  (:cudnn-rnn-algo-persist-static 1)
  (:cudnn-rnn-algo-persist-dynamic 2)
  (:cudnn-rnn-algo-count 3))

(cffi:defctype cudnnrnnalgo-t cudnnrnnalgo-t-enum)

(cffi:defcenum cudnnrnnmode-t-enum
  (:cudnn-rnn-relu 0)
  (:cudnn-rnn-tanh 1)
  (:cudnn-lstm 2)
  (:cudnn-gru 3))

(cffi:defctype cudnnrnnmode-t cudnnrnnmode-t-enum)

(cffi:defcenum cudnnrnnbiasmode-t-enum
  (:cudnn-rnn-no-bias 0)
  (:cudnn-rnn-single-inp-bias 1)
  (:cudnn-rnn-double-bias 2)
  (:cudnn-rnn-single-rec-bias 3))

(cffi:defctype cudnnrnnbiasmode-t cudnnrnnbiasmode-t-enum)

(cffi:defcenum cudnndirectionmode-t-enum
  (:cudnn-unidirectional 0)
  (:cudnn-bidirectional 1))

(cffi:defctype cudnndirectionmode-t cudnndirectionmode-t-enum)

(cffi:defcenum cudnnrnninputmode-t-enum
  (:cudnn-linear-input 0)
  (:cudnn-skip-input 1))

(cffi:defctype cudnnrnninputmode-t cudnnrnninputmode-t-enum)

(cffi:defcenum cudnnrnnclipmode-t-enum
  (:cudnn-rnn-clip-none 0)
  (:cudnn-rnn-clip-minmax 1))

(cffi:defctype cudnnrnnclipmode-t cudnnrnnclipmode-t-enum)

(cffi:defcenum cudnnrnndatalayout-t-enum
  (:cudnn-rnn-data-layout-seq-major-unpacked 0)
  (:cudnn-rnn-data-layout-seq-major-packed 1)
  (:cudnn-rnn-data-layout-batch-major-unpacked 2))

(cffi:defctype cudnnrnndatalayout-t cudnnrnndatalayout-t-enum)

(cffi:defcenum cudnnrnnpaddingmode-t-enum
  (:cudnn-rnn-padded-io-disabled 0)
  (:cudnn-rnn-padded-io-enabled 1))

(cffi:defctype cudnnrnnpaddingmode-t cudnnrnnpaddingmode-t-enum)

(cffi:defcstruct cudnnrnnstruct)

(cffi:defctype cudnnrnndescriptor-t (:pointer (:struct cudnnRNNStruct)))

(cffi:defcstruct cudnnpersistentrnnplan)

(cffi:defctype cudnnpersistentrnnplan-t (:pointer (:struct cudnnPersistentRNNPlan)))

(cffi:defcstruct cudnnrnndatastruct)

(cffi:defctype cudnnrnndatadescriptor-t (:pointer (:struct cudnnRNNDataStruct)))

(cffi:defcfun "cudnncreaternndescriptor" cudnnStatus-t
  (rnndesc (:pointer cudnnRNNDescriptor-t)))

(cffi:defcfun "cudnndestroyrnndescriptor" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t))

(cffi:defcfun "cudnnsetrnndescriptor" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (hiddensize :int)
  (numlayers :int)
  (dropoutdesc cudnnDropoutDescriptor-t)
  (inputmode cudnnRNNInputMode-t)
  (direction cudnnDirectionMode-t)
  (mode cudnnRNNMode-t)
  (algo cudnnRNNAlgo-t)
  (mathprec cudnnDataType-t))

(cffi:defcfun "cudnngetrnndescriptor" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (hiddensize (:pointer :int))
  (numlayers (:pointer :int))
  (dropoutdesc (:pointer cudnnDropoutDescriptor-t))
  (inputmode (:pointer cudnnRNNInputMode-t))
  (direction (:pointer cudnnDirectionMode-t))
  (mode (:pointer cudnnRNNMode-t))
  (algo (:pointer cudnnRNNAlgo-t))
  (mathprec (:pointer cudnnDataType-t)))

(cffi:defcfun "cudnnsetrnnmatrixmathtype" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (mtype cudnnMathType-t))

(cffi:defcfun "cudnngetrnnmatrixmathtype" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (mtype (:pointer cudnnMathType-t)))

(cffi:defcfun "cudnnsetrnnbiasmode" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (biasmode cudnnRNNBiasMode-t))

(cffi:defcfun "cudnngetrnnbiasmode" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (biasmode (:pointer cudnnRNNBiasMode-t)))

(cffi:defcfun "cudnnrnnsetclip" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (clipmode cudnnRNNClipMode-t)
  (clipnanopt cudnnNanPropagation-t)
  (lclip :double)
  (rclip :double))

(cffi:defcfun "cudnnrnngetclip" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (clipmode (:pointer cudnnRNNClipMode-t))
  (clipnanopt (:pointer cudnnNanPropagation-t))
  (lclip (:pointer :double))
  (rclip (:pointer :double)))

(cffi:defcfun "cudnnsetrnnprojectionlayers" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (recprojsize :int)
  (outprojsize :int))

(cffi:defcfun "cudnngetrnnprojectionlayers" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (recprojsize (:pointer :int))
  (outprojsize (:pointer :int)))

(cffi:defcfun "cudnncreatepersistentrnnplan" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (minibatch :int)
  (datatype cudnnDataType-t)
  (plan (:pointer cudnnPersistentRNNPlan-t)))

(cffi:defcfun "cudnndestroypersistentrnnplan" cudnnStatus-t
  (plan cudnnPersistentRNNPlan-t))

(cffi:defcfun "cudnnsetpersistentrnnplan" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (plan cudnnPersistentRNNPlan-t))

(cffi:defcfun "cudnngetrnnworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnngetrnntrainingreservesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (sizeinbytes (:pointer :int)))

(cffi:defcfun "cudnngetrnnparamssize" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (xdesc cudnnTensorDescriptor-t)
  (sizeinbytes (:pointer :int))
  (datatype cudnnDataType-t))

(cffi:defcfun "cudnngetrnnlinlayermatrixparams" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (pseudolayer :int)
  (xdesc cudnnTensorDescriptor-t)
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (linlayerid :int)
  (linlayermatdesc cudnnFilterDescriptor-t)
  (linlayermat (:pointer (:pointer :void))))

(cffi:defcfun "cudnngetrnnlinlayerbiasparams" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (pseudolayer :int)
  (xdesc cudnnTensorDescriptor-t)
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (linlayerid :int)
  (linlayerbiasdesc cudnnFilterDescriptor-t)
  (linlayerbias (:pointer (:pointer :void))))

(cffi:defcfun "cudnnrnnforwardinference" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnnrnnforwardtraining" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnrnnbackwarddata" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (dydesc (:pointer cudnnTensorDescriptor-t))
  (dy (:pointer :void))
  (dhydesc cudnnTensorDescriptor-t)
  (dhy (:pointer :void))
  (dcydesc cudnnTensorDescriptor-t)
  (dcy (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (dxdesc (:pointer cudnnTensorDescriptor-t))
  (dx (:pointer :void))
  (dhxdesc cudnnTensorDescriptor-t)
  (dhx (:pointer :void))
  (dcxdesc cudnnTensorDescriptor-t)
  (dcx (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnrnnbackwardweights" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (dwdesc cudnnFilterDescriptor-t)
  (dw (:pointer :void))
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnsetrnnpaddingmode" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (paddingmode cudnnRNNPaddingMode-t))

(cffi:defcfun "cudnngetrnnpaddingmode" cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (paddingmode (:pointer cudnnRNNPaddingMode-t)))

(cffi:defcfun "cudnncreaternndatadescriptor" cudnnStatus-t
  (rnndatadesc (:pointer cudnnRNNDataDescriptor-t)))

(cffi:defcfun "cudnndestroyrnndatadescriptor" cudnnStatus-t
  (rnndatadesc cudnnRNNDataDescriptor-t))

(cffi:defcfun "cudnnsetrnndatadescriptor" cudnnStatus-t
  (rnndatadesc cudnnRNNDataDescriptor-t)
  (datatype cudnnDataType-t)
  (layout cudnnRNNDataLayout-t)
  (maxseqlength :int)
  (batchsize :int)
  (vectorsize :int)
  (seqlengtharray (:pointer :int) ; array 
)
  (paddingfill (:pointer :void)))

(cffi:defcfun "cudnngetrnndatadescriptor" cudnnStatus-t
  (rnndatadesc cudnnRNNDataDescriptor-t)
  (datatype (:pointer cudnnDataType-t))
  (layout (:pointer cudnnRNNDataLayout-t))
  (maxseqlength (:pointer :int))
  (batchsize (:pointer :int))
  (vectorsize (:pointer :int))
  (arraylengthrequested :int)
  (seqlengtharray (:pointer :int) ; array 
)
  (paddingfill (:pointer :void)))

(cffi:defcfun "cudnnrnnforwardtrainingex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (xdesc cudnnRNNDataDescriptor-t)
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc cudnnRNNDataDescriptor-t)
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (kdesc cudnnRNNDataDescriptor-t)
  (keys (:pointer :void))
  (cdesc cudnnRNNDataDescriptor-t)
  (cattn (:pointer :void))
  (idesc cudnnRNNDataDescriptor-t)
  (iattn (:pointer :void))
  (qdesc cudnnRNNDataDescriptor-t)
  (queries (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnrnnforwardinferenceex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (xdesc cudnnRNNDataDescriptor-t)
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc cudnnRNNDataDescriptor-t)
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (kdesc cudnnRNNDataDescriptor-t)
  (keys (:pointer :void))
  (cdesc cudnnRNNDataDescriptor-t)
  (cattn (:pointer :void))
  (idesc cudnnRNNDataDescriptor-t)
  (iattn (:pointer :void))
  (qdesc cudnnRNNDataDescriptor-t)
  (queries (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnnrnnbackwarddataex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (ydesc cudnnRNNDataDescriptor-t)
  (y (:pointer :void))
  (dydesc cudnnRNNDataDescriptor-t)
  (dy (:pointer :void))
  (dcdesc cudnnRNNDataDescriptor-t)
  (dcattn (:pointer :void))
  (dhydesc cudnnTensorDescriptor-t)
  (dhy (:pointer :void))
  (dcydesc cudnnTensorDescriptor-t)
  (dcy (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (dxdesc cudnnRNNDataDescriptor-t)
  (dx (:pointer :void))
  (dhxdesc cudnnTensorDescriptor-t)
  (dhx (:pointer :void))
  (dcxdesc cudnnTensorDescriptor-t)
  (dcx (:pointer :void))
  (dkdesc cudnnRNNDataDescriptor-t)
  (dkeys (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnnrnnbackwardweightsex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (xdesc cudnnRNNDataDescriptor-t)
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (ydesc cudnnRNNDataDescriptor-t)
  (y (:pointer :void))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (dwdesc cudnnFilterDescriptor-t)
  (dw (:pointer :void))
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcstruct cudnnalgorithmstruct)

(cffi:defctype cudnnalgorithmdescriptor-t (:pointer (:struct cudnnAlgorithmStruct)))

(cffi:defcstruct cudnnalgorithmperformancestruct)

(cffi:defctype cudnnalgorithmperformance-t (:pointer (:struct cudnnAlgorithmPerformanceStruct)))

(cffi:defcfun "cudnnsetrnnalgorithmdescriptor" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (algodesc cudnnAlgorithmDescriptor-t))

(cffi:defcfun "cudnngetrnnforwardinferencealgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindrnnforwardinferencealgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (findintensity :float)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnAlgorithmPerformance-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnngetrnnforwardtrainingalgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindrnnforwardtrainingalgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (hydesc cudnnTensorDescriptor-t)
  (hy (:pointer :void))
  (cydesc cudnnTensorDescriptor-t)
  (cy (:pointer :void))
  (findintensity :float)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnAlgorithmPerformance-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnngetrnnbackwarddataalgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindrnnbackwarddataalgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (dydesc (:pointer cudnnTensorDescriptor-t))
  (dy (:pointer :void))
  (dhydesc cudnnTensorDescriptor-t)
  (dhy (:pointer :void))
  (dcydesc cudnnTensorDescriptor-t)
  (dcy (:pointer :void))
  (wdesc cudnnFilterDescriptor-t)
  (w (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (cxdesc cudnnTensorDescriptor-t)
  (cx (:pointer :void))
  (dxdesc (:pointer cudnnTensorDescriptor-t))
  (dx (:pointer :void))
  (dhxdesc cudnnTensorDescriptor-t)
  (dhx (:pointer :void))
  (dcxdesc cudnnTensorDescriptor-t)
  (dcx (:pointer :void))
  (findintensity :float)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnAlgorithmPerformance-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcfun "cudnngetrnnbackwardweightsalgorithmmaxcount" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (count (:pointer :int)))

(cffi:defcfun "cudnnfindrnnbackwardweightsalgorithmex" cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (seqlength :int)
  (xdesc (:pointer cudnnTensorDescriptor-t))
  (x (:pointer :void))
  (hxdesc cudnnTensorDescriptor-t)
  (hx (:pointer :void))
  (ydesc (:pointer cudnnTensorDescriptor-t))
  (y (:pointer :void))
  (findintensity :float)
  (requestedalgocount :int)
  (returnedalgocount (:pointer :int))
  (perfresults (:pointer cudnnAlgorithmPerformance-t))
  (workspace (:pointer :void))
  (workspacesizeinbytes :int)
  (dwdesc cudnnFilterDescriptor-t)
  (dw (:pointer :void))
  (reservespace (:pointer :void))
  (reservespacesizeinbytes :int))

(cffi:defcenum cudnnseqdataaxis-t-enum
  (:cudnn-seqdata-time-dim 0)
  (:cudnn-seqdata-batch-dim 1)
  (:cudnn-seqdata-beam-dim 2)
  (:cudnn-seqdata-vect-dim 3))

(cffi:defctype cudnnseqdataaxis-t cudnnseqdataaxis-t-enum)

(cffi:defcstruct cudnnseqdatastruct)

(cffi:defctype cudnnseqdatadescriptor-t (:pointer (:struct cudnnSeqDataStruct)))

(cffi:defcfun "cudnncreateseqdatadescriptor" cudnnStatus-t
  (seqdatadesc (:pointer cudnnSeqDataDescriptor-t)))

(cffi:defcfun "cudnndestroyseqdatadescriptor" cudnnStatus-t
  (seqdatadesc cudnnSeqDataDescriptor-t))

(cffi:defcfun "cudnnsetseqdatadescriptor" cudnnStatus-t
  (seqdatadesc cudnnSeqDataDescriptor-t)
  (datatype cudnnDataType-t)
  (nbdims :int)
  (dima (:pointer :int) ; array 
)
  (axes (:pointer cudnnSeqDataAxis-t) ; array 
)
  (seqlengtharraysize :int)
  (seqlengtharray (:pointer :int) ; array 
)
  (paddingfill (:pointer :void)))

(cffi:defcfun "cudnngetseqdatadescriptor" cudnnStatus-t
  (seqdatadesc cudnnSeqDataDescriptor-t)
  (datatype (:pointer cudnnDataType-t))
  (nbdims (:pointer :int))
  (nbdimsrequested :int)
  (dima (:pointer :int) ; array 
)
  (axes (:pointer cudnnSeqDataAxis-t) ; array 
)
  (seqlengtharraysize (:pointer :int))
  (seqlengthsizerequested :int)
  (seqlengtharray (:pointer :int) ; array 
)
  (paddingfill (:pointer :void)))

(cffi:defctype cudnnattnquerymap-t :unsigned-int)

(cffi:defcstruct cudnnattnstruct)

(cffi:defctype cudnnattndescriptor-t (:pointer (:struct cudnnAttnStruct)))

(cffi:defcfun "cudnncreateattndescriptor" cudnnStatus-t
  (attndesc (:pointer cudnnAttnDescriptor-t)))

(cffi:defcfun "cudnndestroyattndescriptor" cudnnStatus-t
  (attndesc cudnnAttnDescriptor-t))

(cffi:defcfun "cudnnsetattndescriptor" cudnnStatus-t
  (attndesc cudnnAttnDescriptor-t)
  (attnmode :unsigned-int)
  (nheads :int)
  (smscaler :double)
  (datatype cudnnDataType-t)
  (computeprec cudnnDataType-t)
  (mathtype cudnnMathType-t)
  (attndropoutdesc cudnnDropoutDescriptor-t)
  (postdropoutdesc cudnnDropoutDescriptor-t)
  (qsize :int)
  (ksize :int)
  (vsize :int)
  (qprojsize :int)
  (kprojsize :int)
  (vprojsize :int)
  (oprojsize :int)
  (qomaxseqlength :int)
  (kvmaxseqlength :int)
  (maxbatchsize :int)
  (maxbeamsize :int))

(cffi:defcfun "cudnngetattndescriptor" cudnnStatus-t
  (attndesc cudnnAttnDescriptor-t)
  (attnmode (:pointer :unsigned-int))
  (nheads (:pointer :int))
  (smscaler (:pointer :double))
  (datatype (:pointer cudnnDataType-t))
  (computeprec (:pointer cudnnDataType-t))
  (mathtype (:pointer cudnnMathType-t))
  (attndropoutdesc (:pointer cudnnDropoutDescriptor-t))
  (postdropoutdesc (:pointer cudnnDropoutDescriptor-t))
  (qsize (:pointer :int))
  (ksize (:pointer :int))
  (vsize (:pointer :int))
  (qprojsize (:pointer :int))
  (kprojsize (:pointer :int))
  (vprojsize (:pointer :int))
  (oprojsize (:pointer :int))
  (qomaxseqlength (:pointer :int))
  (kvmaxseqlength (:pointer :int))
  (maxbatchsize (:pointer :int))
  (maxbeamsize (:pointer :int)))

(cffi:defcfun "cudnngetmultiheadattnbuffers" cudnnStatus-t
  (handle cudnnHandle-t)
  (attndesc cudnnAttnDescriptor-t)
  (weightsizeinbytes (:pointer :int))
  (workspacesizeinbytes (:pointer :int))
  (reservespacesizeinbytes (:pointer :int)))

(cffi:defcenum cudnnmultiheadattnweightkind-t-enum
  (:cudnn-mh-attn-q-weights 0)
  (:cudnn-mh-attn-k-weights 1)
  (:cudnn-mh-attn-v-weights 2)
  (:cudnn-mh-attn-o-weights 3)
  (:cudnn-mh-attn-q-biases 4)
  (:cudnn-mh-attn-k-biases 5)
  (:cudnn-mh-attn-v-biases 6)
  (:cudnn-mh-attn-o-biases 7))

(cffi:defctype cudnnmultiheadattnweightkind-t cudnnmultiheadattnweightkind-t-enum)

(cffi:defcfun "cudnngetmultiheadattnweights" cudnnStatus-t
  (handle cudnnHandle-t)
  (attndesc cudnnAttnDescriptor-t)
  (wkind cudnnMultiHeadAttnWeightKind-t)
  (weightsizeinbytes :int)
  (weights (:pointer :void))
  (wdesc cudnnTensorDescriptor-t)
  (waddr (:pointer (:pointer :void))))

(cffi:defcfun "cudnnmultiheadattnforward" cudnnStatus-t
  (handle cudnnHandle-t)
  (attndesc cudnnAttnDescriptor-t)
  (curridx :int)
  (lowinidx (:pointer :int) ; array 
)
  (hiwinidx (:pointer :int) ; array 
)
  (devseqlengthsqo (:pointer :int) ; array 
)
  (devseqlengthskv (:pointer :int) ; array 
)
  (qdesc cudnnSeqDataDescriptor-t)
  (queries (:pointer :void))
  (residuals (:pointer :void))
  (kdesc cudnnSeqDataDescriptor-t)
  (keys (:pointer :void))
  (vdesc cudnnSeqDataDescriptor-t)
  (values (:pointer :void))
  (odesc cudnnSeqDataDescriptor-t)
  (out (:pointer :void))
  (weightsizeinbytes :int)
  (weights (:pointer :void))
  (workspacesizeinbytes :int)
  (workspace (:pointer :void))
  (reservespacesizeinbytes :int)
  (reservespace (:pointer :void)))

(cffi:defcfun "cudnnmultiheadattnbackwarddata" cudnnStatus-t
  (handle cudnnHandle-t)
  (attndesc cudnnAttnDescriptor-t)
  (lowinidx (:pointer :int) ; array 
)
  (hiwinidx (:pointer :int) ; array 
)
  (devseqlengthsdqdo (:pointer :int) ; array 
)
  (devseqlengthsdkdv (:pointer :int) ; array 
)
  (dodesc cudnnSeqDataDescriptor-t)
  (dout (:pointer :void))
  (dqdesc cudnnSeqDataDescriptor-t)
  (dqueries (:pointer :void))
  (queries (:pointer :void))
  (dkdesc cudnnSeqDataDescriptor-t)
  (dkeys (:pointer :void))
  (keys (:pointer :void))
  (dvdesc cudnnSeqDataDescriptor-t)
  (dvalues (:pointer :void))
  (values (:pointer :void))
  (weightsizeinbytes :int)
  (weights (:pointer :void))
  (workspacesizeinbytes :int)
  (workspace (:pointer :void))
  (reservespacesizeinbytes :int)
  (reservespace (:pointer :void)))

(cffi:defcenum cudnnwgradmode-t-enum
  (:cudnn-wgrad-mode-add 0)
  (:cudnn-wgrad-mode-set 1))

(cffi:defctype cudnnwgradmode-t cudnnwgradmode-t-enum)

(cffi:defcfun "cudnnmultiheadattnbackwardweights" cudnnStatus-t
  (handle cudnnHandle-t)
  (attndesc cudnnAttnDescriptor-t)
  (addgrad cudnnWgradMode-t)
  (qdesc cudnnSeqDataDescriptor-t)
  (queries (:pointer :void))
  (kdesc cudnnSeqDataDescriptor-t)
  (keys (:pointer :void))
  (vdesc cudnnSeqDataDescriptor-t)
  (values (:pointer :void))
  (dodesc cudnnSeqDataDescriptor-t)
  (dout (:pointer :void))
  (weightsizeinbytes :int)
  (weights (:pointer :void))
  (dweights (:pointer :void))
  (workspacesizeinbytes :int)
  (workspace (:pointer :void))
  (reservespacesizeinbytes :int)
  (reservespace (:pointer :void)))

(cffi:defcenum cudnnctclossalgo-t-enum
  (:cudnn-ctc-loss-algo-deterministic 0)
  (:cudnn-ctc-loss-algo-non-deterministic 1))

(cffi:defctype cudnnctclossalgo-t cudnnctclossalgo-t-enum)

(cffi:defcenum cudnnlossnormalizationmode-t-enum
  (:cudnn-loss-normalization-none 0)
  (:cudnn-loss-normalization-softmax 1))

(cffi:defctype cudnnlossnormalizationmode-t cudnnlossnormalizationmode-t-enum)

(cffi:defcfun "cudnncreatectclossdescriptor" cudnnStatus-t
  (ctclossdesc (:pointer cudnnCTCLossDescriptor-t)))

(cffi:defcfun "cudnnsetctclossdescriptor" cudnnStatus-t
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (comptype cudnnDataType-t))

(cffi:defcfun "cudnnsetctclossdescriptorex" cudnnStatus-t
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (comptype cudnnDataType-t)
  (normmode cudnnLossNormalizationMode-t)
  (gradmode cudnnNanPropagation-t))

(cffi:defcfun "cudnngetctclossdescriptor" cudnnStatus-t
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (comptype (:pointer cudnnDataType-t)))

(cffi:defcfun "cudnngetctclossdescriptorex" cudnnStatus-t
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (comptype (:pointer cudnnDataType-t))
  (normmode (:pointer cudnnLossNormalizationMode-t))
  (gradmode (:pointer cudnnNanPropagation-t)))

(cffi:defcfun "cudnndestroyctclossdescriptor" cudnnStatus-t
  (ctclossdesc cudnnCTCLossDescriptor-t))

(cffi:defcfun "cudnnctcloss" cudnnStatus-t
  (handle cudnnHandle-t)
  (probsdesc cudnnTensorDescriptor-t)
  (probs (:pointer :void))
  (labels (:pointer :int))
  (labellengths (:pointer :int))
  (inputlengths (:pointer :int))
  (costs (:pointer :void))
  (gradientsdesc cudnnTensorDescriptor-t)
  (gradients (:pointer :void))
  (algo cudnnCTCLossAlgo-t)
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (workspace (:pointer :void))
  (workspacesizeinbytes :int))

(cffi:defcfun "cudnngetctclossworkspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (probsdesc cudnnTensorDescriptor-t)
  (gradientsdesc cudnnTensorDescriptor-t)
  (labels (:pointer :int))
  (labellengths (:pointer :int))
  (inputlengths (:pointer :int))
  (algo cudnnCTCLossAlgo-t)
  (ctclossdesc cudnnCTCLossDescriptor-t)
  (sizeinbytes (:pointer :int)))

;(cffi:defcstruct cudnnalgorithm-t-record
  ;(algo (:union Algorithm)))

;(cffi:defctype cudnnalgorithm-t (:struct cudnnalgorithm-t-record))

(cffi:defcfun "cudnncreatealgorithmdescriptor" cudnnStatus-t
  (algodesc (:pointer cudnnAlgorithmDescriptor-t)))

;(cffi:defcfun "cudnnsetalgorithmdescriptor" cudnnStatus-t
  ;(algodesc cudnnAlgorithmDescriptor-t)
  ;(algorithm cudnnAlgorithm-t))

;(cffi:defcfun "cudnngetalgorithmdescriptor" cudnnStatus-t
  ;(algodesc cudnnAlgorithmDescriptor-t)
  ;(algorithm (:pointer cudnnAlgorithm-t)))

(cffi:defcfun "cudnncopyalgorithmdescriptor" cudnnStatus-t
  (src cudnnAlgorithmDescriptor-t)
  (dest cudnnAlgorithmDescriptor-t))

(cffi:defcfun "cudnndestroyalgorithmdescriptor" cudnnStatus-t
  (algodesc cudnnAlgorithmDescriptor-t))

(cffi:defcfun "cudnncreatealgorithmperformance" cudnnStatus-t
  (algoperf (:pointer cudnnAlgorithmPerformance-t))
  (numbertocreate :int))

(cffi:defcfun "cudnnsetalgorithmperformance" cudnnStatus-t
  (algoperf cudnnAlgorithmPerformance-t)
  (algodesc cudnnAlgorithmDescriptor-t)
  (status cudnnStatus-t)
  (time :float)
  (memory :int))

(cffi:defcfun "cudnngetalgorithmperformance" cudnnStatus-t
  (algoperf cudnnAlgorithmPerformance-t)
  (algodesc (:pointer cudnnAlgorithmDescriptor-t))
  (status (:pointer cudnnStatus-t))
  (time (:pointer :float))
  (memory (:pointer :int)))

(cffi:defcfun "cudnndestroyalgorithmperformance" cudnnStatus-t
  (algoperf (:pointer cudnnAlgorithmPerformance-t))
  (numbertodestroy :int))

(cffi:defcfun "cudnngetalgorithmspacesize" cudnnStatus-t
  (handle cudnnHandle-t)
  (algodesc cudnnAlgorithmDescriptor-t)
  (algospacesizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnsavealgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (algodesc cudnnAlgorithmDescriptor-t)
  (algospace (:pointer :void))
  (algospacesizeinbytes :int))

(cffi:defcfun "cudnnrestorealgorithm" cudnnStatus-t
  (handle cudnnHandle-t)
  (algospace (:pointer :void))
  (algospacesizeinbytes :int)
  (algodesc cudnnAlgorithmDescriptor-t))

(cffi:defcenum cudnnseverity-t-enum
  (:cudnn-sev-fatal 0)
  (:cudnn-sev-error 1)
  (:cudnn-sev-warning 2)
  (:cudnn-sev-info 3))

(cffi:defctype cudnnseverity-t cudnnseverity-t-enum)

(cffi:defcstruct cudnndebug-t-record
  (cudnn-version :unsigned-int)
  (cudnnstatus cudnnStatus-t)
  (time-sec :unsigned-int)
  (time-usec :unsigned-int)
  (time-delta :unsigned-int)
  (handle cudnnHandle-t)
  (stream (:pointer))
  (pid :unsigned-long-long)
  (tid :unsigned-long-long)
  (cudadeviceid :int)
  (reserved (:pointer :int)
))

(cffi:defctype cudnndebug-t (:struct cudnndebug-t-record))

(cffi:defctype cudnncallback-t (:pointer :pointer ; function ptr void (cudnnSeverity_t, void *, const cudnnDebug_t *, const char *)
))

(cffi:defcfun "cudnnsetcallback" cudnnStatus-t
  (mask :unsigned-int)
  (udata (:pointer :void))
  (fptr cudnnCallback-t))

(cffi:defcfun "cudnngetcallback" cudnnStatus-t
  (mask (:pointer :unsigned-int))
  (udata (:pointer (:pointer :void)))
  (fptr (:pointer cudnnCallback-t)))

(cffi:defcstruct cudnnfusedopsconstparamstruct)

(cffi:defctype cudnnfusedopsconstparampack-t (:pointer (:struct cudnnFusedOpsConstParamStruct)))

(cffi:defcstruct cudnnfusedopsvariantparamstruct)

(cffi:defctype cudnnfusedopsvariantparampack-t (:pointer (:struct cudnnFusedOpsVariantParamStruct)))

(cffi:defcstruct cudnnfusedopsplanstruct)

(cffi:defctype cudnnfusedopsplan-t (:pointer (:struct cudnnFusedOpsPlanStruct)))

(cffi:defcenum cudnnfusedops-t-enum
  (:cudnn-fused-scale-bias-activation-conv-bnstats 0)
  (:cudnn-fused-scale-bias-activation-wgrad 1)
  (:cudnn-fused-bn-finalize-statistics-training 2)
  (:cudnn-fused-bn-finalize-statistics-inference 3)
  (:cudnn-fused-conv-scale-bias-add-activation 4)
  (:cudnn-fused-scale-bias-add-activation-gen-bitmask 5)
  (:cudnn-fused-dactivation-fork-dbatchnorm 6))

(cffi:defctype cudnnfusedops-t cudnnfusedops-t-enum)

(cffi:defcenum cudnnfusedopsconstparamlabel-t-enum
  (:cudnn-param-xdesc 0)
  (:cudnn-param-xdata-placeholder 1)
  (:cudnn-param-bn-mode 2)
  (:cudnn-param-bn-eqscalebias-desc 3)
  (:cudnn-param-bn-eqscale-placeholder 4)
  (:cudnn-param-bn-eqbias-placeholder 5)
  (:cudnn-param-activation-desc 6)
  (:cudnn-param-conv-desc 7)
  (:cudnn-param-wdesc 8)
  (:cudnn-param-wdata-placeholder 9)
  (:cudnn-param-dwdesc 10)
  (:cudnn-param-dwdata-placeholder 11)
  (:cudnn-param-ydesc 12)
  (:cudnn-param-ydata-placeholder 13)
  (:cudnn-param-dydesc 14)
  (:cudnn-param-dydata-placeholder 15)
  (:cudnn-param-ystats-desc 16)
  (:cudnn-param-ysum-placeholder 17)
  (:cudnn-param-ysqsum-placeholder 18)
  (:cudnn-param-bn-scalebias-meanvar-desc 19)
  (:cudnn-param-bn-scale-placeholder 20)
  (:cudnn-param-bn-bias-placeholder 21)
  (:cudnn-param-bn-saved-mean-placeholder 22)
  (:cudnn-param-bn-saved-invstd-placeholder 23)
  (:cudnn-param-bn-running-mean-placeholder 24)
  (:cudnn-param-bn-running-var-placeholder 25)
  (:cudnn-param-zdesc 26)
  (:cudnn-param-zdata-placeholder 27)
  (:cudnn-param-bn-z-eqscalebias-desc 28)
  (:cudnn-param-bn-z-eqscale-placeholder 29)
  (:cudnn-param-bn-z-eqbias-placeholder 30)
  (:cudnn-param-activation-bitmask-desc 31)
  (:cudnn-param-activation-bitmask-placeholder 32)
  (:cudnn-param-dxdesc 33)
  (:cudnn-param-dxdata-placeholder 34)
  (:cudnn-param-dzdesc 35)
  (:cudnn-param-dzdata-placeholder 36)
  (:cudnn-param-bn-dscale-placeholder 37)
  (:cudnn-param-bn-dbias-placeholder 38))

(cffi:defctype cudnnfusedopsconstparamlabel-t cudnnfusedopsconstparamlabel-t-enum)

(cffi:defcenum cudnnfusedopspointerplaceholder-t-enum
  (:cudnn-ptr-null 0)
  (:cudnn-ptr-elem-aligned 1)
  (:cudnn-ptr-16b-aligned 2))

(cffi:defctype cudnnfusedopspointerplaceholder-t cudnnfusedopspointerplaceholder-t-enum)

(cffi:defcenum cudnnfusedopsvariantparamlabel-t-enum
  (:cudnn-ptr-xdata 0)
  (:cudnn-ptr-bn-eqscale 1)
  (:cudnn-ptr-bn-eqbias 2)
  (:cudnn-ptr-wdata 3)
  (:cudnn-ptr-dwdata 4)
  (:cudnn-ptr-ydata 5)
  (:cudnn-ptr-dydata 6)
  (:cudnn-ptr-ysum 7)
  (:cudnn-ptr-ysqsum 8)
  (:cudnn-ptr-workspace 9)
  (:cudnn-ptr-bn-scale 10)
  (:cudnn-ptr-bn-bias 11)
  (:cudnn-ptr-bn-saved-mean 12)
  (:cudnn-ptr-bn-saved-invstd 13)
  (:cudnn-ptr-bn-running-mean 14)
  (:cudnn-ptr-bn-running-var 15)
  (:cudnn-ptr-zdata 16)
  (:cudnn-ptr-bn-z-eqscale 17)
  (:cudnn-ptr-bn-z-eqbias 18)
  (:cudnn-ptr-activation-bitmask 19)
  (:cudnn-ptr-dxdata 20)
  (:cudnn-ptr-dzdata 21)
  (:cudnn-ptr-bn-dscale 22)
  (:cudnn-ptr-bn-dbias 23)
  (:cudnn-scalar-size-t-workspace-size-in-bytes 100)
  (:cudnn-scalar-int64-t-bn-accumulation-count 101)
  (:cudnn-scalar-double-bn-exp-avg-factor 102)
  (:cudnn-scalar-double-bn-epsilon 103))

(cffi:defctype cudnnfusedopsvariantparamlabel-t cudnnfusedopsvariantparamlabel-t-enum)

(cffi:defcfun "cudnncreatefusedopsconstparampack" cudnnStatus-t
  (constpack (:pointer cudnnFusedOpsConstParamPack-t))
  (ops cudnnFusedOps-t))

(cffi:defcfun "cudnndestroyfusedopsconstparampack" cudnnStatus-t
  (constpack cudnnFusedOpsConstParamPack-t))

(cffi:defcfun "cudnnsetfusedopsconstparampackattribute" cudnnStatus-t
  (constpack cudnnFusedOpsConstParamPack-t)
  (paramlabel cudnnFusedOpsConstParamLabel-t)
  (param (:pointer :void)))

(cffi:defcfun "cudnngetfusedopsconstparampackattribute" cudnnStatus-t
  (constpack cudnnFusedOpsConstParamPack-t)
  (paramlabel cudnnFusedOpsConstParamLabel-t)
  (param (:pointer :void))
  (isnull (:pointer :int)))

(cffi:defcfun "cudnncreatefusedopsvariantparampack" cudnnStatus-t
  (varpack (:pointer cudnnFusedOpsVariantParamPack-t))
  (ops cudnnFusedOps-t))

(cffi:defcfun "cudnndestroyfusedopsvariantparampack" cudnnStatus-t
  (varpack cudnnFusedOpsVariantParamPack-t))

(cffi:defcfun "cudnnsetfusedopsvariantparampackattribute" cudnnStatus-t
  (varpack cudnnFusedOpsVariantParamPack-t)
  (paramlabel cudnnFusedOpsVariantParamLabel-t)
  (ptr (:pointer :void)))

(cffi:defcfun "cudnngetfusedopsvariantparampackattribute" cudnnStatus-t
  (varpack cudnnFusedOpsVariantParamPack-t)
  (paramlabel cudnnFusedOpsVariantParamLabel-t)
  (ptr (:pointer :void)))

(cffi:defcfun "cudnncreatefusedopsplan" cudnnStatus-t
  (plan (:pointer cudnnFusedOpsPlan-t))
  (ops cudnnFusedOps-t))

(cffi:defcfun "cudnndestroyfusedopsplan" cudnnStatus-t
  (plan cudnnFusedOpsPlan-t))

(cffi:defcfun "cudnnmakefusedopsplan" cudnnStatus-t
  (handle cudnnHandle-t)
  (plan cudnnFusedOpsPlan-t)
  (constpack cudnnFusedOpsConstParamPack-t)
  (workspacesizeinbytes (:pointer :int)))

(cffi:defcfun "cudnnfusedopsexecute" cudnnStatus-t
  (handle cudnnHandle-t)
  (plan cudnnFusedOpsPlan-t)
  (varpack cudnnFusedOpsVariantParamPack-t))

(cffi:defcfun ("cudnnsetrnndescriptor_v6" cudnnsetrnndescriptor-v6) cudnnStatus-t
  (handle cudnnHandle-t)
  (rnndesc cudnnRNNDescriptor-t)
  (hiddensize :int)
  (numlayers :int)
  (dropoutdesc cudnnDropoutDescriptor-t)
  (inputmode cudnnRNNInputMode-t)
  (direction cudnnDirectionMode-t)
  (mode cudnnRNNMode-t)
  (algo cudnnRNNAlgo-t)
  (mathprec cudnnDataType-t))

(cffi:defcfun ("cudnnsetrnndescriptor_v5" cudnnsetrnndescriptor-v5) cudnnStatus-t
  (rnndesc cudnnRNNDescriptor-t)
  (hiddensize :int)
  (numlayers :int)
  (dropoutdesc cudnnDropoutDescriptor-t)
  (inputmode cudnnRNNInputMode-t)
  (direction cudnnDirectionMode-t)
  (mode cudnnRNNMode-t)
  (mathprec cudnnDataType-t))

