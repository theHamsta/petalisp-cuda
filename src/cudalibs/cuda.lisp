(defpackage petalisp-cuda.cudalibs
  (:import-from :cl-cuda.lang.type :cffi-type :cffi-type-size)
  (:import-from :cl :defun :let :or :gethash :setf :setq :let* :progn :defmacro :when :nil :values :equalp :assert :t)
  (:export *cudnn-found*
	   :cudnn-init
	   :cudnncreate
	   :cudnnGetReductionWorkspaceSize
	   :cudnnCreateTensorDescriptor
	   :cudnnCreateTensorDescriptor
	   :cudnnDestroy
	   :cudnnDestroyTensorDescriptor
	   :cudnnDestroyReduceTensorDescriptor
	   :cudnn-create-tensor-descriptor
	   :cudnn-create-reduction-descriptor
	   :cudnnGetReductionIndicesSize
	   :CUDNN-STATUS-SUCCESS
	   :cudnnReduceTensor
	   :cudnn-data-int32
	   :cudnn-data-half
	   :cudnn-data-float
	   :cudnn-data-double
	   :cudnn-data-int8
	   :cudnn-data-int8
	   :cudnn-data-uint8
	   :cudnn-data-floatx3
	   :cudnn-data-floatx4
	   :cudnn-data-doublex3
	   :cudnn-data-doublex4))

(in-package petalisp-cuda.cudalibs)
(cl:defparameter *cudnn-found* t)

(cffi:define-foreign-library libcudnn
  (:unix (:or "libcudnn.so.8" "libcudnn.so.7" "libcudnn.so"))
  (t (:default "libcudnn")))
 
(cl:handler-case (cffi:use-foreign-library libcudnn)
  (cffi:load-foreign-library-error (e)
    (cl:format cl:*error-output* "~A~%" e )
    (cl:format cl:*error-output* "CUDNN will not be available for petalisp-cuda!~%")
    (cl:setq *cudnn-found* nil)))

;(cffi:define-foreign-library libcudart
  ;(:unix (:or "/usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so" "libcudart.so"))
  ;(t (:default "libcudart")))
 
;(cffi:use-foreign-library libcudart)
;; next section imported from file /usr/local/cuda/include/cuda.h

#| MACRO_DEFINITION
(defconstant +--cuda-cuda-h--+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +--cuda-deprecated+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +--cuda-api-version+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +--cuda-api-ptds+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +--cuda-api-ptsz+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudevicetotalmem+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuctxcreate+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumodulegetglobal+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemgetinfo+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemalloc+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemallocpitch+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemfree+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemgetaddressrange+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemallochost+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemhostgetdevicepointer+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyhtod+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpydtoh+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpydtod+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpydtoa+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyatod+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyhtoa+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyatoh+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyatoa+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyhtoaasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyatohasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpy2d+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpy2dunaligned+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpy3d+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpyhtodasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpydtohasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpydtodasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpy2dasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemcpy3dasync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd8+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd16+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd32+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd2d8+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd2d16+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemsetd2d32+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuarraycreate+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuarraygetdescriptor+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuarray3dcreate+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuarray3dgetdescriptor+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cutexrefsetaddress+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cutexrefgetaddress+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cugraphicsresourcegetmappedpointer+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuctxdestroy+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuctxpopcurrent+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuctxpushcurrent+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +custreamdestroy+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cueventdestroy+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cutexrefsetaddress2d+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +culinkcreate+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +culinkadddata+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +culinkaddfile+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cumemhostregister+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cugraphicsresourcesetmapflags+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +custreambegincapture+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-version+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-uuid-has-been-defined+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-ipc-handle-size+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-stream-legacy+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-stream-per-thread+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-cb+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostalloc-portable+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostalloc-devicemap+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostalloc-writecombined+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostregister-portable+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostregister-devicemap+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-memhostregister-iomemory+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-external-memory-dedicated+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-cooperative-launch-multi-device-no-pre-launch-sync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-cooperative-launch-multi-device-no-post-launch-sync+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-layered+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-2darray+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-surface-ldst+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-cubemap+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-texture-gather+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-depth-texture+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cuda-array3d-color-attachment+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-trsa-override-format+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-trsf-read-as-integer+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-trsf-normalized-coordinates+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-trsf-srgb+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-launch-param-end+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-launch-param-buffer-pointer+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-launch-param-buffer-size+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-param-tr-default+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-device-cpu+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cu-device-invalid+ ACTUAL_VALUE_HERE)
|#

#| MACRO_DEFINITION
(defconstant +cudaapi+ ACTUAL_VALUE_HERE)
|#

(cffi:defctype cuuint32-t :uint32)

(cffi:defctype cuuint64-t :uint64)

(cffi:defctype cudeviceptr :unsigned-long-long)

(cffi:defctype cudevice :int)

(cffi:defcstruct cuctx-st)

(cffi:defctype cucontext (:pointer (:struct CUctx-st)))

(cffi:defcstruct cumod-st)

(cffi:defctype cumodule (:pointer (:struct CUmod-st)))

(cffi:defcstruct cufunc-st)

(cffi:defctype cufunction (:pointer (:struct CUfunc-st)))

(cffi:defcstruct cuarray-st)

(cffi:defctype cuarray (:pointer (:struct CUarray-st)))

(cffi:defcstruct cumipmappedarray-st)

(cffi:defctype cumipmappedarray (:pointer (:struct CUmipmappedArray-st)))

(cffi:defcstruct cutexref-st)

(cffi:defctype cutexref (:pointer (:struct CUtexref-st)))

(cffi:defcstruct cusurfref-st)

(cffi:defctype cusurfref (:pointer (:struct CUsurfref-st)))

(cffi:defcstruct cuevent-st)

(cffi:defctype cuevent (:pointer (:struct CUevent-st)))

(cffi:defcstruct custream-st)

(cffi:defctype custream (:pointer (:struct CUstream-st)))

(cffi:defcstruct cugraphicsresource-st)

(cffi:defctype cugraphicsresource (:pointer (:struct CUgraphicsResource-st)))

(cffi:defctype cutexobject :unsigned-long-long)

(cffi:defctype cusurfobject :unsigned-long-long)

(cffi:defcstruct cuextmemory-st)

(cffi:defctype cuexternalmemory (:pointer (:struct CUextMemory-st)))

(cffi:defcstruct cuextsemaphore-st)

(cffi:defctype cuexternalsemaphore (:pointer (:struct CUextSemaphore-st)))

(cffi:defcstruct cugraph-st)

(cffi:defctype cugraph (:pointer (:struct CUgraph-st)))

(cffi:defcstruct cugraphnode-st)

(cffi:defctype cugraphnode (:pointer (:struct CUgraphNode-st)))

(cffi:defcstruct cugraphexec-st)

(cffi:defctype cugraphexec (:pointer (:struct CUgraphExec-st)))

(cffi:defcstruct cuuuid-st
  (bytes (:pointer)
))

(cffi:defctype cuuuid (:struct CUuuid-st))

(cffi:defcstruct cuipceventhandle-st
  "CUDA IPC event handle"
  (reserved (:pointer)
))

(cffi:defctype cuipceventhandle (:struct CUipcEventHandle-st))

(cffi:defcstruct cuipcmemhandle-st
  "CUDA IPC mem handle"
  (reserved (:pointer)
))

(cffi:defctype cuipcmemhandle (:struct CUipcMemHandle-st))

(cffi:defcenum cuipcmem-flags-enum
  "CUDA Ipc Mem Flags"
  (:cu-ipc-mem-lazy-enable-peer-access 1))

(cffi:defctype cuipcmem-flags :int ; enum CUipcMem-flags-enum
)

(cffi:defcenum cumemattach-flags-enum
  "CUDA Mem Attach Flags"
  (:cu-mem-attach-global 1)
  (:cu-mem-attach-host 2)
  (:cu-mem-attach-single 4))

(cffi:defctype cumemattach-flags :int ; enum CUmemAttach-flags-enum
)

(cffi:defcenum cuctx-flags-enum
  "Context creation flags"
  (:cu-ctx-sched-auto 0)
  (:cu-ctx-sched-spin 1)
  (:cu-ctx-sched-yield 2)
  (:cu-ctx-sched-blocking-sync 4)
  (:cu-ctx-blocking-sync 4)
  (:cu-ctx-sched-mask 7)
  (:cu-ctx-map-host 8)
  (:cu-ctx-lmem-resize-to-max 16)
  (:cu-ctx-flags-mask 31))

(cffi:defctype cuctx-flags :int ; enum CUctx-flags-enum
)

(cffi:defcenum custream-flags-enum
  "Stream creation flags"
  (:cu-stream-default 0)
  (:cu-stream-non-blocking 1))

(cffi:defctype custream-flags :int ; enum CUstream-flags-enum
)

(cffi:defcenum cuevent-flags-enum
  "Event creation flags"
  (:cu-event-default 0)
  (:cu-event-blocking-sync 1)
  (:cu-event-disable-timing 2)
  (:cu-event-interprocess 4))

(cffi:defctype cuevent-flags :int ; enum CUevent-flags-enum
)

(cffi:defcenum custreamwaitvalue-flags-enum
  "Flags for ::cuStreamWaitValue32 and ::cuStreamWaitValue64"
  (:cu-stream-wait-value-geq 0)
  (:cu-stream-wait-value-eq 1)
  (:cu-stream-wait-value-and 2)
  (:cu-stream-wait-value-nor 3)
  (:cu-stream-wait-value-flush 1073741824))

(cffi:defctype custreamwaitvalue-flags :int ; enum CUstreamWaitValue-flags-enum
)

(cffi:defcenum custreamwritevalue-flags-enum
  "Flags for ::cuStreamWriteValue32"
  (:cu-stream-write-value-default 0)
  (:cu-stream-write-value-no-memory-barrier 1))

(cffi:defctype custreamwritevalue-flags :int ; enum CUstreamWriteValue-flags-enum
)

(cffi:defcenum custreambatchmemoptype-enum
  "Operations for ::cuStreamBatchMemOp"
  (:cu-stream-mem-op-wait-value-32 1)
  (:cu-stream-mem-op-write-value-32 2)
  (:cu-stream-mem-op-wait-value-64 4)
  (:cu-stream-mem-op-write-value-64 5)
  (:cu-stream-mem-op-flush-remote-writes 3))

(cffi:defctype custreambatchmemoptype :int ; enum CUstreamBatchMemOpType-enum
)

;(cffi:defcunion custreambatchmemopparams-union
  ;"Per-operation parameters for ::cuStreamBatchMemOp"
  ;(operation CUstreamBatchMemOpType)
  ;(waitvalue (:struct CUstreamMemOpWaitValueParams-st))
  ;(writevalue (:struct CUstreamMemOpWriteValueParams-st))
  ;(flushremotewrites (:struct CUstreamMemOpFlushRemoteWritesParams-st))
  ;(pad (:pointer)
;))

;(cffi:defctype custreambatchmemopparams (:union CUstreamBatchMemOpParams-union))

(cffi:defcenum cuoccupancy-flags-enum
  "Occupancy calculator flag"
  (:cu-occupancy-default 0)
  (:cu-occupancy-disable-caching-override 1))

(cffi:defctype cuoccupancy-flags :int ; enum CUoccupancy-flags-enum
)

(cffi:defcenum cuarray-format-enum
  "Array formats"
  (:cu-ad-format-unsigned-int8 1)
  (:cu-ad-format-unsigned-int16 2)
  (:cu-ad-format-unsigned-int32 3)
  (:cu-ad-format-signed-int8 8)
  (:cu-ad-format-signed-int16 9)
  (:cu-ad-format-signed-int32 10)
  (:cu-ad-format-half 16)
  (:cu-ad-format-float 32))

(cffi:defctype cuarray-format :int ; enum CUarray-format-enum
)

(cffi:defcenum cuaddress-mode-enum
  "Texture reference addressing modes"
  (:cu-tr-address-mode-wrap 0)
  (:cu-tr-address-mode-clamp 1)
  (:cu-tr-address-mode-mirror 2)
  (:cu-tr-address-mode-border 3))

(cffi:defctype cuaddress-mode :int ; enum CUaddress-mode-enum
)

(cffi:defcenum cufilter-mode-enum
  "Texture reference filtering modes"
  (:cu-tr-filter-mode-point 0)
  (:cu-tr-filter-mode-linear 1))

(cffi:defctype cufilter-mode :int ; enum CUfilter-mode-enum
)

(cffi:defcenum cudevice-attribute-enum
  "Device properties"
  (:cu-device-attribute-max-threads-per-block 1)
  (:cu-device-attribute-max-block-dim-x 2)
  (:cu-device-attribute-max-block-dim-y 3)
  (:cu-device-attribute-max-block-dim-z 4)
  (:cu-device-attribute-max-grid-dim-x 5)
  (:cu-device-attribute-max-grid-dim-y 6)
  (:cu-device-attribute-max-grid-dim-z 7)
  (:cu-device-attribute-max-shared-memory-per-block 8)
  (:cu-device-attribute-shared-memory-per-block 8)
  (:cu-device-attribute-total-constant-memory 9)
  (:cu-device-attribute-warp-size 10)
  (:cu-device-attribute-max-pitch 11)
  (:cu-device-attribute-max-registers-per-block 12)
  (:cu-device-attribute-registers-per-block 12)
  (:cu-device-attribute-clock-rate 13)
  (:cu-device-attribute-texture-alignment 14)
  (:cu-device-attribute-gpu-overlap 15)
  (:cu-device-attribute-multiprocessor-count 16)
  (:cu-device-attribute-kernel-exec-timeout 17)
  (:cu-device-attribute-integrated 18)
  (:cu-device-attribute-can-map-host-memory 19)
  (:cu-device-attribute-compute-mode 20)
  (:cu-device-attribute-maximum-texture1d-width 21)
  (:cu-device-attribute-maximum-texture2d-width 22)
  (:cu-device-attribute-maximum-texture2d-height 23)
  (:cu-device-attribute-maximum-texture3d-width 24)
  (:cu-device-attribute-maximum-texture3d-height 25)
  (:cu-device-attribute-maximum-texture3d-depth 26)
  (:cu-device-attribute-maximum-texture2d-layered-width 27)
  (:cu-device-attribute-maximum-texture2d-layered-height 28)
  (:cu-device-attribute-maximum-texture2d-layered-layers 29)
  (:cu-device-attribute-maximum-texture2d-array-width 27)
  (:cu-device-attribute-maximum-texture2d-array-height 28)
  (:cu-device-attribute-maximum-texture2d-array-numslices 29)
  (:cu-device-attribute-surface-alignment 30)
  (:cu-device-attribute-concurrent-kernels 31)
  (:cu-device-attribute-ecc-enabled 32)
  (:cu-device-attribute-pci-bus-id 33)
  (:cu-device-attribute-pci-device-id 34)
  (:cu-device-attribute-tcc-driver 35)
  (:cu-device-attribute-memory-clock-rate 36)
  (:cu-device-attribute-global-memory-bus-width 37)
  (:cu-device-attribute-l2-cache-size 38)
  (:cu-device-attribute-max-threads-per-multiprocessor 39)
  (:cu-device-attribute-async-engine-count 40)
  (:cu-device-attribute-unified-addressing 41)
  (:cu-device-attribute-maximum-texture1d-layered-width 42)
  (:cu-device-attribute-maximum-texture1d-layered-layers 43)
  (:cu-device-attribute-can-tex2d-gather 44)
  (:cu-device-attribute-maximum-texture2d-gather-width 45)
  (:cu-device-attribute-maximum-texture2d-gather-height 46)
  (:cu-device-attribute-maximum-texture3d-width-alternate 47)
  (:cu-device-attribute-maximum-texture3d-height-alternate 48)
  (:cu-device-attribute-maximum-texture3d-depth-alternate 49)
  (:cu-device-attribute-pci-domain-id 50)
  (:cu-device-attribute-texture-pitch-alignment 51)
  (:cu-device-attribute-maximum-texturecubemap-width 52)
  (:cu-device-attribute-maximum-texturecubemap-layered-width 53)
  (:cu-device-attribute-maximum-texturecubemap-layered-layers 54)
  (:cu-device-attribute-maximum-surface1d-width 55)
  (:cu-device-attribute-maximum-surface2d-width 56)
  (:cu-device-attribute-maximum-surface2d-height 57)
  (:cu-device-attribute-maximum-surface3d-width 58)
  (:cu-device-attribute-maximum-surface3d-height 59)
  (:cu-device-attribute-maximum-surface3d-depth 60)
  (:cu-device-attribute-maximum-surface1d-layered-width 61)
  (:cu-device-attribute-maximum-surface1d-layered-layers 62)
  (:cu-device-attribute-maximum-surface2d-layered-width 63)
  (:cu-device-attribute-maximum-surface2d-layered-height 64)
  (:cu-device-attribute-maximum-surface2d-layered-layers 65)
  (:cu-device-attribute-maximum-surfacecubemap-width 66)
  (:cu-device-attribute-maximum-surfacecubemap-layered-width 67)
  (:cu-device-attribute-maximum-surfacecubemap-layered-layers 68)
  (:cu-device-attribute-maximum-texture1d-linear-width 69)
  (:cu-device-attribute-maximum-texture2d-linear-width 70)
  (:cu-device-attribute-maximum-texture2d-linear-height 71)
  (:cu-device-attribute-maximum-texture2d-linear-pitch 72)
  (:cu-device-attribute-maximum-texture2d-mipmapped-width 73)
  (:cu-device-attribute-maximum-texture2d-mipmapped-height 74)
  (:cu-device-attribute-compute-capability-major 75)
  (:cu-device-attribute-compute-capability-minor 76)
  (:cu-device-attribute-maximum-texture1d-mipmapped-width 77)
  (:cu-device-attribute-stream-priorities-supported 78)
  (:cu-device-attribute-global-l1-cache-supported 79)
  (:cu-device-attribute-local-l1-cache-supported 80)
  (:cu-device-attribute-max-shared-memory-per-multiprocessor 81)
  (:cu-device-attribute-max-registers-per-multiprocessor 82)
  (:cu-device-attribute-managed-memory 83)
  (:cu-device-attribute-multi-gpu-board 84)
  (:cu-device-attribute-multi-gpu-board-group-id 85)
  (:cu-device-attribute-host-native-atomic-supported 86)
  (:cu-device-attribute-single-to-double-precision-perf-ratio 87)
  (:cu-device-attribute-pageable-memory-access 88)
  (:cu-device-attribute-concurrent-managed-access 89)
  (:cu-device-attribute-compute-preemption-supported 90)
  (:cu-device-attribute-can-use-host-pointer-for-registered-mem 91)
  (:cu-device-attribute-can-use-stream-mem-ops 92)
  (:cu-device-attribute-can-use-64-bit-stream-mem-ops 93)
  (:cu-device-attribute-can-use-stream-wait-value-nor 94)
  (:cu-device-attribute-cooperative-launch 95)
  (:cu-device-attribute-cooperative-multi-device-launch 96)
  (:cu-device-attribute-max-shared-memory-per-block-optin 97)
  (:cu-device-attribute-can-flush-remote-writes 98)
  (:cu-device-attribute-host-register-supported 99)
  (:cu-device-attribute-pageable-memory-access-uses-host-page-tables 100)
  (:cu-device-attribute-direct-managed-mem-access-from-host 101)
  (:cu-device-attribute-max 102))

(cffi:defctype cudevice-attribute :int ; enum CUdevice-attribute-enum
)

(cffi:defcstruct cudevprop-st
  "Legacy device properties"
  (maxthreadsperblock :int)
  (maxthreadsdim (:pointer)
)
  (maxgridsize (:pointer)
)
  (sharedmemperblock :int)
  (totalconstantmemory :int)
  (simdwidth :int)
  (mempitch :int)
  (regsperblock :int)
  (clockrate :int)
  (texturealign :int))

(cffi:defctype cudevprop (:struct CUdevprop-st))

(cffi:defcenum cupointer-attribute-enum
  "Pointer information"
  (:cu-pointer-attribute-context 1)
  (:cu-pointer-attribute-memory-type 2)
  (:cu-pointer-attribute-device-pointer 3)
  (:cu-pointer-attribute-host-pointer 4)
  (:cu-pointer-attribute-p2p-tokens 5)
  (:cu-pointer-attribute-sync-memops 6)
  (:cu-pointer-attribute-buffer-id 7)
  (:cu-pointer-attribute-is-managed 8)
  (:cu-pointer-attribute-device-ordinal 9))

(cffi:defctype cupointer-attribute :int ; enum CUpointer-attribute-enum
)

(cffi:defcenum cufunction-attribute-enum
  "Function properties"
  (:cu-func-attribute-max-threads-per-block 0)
  (:cu-func-attribute-shared-size-bytes 1)
  (:cu-func-attribute-const-size-bytes 2)
  (:cu-func-attribute-local-size-bytes 3)
  (:cu-func-attribute-num-regs 4)
  (:cu-func-attribute-ptx-version 5)
  (:cu-func-attribute-binary-version 6)
  (:cu-func-attribute-cache-mode-ca 7)
  (:cu-func-attribute-max-dynamic-shared-size-bytes 8)
  (:cu-func-attribute-preferred-shared-memory-carveout 9)
  (:cu-func-attribute-max 10))

(cffi:defctype cufunction-attribute :int ; enum CUfunction-attribute-enum
)

(cffi:defcenum cufunc-cache-enum
  "Function cache configurations"
  (:cu-func-cache-prefer-none 0)
  (:cu-func-cache-prefer-shared 1)
  (:cu-func-cache-prefer-l1 2)
  (:cu-func-cache-prefer-equal 3))

(cffi:defctype cufunc-cache :int ; enum CUfunc-cache-enum
)

(cffi:defcenum cusharedconfig-enum
  "Shared memory configurations"
  (:cu-shared-mem-config-default-bank-size 0)
  (:cu-shared-mem-config-four-byte-bank-size 1)
  (:cu-shared-mem-config-eight-byte-bank-size 2))

(cffi:defctype cusharedconfig :int ; enum CUsharedconfig-enum
)

(cffi:defcenum cushared-carveout-enum
  "Shared memory carveout configurations. These may be passed to ::cuFuncSetAttribute"
  (:cu-sharedmem-carveout-default -1)
  (:cu-sharedmem-carveout-max-shared 100)
  (:cu-sharedmem-carveout-max-l1 0))

(cffi:defctype cushared-carveout :int ; enum CUshared-carveout-enum
)

(cffi:defcenum cumemorytype-enum
  "Memory types"
  (:cu-memorytype-host 1)
  (:cu-memorytype-device 2)
  (:cu-memorytype-array 3)
  (:cu-memorytype-unified 4))

(cffi:defctype cumemorytype :int ; enum CUmemorytype-enum
)

(cffi:defcenum cucomputemode-enum
  "Compute Modes"
  (:cu-computemode-default 0)
  (:cu-computemode-prohibited 2)
  (:cu-computemode-exclusive-process 3))

(cffi:defctype cucomputemode :int ; enum CUcomputemode-enum
)

(cffi:defcenum cumem-advise-enum
  "Memory advise values"
  (:cu-mem-advise-set-read-mostly 1)
  (:cu-mem-advise-unset-read-mostly 2)
  (:cu-mem-advise-set-preferred-location 3)
  (:cu-mem-advise-unset-preferred-location 4)
  (:cu-mem-advise-set-accessed-by 5)
  (:cu-mem-advise-unset-accessed-by 6))

(cffi:defctype cumem-advise :int ; enum CUmem-advise-enum
)

(cffi:defcenum cumem-range-attribute-enum
  (:cu-mem-range-attribute-read-mostly 1)
  (:cu-mem-range-attribute-preferred-location 2)
  (:cu-mem-range-attribute-accessed-by 3)
  (:cu-mem-range-attribute-last-prefetch-location 4))

(cffi:defctype cumem-range-attribute :int ; enum CUmem-range-attribute-enum
)

(cffi:defcenum cujit-option-enum
  "Online compiler and linker options"
  (:cu-jit-max-registers 0)
  (:cu-jit-threads-per-block 1)
  (:cu-jit-wall-time 2)
  (:cu-jit-info-log-buffer 3)
  (:cu-jit-info-log-buffer-size-bytes 4)
  (:cu-jit-error-log-buffer 5)
  (:cu-jit-error-log-buffer-size-bytes 6)
  (:cu-jit-optimization-level 7)
  (:cu-jit-target-from-cucontext 8)
  (:cu-jit-target 9)
  (:cu-jit-fallback-strategy 10)
  (:cu-jit-generate-debug-info 11)
  (:cu-jit-log-verbose 12)
  (:cu-jit-generate-line-info 13)
  (:cu-jit-cache-mode 14)
  (:cu-jit-new-sm3x-opt 15)
  (:cu-jit-fast-compile 16)
  (:cu-jit-global-symbol-names 17)
  (:cu-jit-global-symbol-addresses 18)
  (:cu-jit-global-symbol-count 19)
  (:cu-jit-num-options 20))

(cffi:defctype cujit-option :int ; enum CUjit-option-enum
)

(cffi:defcenum cujit-target-enum
  "Online compilation targets"
  (:cu-target-compute-20 20)
  (:cu-target-compute-21 21)
  (:cu-target-compute-30 30)
  (:cu-target-compute-32 32)
  (:cu-target-compute-35 35)
  (:cu-target-compute-37 37)
  (:cu-target-compute-50 50)
  (:cu-target-compute-52 52)
  (:cu-target-compute-53 53)
  (:cu-target-compute-60 60)
  (:cu-target-compute-61 61)
  (:cu-target-compute-62 62)
  (:cu-target-compute-70 70)
  (:cu-target-compute-72 72)
  (:cu-target-compute-75 75))

(cffi:defctype cujit-target :int ; enum CUjit-target-enum
)

(cffi:defcenum cujit-fallback-enum
  "Cubin matching fallback strategies"
  (:cu-prefer-ptx 0)
  (:cu-prefer-binary 1))

(cffi:defctype cujit-fallback :int ; enum CUjit-fallback-enum
)

(cffi:defcenum cujit-cachemode-enum
  "Caching modes for dlcm"
  (:cu-jit-cache-option-none 0)
  (:cu-jit-cache-option-cg 1)
  (:cu-jit-cache-option-ca 2))

(cffi:defctype cujit-cachemode :int ; enum CUjit-cacheMode-enum
)

(cffi:defcenum cujitinputtype-enum
  "Device code formats"
  (:cu-jit-input-cubin 0)
  (:cu-jit-input-ptx 1)
  (:cu-jit-input-fatbinary 2)
  (:cu-jit-input-object 3)
  (:cu-jit-input-library 4)
  (:cu-jit-num-input-types 5))

(cffi:defctype cujitinputtype :int ; enum CUjitInputType-enum
)

(cffi:defcstruct culinkstate-st)

(cffi:defctype culinkstate (:pointer))

(cffi:defcenum cugraphicsregisterflags-enum
  "Flags to register a graphics resource"
  (:cu-graphics-register-flags-none 0)
  (:cu-graphics-register-flags-read-only 1)
  (:cu-graphics-register-flags-write-discard 2)
  (:cu-graphics-register-flags-surface-ldst 4)
  (:cu-graphics-register-flags-texture-gather 8))

(cffi:defctype cugraphicsregisterflags :int ; enum CUgraphicsRegisterFlags-enum
)

(cffi:defcenum cugraphicsmapresourceflags-enum
  "Flags for mapping and unmapping interop resources"
  (:cu-graphics-map-resource-flags-none 0)
  (:cu-graphics-map-resource-flags-read-only 1)
  (:cu-graphics-map-resource-flags-write-discard 2))

(cffi:defctype cugraphicsmapresourceflags :int ; enum CUgraphicsMapResourceFlags-enum
)

(cffi:defcenum cuarray-cubemap-face-enum
  "Array indices for cube faces"
  (:cu-cubemap-face-positive-x 0)
  (:cu-cubemap-face-negative-x 1)
  (:cu-cubemap-face-positive-y 2)
  (:cu-cubemap-face-negative-y 3)
  (:cu-cubemap-face-positive-z 4)
  (:cu-cubemap-face-negative-z 5))

(cffi:defctype cuarray-cubemap-face :int ; enum CUarray-cubemap-face-enum
)

(cffi:defcenum culimit-enum
  "Limits"
  (:cu-limit-stack-size 0)
  (:cu-limit-printf-fifo-size 1)
  (:cu-limit-malloc-heap-size 2)
  (:cu-limit-dev-runtime-sync-depth 3)
  (:cu-limit-dev-runtime-pending-launch-count 4)
  (:cu-limit-max-l2-fetch-granularity 5)
  (:cu-limit-max 6))

(cffi:defctype culimit :int ; enum CUlimit-enum
)

(cffi:defcenum curesourcetype-enum
  "Resource types"
  (:cu-resource-type-array 0)
  (:cu-resource-type-mipmapped-array 1)
  (:cu-resource-type-linear 2)
  (:cu-resource-type-pitch2d 3))

(cffi:defctype curesourcetype :int ; enum CUresourcetype-enum
)

(cffi:defctype cuhostfn (:pointer :pointer ; function ptr void (void *)
))

(cffi:defcstruct cuda-kernel-node-params-st
  "GPU kernel node parameters"
  (func CUfunction)
  (griddimx :unsigned-int)
  (griddimy :unsigned-int)
  (griddimz :unsigned-int)
  (blockdimx :unsigned-int)
  (blockdimy :unsigned-int)
  (blockdimz :unsigned-int)
  (sharedmembytes :unsigned-int)
  (kernelparams (:pointer (:pointer :void)))
  (extra (:pointer (:pointer :void))))

(cffi:defctype cuda-kernel-node-params (:struct CUDA-KERNEL-NODE-PARAMS-st))

;(cffi:defcstruct cuda-memset-node-params-st
  ;"Memset node parameters")

;(cffi:defctype cuda-memset-node-params (:struct CUDA-MEMSET-NODE-PARAMS-st))

(cffi:defcstruct cuda-host-node-params-st
  "Host node parameters"
  (fn CUhostFn)
  (userdata (:pointer)))

(cffi:defctype cuda-host-node-params (:struct CUDA-HOST-NODE-PARAMS-st))

(cffi:defcenum cugraphnodetype-enum
  "Graph node types"
  (:cu-graph-node-type-kernel 0)
  (:cu-graph-node-type-memcpy 1)
  (:cu-graph-node-type-memset 2)
  (:cu-graph-node-type-host 3)
  (:cu-graph-node-type-graph 4)
  (:cu-graph-node-type-empty 5)
  (:cu-graph-node-type-count 6))

(cffi:defctype cugraphnodetype :int ; enum CUgraphNodeType-enum
)

(cffi:defcenum custreamcapturestatus-enum
  "Possible stream capture statuses returned by ::cuStreamIsCapturing"
  (:cu-stream-capture-status-none 0)
  (:cu-stream-capture-status-active 1)
  (:cu-stream-capture-status-invalidated 2))

(cffi:defctype custreamcapturestatus :int ; enum CUstreamCaptureStatus-enum
)

(cffi:defcenum custreamcapturemode-enum
  "Possible modes for stream capture thread interactions. For more details see
  ::cuStreamBeginCapture and ::cuThreadExchangeStreamCaptureMode"
  (:cu-stream-capture-mode-global 0)
  (:cu-stream-capture-mode-thread-local 1)
  (:cu-stream-capture-mode-relaxed 2))

(cffi:defctype custreamcapturemode :int ; enum CUstreamCaptureMode-enum
)

(cffi:defcenum cudaerror-enum
  "Error codes"
  (:cuda-success 0)
  (:cuda-error-invalid-value 1)
  (:cuda-error-out-of-memory 2)
  (:cuda-error-not-initialized 3)
  (:cuda-error-deinitialized 4)
  (:cuda-error-profiler-disabled 5)
  (:cuda-error-profiler-not-initialized 6)
  (:cuda-error-profiler-already-started 7)
  (:cuda-error-profiler-already-stopped 8)
  (:cuda-error-no-device 100)
  (:cuda-error-invalid-device 101)
  (:cuda-error-invalid-image 200)
  (:cuda-error-invalid-context 201)
  (:cuda-error-context-already-current 202)
  (:cuda-error-map-failed 205)
  (:cuda-error-unmap-failed 206)
  (:cuda-error-array-is-mapped 207)
  (:cuda-error-already-mapped 208)
  (:cuda-error-no-binary-for-gpu 209)
  (:cuda-error-already-acquired 210)
  (:cuda-error-not-mapped 211)
  (:cuda-error-not-mapped-as-array 212)
  (:cuda-error-not-mapped-as-pointer 213)
  (:cuda-error-ecc-uncorrectable 214)
  (:cuda-error-unsupported-limit 215)
  (:cuda-error-context-already-in-use 216)
  (:cuda-error-peer-access-unsupported 217)
  (:cuda-error-invalid-ptx 218)
  (:cuda-error-invalid-graphics-context 219)
  (:cuda-error-nvlink-uncorrectable 220)
  (:cuda-error-jit-compiler-not-found 221)
  (:cuda-error-invalid-source 300)
  (:cuda-error-file-not-found 301)
  (:cuda-error-shared-object-symbol-not-found 302)
  (:cuda-error-shared-object-init-failed 303)
  (:cuda-error-operating-system 304)
  (:cuda-error-invalid-handle 400)
  (:cuda-error-illegal-state 401)
  (:cuda-error-not-found 500)
  (:cuda-error-not-ready 600)
  (:cuda-error-illegal-address 700)
  (:cuda-error-launch-out-of-resources 701)
  (:cuda-error-launch-timeout 702)
  (:cuda-error-launch-incompatible-texturing 703)
  (:cuda-error-peer-access-already-enabled 704)
  (:cuda-error-peer-access-not-enabled 705)
  (:cuda-error-primary-context-active 708)
  (:cuda-error-context-is-destroyed 709)
  (:cuda-error-assert 710)
  (:cuda-error-too-many-peers 711)
  (:cuda-error-host-memory-already-registered 712)
  (:cuda-error-host-memory-not-registered 713)
  (:cuda-error-hardware-stack-error 714)
  (:cuda-error-illegal-instruction 715)
  (:cuda-error-misaligned-address 716)
  (:cuda-error-invalid-address-space 717)
  (:cuda-error-invalid-pc 718)
  (:cuda-error-launch-failed 719)
  (:cuda-error-cooperative-launch-too-large 720)
  (:cuda-error-not-permitted 800)
  (:cuda-error-not-supported 801)
  (:cuda-error-system-not-ready 802)
  (:cuda-error-system-driver-mismatch 803)
  (:cuda-error-compat-not-supported-on-device 804)
  (:cuda-error-stream-capture-unsupported 900)
  (:cuda-error-stream-capture-invalidated 901)
  (:cuda-error-stream-capture-merge 902)
  (:cuda-error-stream-capture-unmatched 903)
  (:cuda-error-stream-capture-unjoined 904)
  (:cuda-error-stream-capture-isolation 905)
  (:cuda-error-stream-capture-implicit 906)
  (:cuda-error-captured-event 907)
  (:cuda-error-stream-capture-wrong-thread 908)
  (:cuda-error-unknown 999))

(cffi:defctype curesult :int ; enum cudaError-enum
)

(cffi:defcenum cudevice-p2pattribute-enum
  "P2P Attributes"
  (:cu-device-p2p-attribute-performance-rank 1)
  (:cu-device-p2p-attribute-access-supported 2)
  (:cu-device-p2p-attribute-native-atomic-supported 3)
  (:cu-device-p2p-attribute-access-access-supported 4)
  (:cu-device-p2p-attribute-cuda-array-access-supported 4))

(cffi:defctype cudevice-p2pattribute :int ; enum CUdevice-P2PAttribute-enum
)

(cffi:defctype custreamcallback (:pointer :pointer ; function ptr void (CUstream, CUresult, void *)
))

(cffi:defctype size-t :pointer ; function ptr int (int *)
)


(cffi:defcstruct cuda-texture-desc-st
  "Texture descriptor"
  (addressmode (:pointer)
)
  (filtermode CUfilter-mode)
  (flags :unsigned-int)
  (maxanisotropy :unsigned-int)
  (mipmapfiltermode CUfilter-mode)
  (mipmaplevelbias :float)
  (minmipmaplevelclamp :float)
  (maxmipmaplevelclamp :float)
  (bordercolor (:pointer)
)
  (reserved (:pointer)
))

(cffi:defctype cuda-texture-desc (:struct CUDA-TEXTURE-DESC-st))

(cffi:defcenum curesourceviewformat-enum
  "Resource view format"
  (:cu-res-view-format-none 0)
  (:cu-res-view-format-uint-1x8 1)
  (:cu-res-view-format-uint-2x8 2)
  (:cu-res-view-format-uint-4x8 3)
  (:cu-res-view-format-sint-1x8 4)
  (:cu-res-view-format-sint-2x8 5)
  (:cu-res-view-format-sint-4x8 6)
  (:cu-res-view-format-uint-1x16 7)
  (:cu-res-view-format-uint-2x16 8)
  (:cu-res-view-format-uint-4x16 9)
  (:cu-res-view-format-sint-1x16 10)
  (:cu-res-view-format-sint-2x16 11)
  (:cu-res-view-format-sint-4x16 12)
  (:cu-res-view-format-uint-1x32 13)
  (:cu-res-view-format-uint-2x32 14)
  (:cu-res-view-format-uint-4x32 15)
  (:cu-res-view-format-sint-1x32 16)
  (:cu-res-view-format-sint-2x32 17)
  (:cu-res-view-format-sint-4x32 18)
  (:cu-res-view-format-float-1x16 19)
  (:cu-res-view-format-float-2x16 20)
  (:cu-res-view-format-float-4x16 21)
  (:cu-res-view-format-float-1x32 22)
  (:cu-res-view-format-float-2x32 23)
  (:cu-res-view-format-float-4x32 24)
  (:cu-res-view-format-unsigned-bc1 25)
  (:cu-res-view-format-unsigned-bc2 26)
  (:cu-res-view-format-unsigned-bc3 27)
  (:cu-res-view-format-unsigned-bc4 28)
  (:cu-res-view-format-signed-bc4 29)
  (:cu-res-view-format-unsigned-bc5 30)
  (:cu-res-view-format-signed-bc5 31)
  (:cu-res-view-format-unsigned-bc6h 32)
  (:cu-res-view-format-signed-bc6h 33)
  (:cu-res-view-format-unsigned-bc7 34))

(cffi:defctype curesourceviewformat :int ; enum CUresourceViewFormat-enum
)

;(cffi:defcstruct cuda-resource-view-desc-st
  ;"Resource view descriptor")

;(cffi:defctype cuda-resource-view-desc (:struct CUDA-RESOURCE-VIEW-DESC-st))

(cffi:defcstruct cuda-pointer-attribute-p2p-tokens-st
  "GPU Direct v3 tokens"
  (p2ptoken :unsigned-long-long)
  (vaspacetoken :unsigned-int))

(cffi:defctype cuda-pointer-attribute-p2p-tokens (:struct CUDA-POINTER-ATTRIBUTE-P2P-TOKENS-st))

(cffi:defcstruct cuda-launch-params-st
  "Kernel launch parameters"
  (function CUfunction)
  (griddimx :unsigned-int)
  (griddimy :unsigned-int)
  (griddimz :unsigned-int)
  (blockdimx :unsigned-int)
  (blockdimy :unsigned-int)
  (blockdimz :unsigned-int)
  (sharedmembytes :unsigned-int)
  (hstream CUstream)
  (kernelparams (:pointer (:pointer :void))))

(cffi:defctype cuda-launch-params (:struct CUDA-LAUNCH-PARAMS-st))

(cffi:defcenum cuexternalmemoryhandletype-enum
  "External memory handle types"
  (:cu-external-memory-handle-type-opaque-fd 1)
  (:cu-external-memory-handle-type-opaque-win32 2)
  (:cu-external-memory-handle-type-opaque-win32-kmt 3)
  (:cu-external-memory-handle-type-d3d12-heap 4)
  (:cu-external-memory-handle-type-d3d12-resource 5))

(cffi:defctype cuexternalmemoryhandletype :int ; enum CUexternalMemoryHandleType-enum
)

(cffi:defcstruct cuda-external-memory-handle-desc-st-handle-win32
  (handle (:pointer :void))
  (name (:pointer :void)))

(cffi:defcunion cuda-external-memory-handle-desc-st-handle
  (fd :int)
  (win32 (:struct cuda-external-memory-handle-desc-st-handle-win32)))

(cffi:defcstruct cuda-external-memory-handle-desc-st
  "External memory handle descriptor"
  (type CUexternalMemoryHandleType)
  (handle (:union cuda-external-memory-handle-desc-st-handle))
  (size :unsigned-long-long)
  (flags :unsigned-int)
  (reserved (:pointer :unsigned-int)
))

(cffi:defctype cuda-external-memory-handle-desc (:struct CUDA-EXTERNAL-MEMORY-HANDLE-DESC-st))

(cffi:defcstruct cuda-external-memory-buffer-desc-st
  "External memory buffer descriptor"
  (offset :unsigned-long-long)
  (size :unsigned-long-long)
  (flags :unsigned-int)
  (reserved (:pointer :unsigned-int)
))

(cffi:defctype cuda-external-memory-buffer-desc (:struct CUDA-EXTERNAL-MEMORY-BUFFER-DESC-st))

;(cffi:defcstruct cuda-external-memory-mipmapped-array-desc-st
  ;"External memory mipmap descriptor")

;(cffi:defctype cuda-external-memory-mipmapped-array-desc (:struct CUDA-EXTERNAL-MEMORY-MIPMAPPED-ARRAY-DESC-st))

(cffi:defcenum cuexternalsemaphorehandletype-enum
  "External semaphore handle types"
  (:cu-external-semaphore-handle-type-opaque-fd 1)
  (:cu-external-semaphore-handle-type-opaque-win32 2)
  (:cu-external-semaphore-handle-type-opaque-win32-kmt 3)
  (:cu-external-semaphore-handle-type-d3d12-fence 4))

(cffi:defctype cuexternalsemaphorehandletype :int ; enum CUexternalSemaphoreHandleType-enum
)

(cffi:defcstruct cuda-external-semaphore-handle-desc-st-handle-win32
  (handle (:pointer :void))
  (name (:pointer :void)))

(cffi:defcunion cuda-external-semaphore-handle-desc-st-handle
  (fd :int)
  (win32 (:struct cuda-external-semaphore-handle-desc-st-handle-win32)))

(cffi:defcstruct cuda-external-semaphore-handle-desc-st
  "External semaphore handle descriptor"
  (type CUexternalSemaphoreHandleType)
  (handle (:union cuda-external-semaphore-handle-desc-st-handle))
  (flags :unsigned-int)
  (reserved (:pointer :unsigned-int )
))

(cffi:defctype cuda-external-semaphore-handle-desc (:struct CUDA-EXTERNAL-SEMAPHORE-HANDLE-DESC-st))

(cffi:defcstruct cuda-external-semaphore-signal-params-st-params-fence
  (value :unsigned-long-long))

(cffi:defcstruct cuda-external-semaphore-signal-params-st-params
  (fence (:struct cuda-external-semaphore-signal-params-st-params-fence))
  (reserved (:pointer :unsigned-int )
))

(cffi:defcstruct cuda-external-semaphore-signal-params-st
  "External semaphore signal parameters"
  (params (:struct cuda-external-semaphore-signal-params-st-params))
  (flags :unsigned-int)
  (reserved (:pointer :unsigned-int )
))

(cffi:defctype cuda-external-semaphore-signal-params (:struct CUDA-EXTERNAL-SEMAPHORE-SIGNAL-PARAMS-st))

(cffi:defcstruct cuda-external-semaphore-wait-params-st-params-fence
  (value :unsigned-long-long))

(cffi:defcstruct cuda-external-semaphore-wait-params-st-params
  (fence (:struct cuda-external-semaphore-wait-params-st-params-fence))
  (reserved (:pointer :unsigned-int )
))

(cffi:defcstruct cuda-external-semaphore-wait-params-st
  "External semaphore wait parameters"
  (params (:struct cuda-external-semaphore-wait-params-st-params))
  (flags :unsigned-int)
  (reserved (:pointer :unsigned-int )
))

(cffi:defctype cuda-external-semaphore-wait-params (:struct CUDA-EXTERNAL-SEMAPHORE-WAIT-PARAMS-st))

(cffi:defcfun "cugeterrorstring" CUresult
  "\brief Gets the string description of an error code
 
  Sets \p pStr to the address of a NULL-terminated string description
  of the error code \p error.
  If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
  will be returned and \p pStr will be set to the NULL address.
 
  \param error - Error code to convert to string
  \param pStr - Address of the string pointer.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::CUresult,
  ::cudaGetErrorString"
  (error CUresult)
  (pstr (:pointer (:pointer :char))))

(cffi:defcfun "cugeterrorname" CUresult
  "\brief Gets the string representation of an error code enum name
 
  Sets \p pStr to the address of a NULL-terminated string representation
  of the name of the enum error code \p error.
  If the error code is not recognized, ::CUDA_ERROR_INVALID_VALUE
  will be returned and \p pStr will be set to the NULL address.
 
  \param error - Error code to convert to string
  \param pStr - Address of the string pointer.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::CUresult,
  ::cudaGetErrorName"
  (error CUresult)
  (pstr (:pointer (:pointer :char))))

(cffi:defcfun "cuinit" CUresult
  "\brief Initialize the CUDA driver API
 
  Initializes the driver API and must be called before any other function from
  the driver API. Currently, the \p Flags parameter must be 0. If ::cuInit()
  has not been called, any function from the driver API will return
  ::CUDA_ERROR_NOT_INITIALIZED.
 
  \param Flags - Initialization flag for CUDA.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH,
  ::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE
  \notefnerr"
  (flags :unsigned-int))

(cffi:defcfun "cudrivergetversion" CUresult
  "\brief Returns the latest CUDA version supported by driver
 
  Returns in \p driverVersion the version of CUDA supported by
  the driver.  The version is returned as
  (1000 &times; major + 10 &times; minor). For example, CUDA 9.2
  would be represented by 9020.
 
  This function automatically returns ::CUDA_ERROR_INVALID_VALUE if
  \p driverVersion is NULL.
 
  \param driverVersion - Returns the CUDA driver version
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cudaDriverGetVersion,
  ::cudaRuntimeGetVersion"
  (driverversion (:pointer :int)))

(cffi:defcfun "cudeviceget" CUresult
  "\brief Returns a handle to a compute device
 
  Returns in \p device a device handle given an ordinal in the range <b>[0,
  ::cuDeviceGetCount()-1]<b>.
 
  \param device  - Returned device handle
  \param ordinal - Device number to get handle for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetCount,
  ::cuDeviceGetName,
  ::cuDeviceGetUuid,
  ::cuDeviceGetLuid,
  ::cuDeviceTotalMem"
  (device (:pointer CUdevice))
  (ordinal :int))

(cffi:defcfun "cudevicegetcount" CUresult
  "\brief Returns the number of compute-capable devices
 
  Returns in \p count the number of devices with compute capability greater
  than or equal to 2.0 that are available for execution. If there is no such
  device, ::cuDeviceGetCount() returns 0.
 
  \param count - Returned number of compute-capable devices
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetName,
  ::cuDeviceGetUuid,
  ::cuDeviceGetLuid,
  ::cuDeviceGet,
  ::cuDeviceTotalMem,
  ::cudaGetDeviceCount"
  (count (:pointer :int)))

(cffi:defcfun "cudevicegetname" CUresult
  "\brief Returns an identifer string for the device
 
  Returns an ASCII string identifying the device \p dev in the NULL-terminated
  string pointed to by \p name. \p len specifies the maximum length of the
  string that may be returned.
 
  \param name - Returned identifier string for the device
  \param len  - Maximum length of string to store in \p name
  \param dev  - Device to get identifier string for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetUuid,
  ::cuDeviceGetLuid,
  ::cuDeviceGetCount,
  ::cuDeviceGet,
  ::cuDeviceTotalMem,
  ::cudaGetDeviceProperties"
  (name (:pointer :char))
  (len :int)
  (dev CUdevice))

(cffi:defcfun "cudevicegetuuid" CUresult
  "\brief Return an UUID for the device
 
  Returns 16-octets identifing the device \p dev in the structure
  pointed by the \p uuid.
 
  \param uuid - Returned UUID
  \param dev  - Device to get identifier string for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetCount,
  ::cuDeviceGetName,
  ::cuDeviceGetLuid,
  ::cuDeviceGet,
  ::cuDeviceTotalMem,
  ::cudaGetDeviceProperties"
  (uuid (:pointer CUuuid))
  (dev CUdevice))

(cffi:defcfun ("cudevicetotalmem_v2" cudevicetotalmem-v2) CUresult
  (bytes (:pointer size-t))
  (dev CUdevice))

(cffi:defcfun "cudevicegetattribute" CUresult
  "\brief Returns information about the device
 
  Returns in \p pi the integer value of the attribute \p attrib on device
  \p dev. The supported attributes are:
  - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: Maximum number of threads per
    block;
  - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: Maximum x-dimension of a block;
  - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: Maximum y-dimension of a block;
  - ::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: Maximum z-dimension of a block;
  - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: Maximum x-dimension of a grid;
  - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: Maximum y-dimension of a grid;
  - ::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: Maximum z-dimension of a grid;
  - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: Maximum amount of
    shared memory available to a thread block in bytes;
  - ::CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: Memory available on device for
    __constant__ variables in a CUDA C kernel in bytes;
  - ::CU_DEVICE_ATTRIBUTE_WARP_SIZE: Warp size in threads;
  - ::CU_DEVICE_ATTRIBUTE_MAX_PITCH: Maximum pitch in bytes allowed by the
    memory copy functions that involve memory regions allocated through
    ::cuMemAllocPitch();
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: Maximum 1D
   texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: Maximum width
   for a 1D texture bound to linear memory;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH: Maximum
   mipmapped 1D texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: Maximum 2D
   texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: Maximum 2D
   texture height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: Maximum width
   for a 2D texture bound to linear memory;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: Maximum height
   for a 2D texture bound to linear memory;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: Maximum pitch
   in bytes for a 2D texture bound to linear memory;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: Maximum
   mipmapped 2D texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: Maximum
   mipmapped 2D texture height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: Maximum 3D
   texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: Maximum 3D
   texture height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: Maximum 3D
   texture depth;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE:
   Alternate maximum 3D texture width, 0 if no alternate
   maximum 3D texture size is supported;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE:
   Alternate maximum 3D texture height, 0 if no alternate
   maximum 3D texture size is supported;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE:
   Alternate maximum 3D texture depth, 0 if no alternate
   maximum 3D texture size is supported;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH:
   Maximum cubemap texture width or height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH:
   Maximum 1D layered texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:
    Maximum layers in a 1D layered texture;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
   Maximum 2D layered texture width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
    Maximum 2D layered texture height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
    Maximum layers in a 2D layered texture;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH:
    Maximum cubemap layered texture width or height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS:
    Maximum layers in a cubemap layered texture;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH:
    Maximum 1D surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH:
    Maximum 2D surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT:
    Maximum 2D surface height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH:
    Maximum 3D surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT:
    Maximum 3D surface height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH:
    Maximum 3D surface depth;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH:
    Maximum 1D layered surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS:
    Maximum layers in a 1D layered surface;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH:
    Maximum 2D layered surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT:
    Maximum 2D layered surface height;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS:
    Maximum layers in a 2D layered surface;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH:
    Maximum cubemap surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH:
    Maximum cubemap layered surface width;
  - ::CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS:
    Maximum layers in a cubemap layered surface;
  - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: Maximum number of 32-bit
    registers available to a thread block;
  - ::CU_DEVICE_ATTRIBUTE_CLOCK_RATE: The typical clock frequency in kilohertz;
  - ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: Alignment requirement; texture
    base addresses aligned to ::textureAlign bytes do not need an offset
    applied to texture fetches;
  - ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: Pitch alignment requirement
    for 2D texture references bound to pitched memory;
  - ::CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: 1 if the device can concurrently copy
    memory between host and device while executing a kernel, or 0 if not;
  - ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: Number of multiprocessors on
    the device;
  - ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: 1 if there is a run time limit
    for kernels executed on the device, or 0 if not;
  - ::CU_DEVICE_ATTRIBUTE_INTEGRATED: 1 if the device is integrated with the
    memory subsystem, or 0 if not;
  - ::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: 1 if the device can map host
    memory into the CUDA address space, or 0 if not;
  - ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: Compute mode that device is currently
    in. Available modes are as follows:
    - ::CU_COMPUTEMODE_DEFAULT: Default mode - Device is not restricted and
      can have multiple CUDA contexts present at a single time.
    - ::CU_COMPUTEMODE_PROHIBITED: Compute-prohibited mode - Device is
      prohibited from creating new CUDA contexts.
    - ::CU_COMPUTEMODE_EXCLUSIVE_PROCESS:  Compute-exclusive-process mode - Device
      can have only one context used by a single process at a time.
  - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: 1 if the device supports
    executing multiple kernels within the same context simultaneously, or 0 if
    not. It is not guaranteed that multiple kernels will be resident
    on the device concurrently so this feature should not be relied upon for
    correctness;
  - ::CU_DEVICE_ATTRIBUTE_ECC_ENABLED: 1 if error correction is enabled on the
     device, 0 if error correction is disabled or not supported by the device;
  - ::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: PCI bus identifier of the device;
  - ::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: PCI device (also known as slot) identifier
    of the device;
  - ::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: PCI domain identifier of the device
  - ::CU_DEVICE_ATTRIBUTE_TCC_DRIVER: 1 if the device is using a TCC driver. TCC
     is only available on Tesla hardware running Windows Vista or later;
  - ::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: Peak memory clock frequency in kilohertz;
  - ::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: Global memory bus width in bits;
  - ::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: Size of L2 cache in bytes. 0 if the device doesn't have L2 cache;
  - ::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: Maximum resident threads per multiprocessor;
  - ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: 1 if the device shares a unified address space with
    the host, or 0 if not;
  - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: Major compute capability version number;
  - ::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: Minor compute capability version number;
  - ::CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: 1 if device supports caching globals
     in L1 cache, 0 if caching globals in L1 cache is not supported by the device;
  - ::CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: 1 if device supports caching locals
     in L1 cache, 0 if caching locals in L1 cache is not supported by the device;
  - ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: Maximum amount of
    shared memory available to a multiprocessor in bytes; this amount is shared
    by all thread blocks simultaneously resident on a multiprocessor;
  - ::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: Maximum number of 32-bit
    registers available to a multiprocessor; this number is shared by all thread
    blocks simultaneously resident on a multiprocessor;
  - ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: 1 if device supports allocating managed memory
    on this system, 0 if allocating managed memory is not supported by the device on this system.
  - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD: 1 if device is on a multi-GPU board, 0 if not.
  - ::CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID: Unique identifier for a group of devices
    associated with the same board. Devices on the same multi-GPU board will share the same identifier.
  - ::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED: 1 if Link between the device and the host
    supports native atomic operations.
  - ::CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO: Ratio of single precision performance
    (in floating-point operations per second) to double precision performance.
  - ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: Device suppports coherently accessing
    pageable memory without calling cudaHostRegister on it.
  - ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: Device can coherently access managed memory
    concurrently with the CPU.
  - ::CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED: Device supports Compute Preemption.
  - ::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM: Device can access host registered
    memory at the same virtual address as the CPU.
  -  ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: The maximum per block shared memory size
     suported on this device. This is the maximum value that can be opted into when using the cuFuncSetAttribute() call.
     For more details see ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
  - ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES: Device accesses pageable memory via the host's
    page tables.
  - ::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST: The host can directly access managed memory on the device without migration.
 
  \param pi     - Returned device attribute value
  \param attrib - Device attribute to query
  \param dev    - Device handle
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetCount,
  ::cuDeviceGetName,
  ::cuDeviceGetUuid,
  ::cuDeviceGet,
  ::cuDeviceTotalMem,
  ::cudaDeviceGetAttribute,
  ::cudaGetDeviceProperties"
  (pi (:pointer :int))
  (attrib CUdevice-attribute)
  (dev CUdevice))

(cffi:defcfun "cudevicegetproperties" CUresult
  "\brief Returns properties for a selected device
 
  \deprecated
 
  This function was deprecated as of CUDA 5.0 and replaced by ::cuDeviceGetAttribute().
 
  Returns in \p prop the properties of device \p dev. The ::CUdevprop
  structure is defined as:
 
  \code
     typedef struct CUdevprop_st {
     int maxThreadsPerBlock;
     int maxThreadsDim[3];
     int maxGridSize[3];
     int sharedMemPerBlock;
     int totalConstantMemory;
     int SIMDWidth;
     int memPitch;
     int regsPerBlock;
     int clockRate;
     int textureAlign
  } CUdevprop;
  \endcode
  where:
 
  - ::maxThreadsPerBlock is the maximum number of threads per block;
  - ::maxThreadsDim[3] is the maximum sizes of each dimension of a block;
  - ::maxGridSize[3] is the maximum sizes of each dimension of a grid;
  - ::sharedMemPerBlock is the total amount of shared memory available per
    block in bytes;
  - ::totalConstantMemory is the total amount of constant memory available on
    the device in bytes;
  - ::SIMDWidth is the warp size;
  - ::memPitch is the maximum pitch allowed by the memory copy functions that
    involve memory regions allocated through ::cuMemAllocPitch();
  - ::regsPerBlock is the total number of registers available per block;
  - ::clockRate is the clock frequency in kilohertz;
  - ::textureAlign is the alignment requirement; texture base addresses that
    are aligned to ::textureAlign bytes do not need an offset applied to
    texture fetches.
 
  \param prop - Returned properties of device
  \param dev  - Device to get properties for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetCount,
  ::cuDeviceGetName,
  ::cuDeviceGetUuid,
  ::cuDeviceGet,
  ::cuDeviceTotalMem"
  (prop (:pointer CUdevprop))
  (dev CUdevice))

(cffi:defcfun "cudevicecomputecapability" CUresult
  "\brief Returns the compute capability of the device
 
  \deprecated
 
  This function was deprecated as of CUDA 5.0 and its functionality superceded
  by ::cuDeviceGetAttribute().
 
  Returns in \p major and \p minor the major and minor revision numbers that
  define the compute capability of the device \p dev.
 
  \param major - Major revision number
  \param minor - Minor revision number
  \param dev   - Device handle
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGetAttribute,
  ::cuDeviceGetCount,
  ::cuDeviceGetName,
  ::cuDeviceGetUuid,
  ::cuDeviceGet,
  ::cuDeviceTotalMem"
  (major (:pointer :int))
  (minor (:pointer :int))
  (dev CUdevice))

(cffi:defcfun "cudeviceprimaryctxretain" CUresult
  "\brief Retain the primary context on the GPU
 
  Retains the primary context on the device, creating it if necessary,
  increasing its usage count. The caller must call
  ::cuDevicePrimaryCtxRelease() when done using the context.
  Unlike ::cuCtxCreate() the newly created context is not pushed onto the stack.
 
  Context creation will fail with ::CUDA_ERROR_UNKNOWN if the compute mode of
  the device is ::CU_COMPUTEMODE_PROHIBITED.  The function ::cuDeviceGetAttribute()
  can be used with ::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE to determine the compute mode
  of the device.
  The <i>nvidia-smi<i> tool can be used to set the compute mode for
  devices. Documentation for <i>nvidia-smi<i> can be obtained by passing a
  -h option to it.
 
  Please note that the primary context always supports pinned allocations. Other
  flags can be specified by ::cuDevicePrimaryCtxSetFlags().
 
  \param pctx  - Returned context handle of the new context
  \param dev   - Device for which primary context is requested
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa ::cuDevicePrimaryCtxRelease,
  ::cuDevicePrimaryCtxSetFlags,
  ::cuCtxCreate,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize"
  (pctx (:pointer CUcontext))
  (dev CUdevice))

(cffi:defcfun "cudeviceprimaryctxrelease" CUresult
  "\brief Release the primary context on the GPU
 
  Releases the primary context interop on the device by decreasing the usage
  count by 1. If the usage drops to 0 the primary context of device \p dev
  will be destroyed regardless of how many threads it is current to.
 
  Please note that unlike ::cuCtxDestroy() this method does not pop the context
  from stack in any circumstances.
 
  \param dev - Device which primary context is released
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa ::cuDevicePrimaryCtxRetain,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize"
  (dev CUdevice))

(cffi:defcfun "cudeviceprimaryctxsetflags" CUresult
  "\brief Set flags for the primary context
 
  Sets the flags for the primary context on the device overwriting perviously
  set ones. If the primary context is already created
  ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE is returned.
 
  The three LSBs of the \p flags parameter can be used to control how the OS
  thread, which owns the CUDA context at the time of an API call, interacts
  with the OS scheduler when waiting for results from the GPU. Only one of
  the scheduling flags can be set when creating a context.
 
  - ::CU_CTX_SCHED_SPIN: Instruct CUDA to actively spin when waiting for
  results from the GPU. This can decrease latency when waiting for the GPU,
  but may lower the performance of CPU threads if they are performing work in
  parallel with the CUDA thread.
 
  - ::CU_CTX_SCHED_YIELD: Instruct CUDA to yield its thread when waiting for
  results from the GPU. This can increase latency when waiting for the GPU,
  but can increase the performance of CPU threads performing work in parallel
  with the GPU.
 
  - ::CU_CTX_SCHED_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  synchronization primitive when waiting for the GPU to finish work.
 
  - ::CU_CTX_BLOCKING_SYNC: Instruct CUDA to block the CPU thread on a
  synchronization primitive when waiting for the GPU to finish work. <br>
  <b>Deprecated:<b> This flag was deprecated as of CUDA 4.0 and was
  replaced with ::CU_CTX_SCHED_BLOCKING_SYNC.
 
  - ::CU_CTX_SCHED_AUTO: The default value if the \p flags parameter is zero,
  uses a heuristic based on the number of active CUDA contexts in the
  process \e C and the number of logical processors in the system \e P. If
  \e C > \e P, then CUDA will yield to other OS threads when waiting for
  the GPU (::CU_CTX_SCHED_YIELD), otherwise CUDA will not yield while
  waiting for results and actively spin on the processor (::CU_CTX_SCHED_SPIN).
  Additionally, on Tegra devices, ::CU_CTX_SCHED_AUTO uses a heuristic based on
  the power profile of the platform and may choose ::CU_CTX_SCHED_BLOCKING_SYNC
  for low-powered devices.
 
  - ::CU_CTX_LMEM_RESIZE_TO_MAX: Instruct CUDA to not reduce local memory
  after resizing local memory for a kernel. This can prevent thrashing by
  local memory allocations when launching many kernels with high local
  memory usage at the cost of potentially increased memory usage.
 
  \param dev   - Device for which the primary context flags are set
  \param flags - New flags for the device
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  \notefnerr
 
  \sa ::cuDevicePrimaryCtxRetain,
  ::cuDevicePrimaryCtxGetState,
  ::cuCtxCreate,
  ::cuCtxGetFlags,
  ::cudaSetDeviceFlags"
  (dev CUdevice)
  (flags :unsigned-int))

(cffi:defcfun "cudeviceprimaryctxgetstate" CUresult
  "\brief Get the state of the primary context
 
  Returns in \p flags the flags for the primary context of \p dev, and in
  \p active whether it is active.  See ::cuDevicePrimaryCtxSetFlags for flag
  values.
 
  \param dev    - Device to get primary context flags for
  \param flags  - Pointer to store flags
  \param active - Pointer to store context state; 0 = inactive, 1 = active
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_INVALID_VALUE,
  \notefnerr
 
  \sa
  ::cuDevicePrimaryCtxSetFlags,
  ::cuCtxGetFlags,
  ::cudaGetDeviceFlags"
  (dev CUdevice)
  (flags (:pointer :unsigned-int))
  (active (:pointer :int)))

(cffi:defcfun "cudeviceprimaryctxreset" CUresult
  "\brief Destroy all allocations and reset all state on the primary context
 
  Explicitly destroys and cleans up all resources associated with the current
  device in the current process.
 
  Note that it is responsibility of the calling function to ensure that no
  other module in the process is using the device any more. For that reason
  it is recommended to use ::cuDevicePrimaryCtxRelease() in most cases.
  However it is safe for other modules to call ::cuDevicePrimaryCtxRelease()
  even after resetting the device.
 
  \param dev - Device for which primary context is destroyed
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE
  \notefnerr
 
  \sa ::cuDevicePrimaryCtxRetain,
  ::cuDevicePrimaryCtxRelease,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cudaDeviceReset"
  (dev CUdevice))

(cffi:defcfun ("cuctxcreate_v2" cuctxcreate-v2) CUresult
  (pctx (:pointer CUcontext))
  (flags :unsigned-int)
  (dev CUdevice))

(cffi:defcfun ("cuctxdestroy_v2" cuctxdestroy-v2) CUresult
  (ctx CUcontext))

(cffi:defcfun ("cuCtxPushCurrent_v2" cuCtxPushCurrent_v2) CUresult
  (ctx CUcontext))

(cffi:defcfun ("cuCtxPopCurrent_v2" cuCtxPopCurrent_v2) CUresult
  (pctx (:pointer CUcontext)))

(cffi:defcfun "cuCtxSetCurrent" CUresult
  "\brief Binds the specified CUDA context to the calling CPU thread
 
  Binds the specified CUDA context to the calling CPU thread.
  If \p ctx is NULL then the CUDA context previously bound to the
  calling CPU thread is unbound and ::CUDA_SUCCESS is returned.
 
  If there exists a CUDA context stack on the calling CPU thread, this
  will replace the top of that stack with \p ctx.
  If \p ctx is NULL then this will be equivalent to popping the top
  of the calling CPU thread's CUDA context stack (or a no-op if the
  calling CPU thread's CUDA context stack is empty).
 
  \param ctx - Context to bind to the calling CPU thread
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa
  ::cuCtxGetCurrent,
  ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cudaSetDevice"
  (ctx CUcontext))

(cffi:defcfun "cuctxgetcurrent" CUresult
  "\brief Returns the CUDA context bound to the calling CPU thread.
 
  Returns in \p pctx the CUDA context bound to the calling CPU thread.
  If no context is bound to the calling CPU thread then \p pctx is
  set to NULL and ::CUDA_SUCCESS is returned.
 
  \param pctx - Returned context handle
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  \notefnerr
 
  \sa
  ::cuCtxSetCurrent,
  ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cudaGetDevice"
  (pctx (:pointer CUcontext)))

(cffi:defcfun "cuctxgetdevice" CUresult
  "\brief Returns the device ID for the current context
 
  Returns in \p device the ordinal of the current context's device.
 
  \param device - Returned device ID for the current context
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cudaGetDevice"
  (device (:pointer CUdevice)))

(cffi:defcfun "cuctxgetflags" CUresult
  "\brief Returns the flags for the current context
 
  Returns in \p flags the flags of the current context. See ::cuCtxCreate
  for flag values.
 
  \param flags - Pointer to store flags of current context
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetCurrent,
  ::cuCtxGetDevice
  ::cuCtxGetLimit,
  ::cuCtxGetSharedMemConfig,
  ::cuCtxGetStreamPriorityRange,
  ::cudaGetDeviceFlags"
  (flags (:pointer :unsigned-int)))

(cffi:defcfun "cuctxsynchronize" CUresult
  "\brief Block for a context's tasks to complete
 
  Blocks until the device has completed all preceding requested tasks.
  ::cuCtxSynchronize() returns an error if one of the preceding tasks failed.
  If the context was created with the ::CU_CTX_SCHED_BLOCKING_SYNC flag, the
  CPU thread will block until the GPU context has finished its work.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cudaDeviceSynchronize")

(cffi:defcfun "cuctxsetlimit" CUresult
  "\brief Set resource limits
 
  Setting \p limit to \p value is a request by the application to update
  the current limit maintained by the context. The driver is free to
  modify the requested value to meet hw requirements (this could be
  clamping to minimum or maximum values, rounding up to nearest element
  size, etc). The application can use ::cuCtxGetLimit() to find out exactly
  what the limit has been set to.
 
  Setting each ::CUlimit has its own specific restrictions, so each is
  discussed here.
 
  - ::CU_LIMIT_STACK_SIZE controls the stack size in bytes of each GPU thread.
  Note that the CUDA driver will set the \p limit to the maximum of \p value
  and what the kernel function requires.
 
  - ::CU_LIMIT_PRINTF_FIFO_SIZE controls the size in bytes of the FIFO used
    by the ::printf() device system call. Setting ::CU_LIMIT_PRINTF_FIFO_SIZE
    must be performed before launching any kernel that uses the ::printf()
    device system call, otherwise ::CUDA_ERROR_INVALID_VALUE will be returned.
 
  - ::CU_LIMIT_MALLOC_HEAP_SIZE controls the size in bytes of the heap used
    by the ::malloc() and ::free() device system calls. Setting
    ::CU_LIMIT_MALLOC_HEAP_SIZE must be performed before launching any kernel
    that uses the ::malloc() or ::free() device system calls, otherwise
    ::CUDA_ERROR_INVALID_VALUE will be returned.
 
  - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH controls the maximum nesting depth of
    a grid at which a thread can safely call ::cudaDeviceSynchronize(). Setting
    this limit must be performed before any launch of a kernel that uses the
    device runtime and calls ::cudaDeviceSynchronize() above the default sync
    depth, two levels of grids. Calls to ::cudaDeviceSynchronize() will fail
    with error code ::cudaErrorSyncDepthExceeded if the limitation is
    violated. This limit can be set smaller than the default or up the maximum
    launch depth of 24. When setting this limit, keep in mind that additional
    levels of sync depth require the driver to reserve large amounts of device
    memory which can no longer be used for user allocations. If these
    reservations of device memory fail, ::cuCtxSetLimit will return
    ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
    This limit is only applicable to devices of compute capability 3.5 and
    higher. Attempting to set this limit on devices of compute capability less
    than 3.5 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being
    returned.
 
  - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT controls the maximum number of
    outstanding device runtime launches that can be made from the current
    context. A grid is outstanding from the point of launch up until the grid
    is known to have been completed. Device runtime launches which violate
    this limitation fail and return ::cudaErrorLaunchPendingCountExceeded when
    ::cudaGetLastError() is called after launch. If more pending launches than
    the default (2048 launches) are needed for a module using the device
    runtime, this limit can be increased. Keep in mind that being able to
    sustain additional pending launches will require the driver to reserve
    larger amounts of device memory upfront which can no longer be used for
    allocations. If these reservations fail, ::cuCtxSetLimit will return
    ::CUDA_ERROR_OUT_OF_MEMORY, and the limit can be reset to a lower value.
    This limit is only applicable to devices of compute capability 3.5 and
    higher. Attempting to set this limit on devices of compute capability less
    than 3.5 will result in the error ::CUDA_ERROR_UNSUPPORTED_LIMIT being
    returned.
 
  - ::CU_LIMIT_MAX_L2_FETCH_GRANULARITY controls the L2 cache fetch granularity.
    Values can range from 0B to 128B. This is purely a performance hint and
    it can be ignored or clamped depending on the platform.
 
  \param limit - Limit to set
  \param value - Size of limit
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNSUPPORTED_LIMIT,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSynchronize,
  ::cudaDeviceSetLimit"
  (limit CUlimit)
  (value size-t))

(cffi:defcfun "cuctxgetlimit" CUresult
  "\brief Returns resource limits
 
  Returns in \p pvalue the current size of \p limit.  The supported
  ::CUlimit values are:
  - ::CU_LIMIT_STACK_SIZE: stack size in bytes of each GPU thread.
  - ::CU_LIMIT_PRINTF_FIFO_SIZE: size in bytes of the FIFO used by the
    ::printf() device system call.
  - ::CU_LIMIT_MALLOC_HEAP_SIZE: size in bytes of the heap used by the
    ::malloc() and ::free() device system calls.
  - ::CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH: maximum grid depth at which a thread
    can issue the device runtime call ::cudaDeviceSynchronize() to wait on
    child grid launches to complete.
  - ::CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT: maximum number of outstanding
    device runtime launches that can be made from this context.
  - ::CU_LIMIT_MAX_L2_FETCH_GRANULARITY: L2 cache fetch granularity.
 
  \param limit  - Limit to query
  \param pvalue - Returned size of limit
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNSUPPORTED_LIMIT
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cudaDeviceGetLimit"
  (pvalue (:pointer size-t))
  (limit CUlimit))

(cffi:defcfun "cuctxgetcacheconfig" CUresult
  "\brief Returns the preferred cache configuration for the current context.
 
  On devices where the L1 cache and shared memory use the same hardware
  resources, this function returns through \p pconfig the preferred cache configuration
  for the current context. This is only a preference. The driver will use
  the requested configuration if possible, but it is free to choose a different
  configuration if required to execute functions.
 
  This will return a \p pconfig of ::CU_FUNC_CACHE_PREFER_NONE on devices
  where the size of the L1 cache and shared memory are fixed.
 
  The supported cache configurations are:
  - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 
  \param pconfig - Returned cache configuration
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cuFuncSetCacheConfig,
  ::cudaDeviceGetCacheConfig"
  (pconfig (:pointer CUfunc-cache)))

(cffi:defcfun "cuctxsetcacheconfig" CUresult
  "\brief Sets the preferred cache configuration for the current context.
 
  On devices where the L1 cache and shared memory use the same hardware
  resources, this sets through \p config the preferred cache configuration for
  the current context. This is only a preference. The driver will use
  the requested configuration if possible, but it is free to choose a different
  configuration if required to execute the function. Any function preference
  set via ::cuFuncSetCacheConfig() will be preferred over this context-wide
  setting. Setting the context-wide cache configuration to
  ::CU_FUNC_CACHE_PREFER_NONE will cause subsequent kernel launches to prefer
  to not change the cache configuration unless required to launch the kernel.
 
  This setting does nothing on devices where the size of the L1 cache and
  shared memory are fixed.
 
  Launching a kernel with a different preference than the most recent
  preference setting may insert a device-side synchronization point.
 
  The supported cache configurations are:
  - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 
  \param config - Requested cache configuration
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cuFuncSetCacheConfig,
  ::cudaDeviceSetCacheConfig"
  (config CUfunc-cache))

(cffi:defcfun "cuctxgetsharedmemconfig" CUresult
  "\brief Returns the current shared memory configuration for the current context.
 
  This function will return in \p pConfig the current size of shared memory banks
  in the current context. On devices with configurable shared memory banks,
  ::cuCtxSetSharedMemConfig can be used to change this setting, so that all
  subsequent kernel launches will by default use the new bank size. When
  ::cuCtxGetSharedMemConfig is called on devices without configurable shared
  memory, it will return the fixed bank size of the hardware.
 
  The returned bank configurations can be either:
  - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:  shared memory bank width is
    four bytes.
  - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: shared memory bank width will
    eight bytes.
 
  \param pConfig - returned shared memory configuration
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cuCtxGetSharedMemConfig,
  ::cuFuncSetCacheConfig,
  ::cudaDeviceGetSharedMemConfig"
  (pconfig (:pointer CUsharedconfig)))

(cffi:defcfun "cuctxsetsharedmemconfig" CUresult
  "\brief Sets the shared memory configuration for the current context.
 
  On devices with configurable shared memory banks, this function will set
  the context's shared memory bank size which is used for subsequent kernel
  launches.
 
  Changed the shared memory configuration between launches may insert a device
  side synchronization point between those launches.
 
  Changing the shared memory bank size will not increase shared memory usage
  or affect occupancy of kernels, but may have major effects on performance.
  Larger bank sizes will allow for greater potential bandwidth to shared memory,
  but will change what kinds of accesses to shared memory will result in bank
  conflicts.
 
  This function will do nothing on devices with fixed shared memory bank size.
 
  The supported bank configurations are:
  - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: set bank width to the default initial
    setting (currently, four bytes).
  - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
    be natively four bytes.
  - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
    be natively eight bytes.
 
  \param config - requested shared memory configuration
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cuCtxGetSharedMemConfig,
  ::cuFuncSetCacheConfig,
  ::cudaDeviceSetSharedMemConfig"
  (config CUsharedconfig))

(cffi:defcfun "cuctxgetapiversion" CUresult
  "\brief Gets the context's API version.
 
  Returns a version number in \p version corresponding to the capabilities of
  the context (e.g. 3010 or 3020), which library developers can use to direct
  callers to a specific API version. If \p ctx is NULL, returns the API version
  used to create the currently bound context.
 
  Note that new API versions are only introduced when context capabilities are
  changed that break binary compatibility, so the API version and driver version
  may be different. For example, it is valid for the API version to be 3020 while
  the driver version is 4020.
 
  \param ctx     - Context to check
  \param version - Pointer to version
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize"
  (ctx CUcontext)
  (version (:pointer :unsigned-int)))

(cffi:defcfun "cuctxgetstreampriorityrange" CUresult
  "\brief Returns numerical values that correspond to the least and
  greatest stream priorities.
 
  Returns in \p leastPriority and \p greatestPriority the numerical values that correspond
  to the least and greatest stream priorities respectively. Stream priorities
  follow a convention where lower numbers imply greater priorities. The range of
  meaningful stream priorities is given by [\p greatestPriority, \p leastPriority].
  If the user attempts to create a stream with a priority value that is
  outside the meaningful range as specified by this API, the priority is
  automatically clamped down or up to either \p leastPriority or \p greatestPriority
  respectively. See ::cuStreamCreateWithPriority for details on creating a
  priority stream.
  A NULL may be passed in for \p leastPriority or \p greatestPriority if the value
  is not desired.
 
  This function will return '0' in both \p leastPriority and \p greatestPriority if
  the current context's device does not support stream priorities
  (see ::cuDeviceGetAttribute).
 
  \param leastPriority    - Pointer to an int in which the numerical value for least
                            stream priority is returned
  \param greatestPriority - Pointer to an int in which the numerical value for greatest
                            stream priority is returned
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  \notefnerr
 
  \sa ::cuStreamCreateWithPriority,
  ::cuStreamGetPriority,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize,
  ::cudaDeviceGetStreamPriorityRange"
  (leastpriority (:pointer :int))
  (greatestpriority (:pointer :int)))

(cffi:defcfun "cuctxattach" CUresult
  "\brief Increment a context's usage-count
 
  \deprecated
 
  Note that this function is deprecated and should not be used.
 
  Increments the usage count of the context and passes back a context handle
  in \p pctx that must be passed to ::cuCtxDetach() when the application is
  done with the context. ::cuCtxAttach() fails if there is no context current
  to the thread.
 
  Currently, the \p flags parameter must be 0.
 
  \param pctx  - Returned context handle of the current context
  \param flags - Context attach flags (must be 0)
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxDetach,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize"
  (pctx (:pointer CUcontext))
  (flags :unsigned-int))

(cffi:defcfun "cuctxdetach" CUresult
  "\brief Decrement a context's usage-count
 
  \deprecated
 
  Note that this function is deprecated and should not be used.
 
  Decrements the usage count of the context \p ctx, and destroys the context
  if the usage count goes to 0. The context must be a handle that was passed
  back by ::cuCtxCreate() or ::cuCtxAttach(), and must be current to the
  calling thread.
 
  \param ctx - Context to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa ::cuCtxCreate,
  ::cuCtxDestroy,
  ::cuCtxGetApiVersion,
  ::cuCtxGetCacheConfig,
  ::cuCtxGetDevice,
  ::cuCtxGetFlags,
  ::cuCtxGetLimit,
  ::cuCtxPopCurrent,
  ::cuCtxPushCurrent,
  ::cuCtxSetCacheConfig,
  ::cuCtxSetLimit,
  ::cuCtxSynchronize"
  (ctx CUcontext))

(cffi:defcfun "cumoduleload" CUresult
  "\brief Loads a compute module
 
  Takes a filename \p fname and loads the corresponding module \p module into
  the current context. The CUDA driver API does not attempt to lazily
  allocate the resources needed by a module; if the memory for functions and
  data (constant and global) needed by the module cannot be allocated,
  ::cuModuleLoad() fails. The file should be a \e cubin file as output by
  \b nvcc, or a \e PTX file either as output by \b nvcc or handwritten, or
  a \e fatbin file as output by \b nvcc from toolchain 4.0 or later.
 
  \param module - Returned module
  \param fname  - Filename of module to load
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_PTX,
  ::CUDA_ERROR_NOT_FOUND,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_FILE_NOT_FOUND,
  ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
  ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload"
  (module (:pointer CUmodule))
  (fname (:pointer :char)))

(cffi:defcfun "cumoduleloaddata" CUresult
  "\brief Load a module's data
 
  Takes a pointer \p image and loads the corresponding module \p module into
  the current context. The pointer may be obtained by mapping a \e cubin or
  \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
  as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
  object into the executable resources and using operating system calls such
  as Windows \c FindResource() to obtain the pointer.
 
  \param module - Returned module
  \param image  - Module data to load
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_PTX,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
  ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload"
  (module (:pointer CUmodule))
  (image (:pointer :void)))

(cffi:defcfun "cumoduleloaddataex" CUresult
  "\brief Load a module's data with options
 
  Takes a pointer \p image and loads the corresponding module \p module into
  the current context. The pointer may be obtained by mapping a \e cubin or
  \e PTX or \e fatbin file, passing a \e cubin or \e PTX or \e fatbin file
  as a NULL-terminated text string, or incorporating a \e cubin or \e fatbin
  object into the executable resources and using operating system calls such
  as Windows \c FindResource() to obtain the pointer. Options are passed as
  an array via \p options and any corresponding parameters are passed in
  \p optionValues. The number of total options is supplied via \p numOptions.
  Any outputs will be returned via \p optionValues.
 
  \param module       - Returned module
  \param image        - Module data to load
  \param numOptions   - Number of options
  \param options      - Options for JIT
  \param optionValues - Option values for JIT
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_PTX,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
  ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload"
  (module (:pointer CUmodule))
  (image (:pointer :void))
  (numoptions :unsigned-int)
  (options (:pointer CUjit-option))
  (optionvalues (:pointer (:pointer :void))))

(cffi:defcfun "cumoduleloadfatbinary" CUresult
  "\brief Load a module's data
 
  Takes a pointer \p fatCubin and loads the corresponding module \p module
  into the current context. The pointer represents a <i>fat binary<i> object,
  which is a collection of different \e cubin andor \e PTX files, all
  representing the same device code, but compiled and optimized for different
  architectures.
 
  Prior to CUDA 4.0, there was no documented API for constructing and using
  fat binary objects by programmers.  Starting with CUDA 4.0, fat binary
  objects can be constructed by providing the <i>-fatbin option<i> to \b nvcc.
  More information can be found in the \b nvcc document.
 
  \param module   - Returned module
  \param fatCubin - Fat binary to load
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_PTX,
  ::CUDA_ERROR_NOT_FOUND,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_NO_BINARY_FOR_GPU,
  ::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
  ::CUDA_ERROR_JIT_COMPILER_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleUnload"
  (module (:pointer CUmodule))
  (fatcubin (:pointer :void)))

(cffi:defcfun "cumoduleunload" CUresult
  "\brief Unloads a module
 
  Unloads a module \p hmod from the current context.
 
  \param hmod - Module to unload
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary"
  (hmod CUmodule))

(cffi:defcfun "cumodulegetfunction" CUresult
  "\brief Returns a function handle
 
  Returns in \p hfunc the handle of the function of name \p name located in
  module \p hmod. If no function of that name exists, ::cuModuleGetFunction()
  returns ::CUDA_ERROR_NOT_FOUND.
 
  \param hfunc - Returned function handle
  \param hmod  - Module to retrieve function from
  \param name  - Name of function to retrieve
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload"
  (hfunc (:pointer CUfunction))
  (hmod CUmodule)
  (name (:pointer :char)))

(cffi:defcfun ("cumodulegetglobal_v2" cumodulegetglobal-v2) CUresult
  (dptr (:pointer CUdeviceptr))
  (bytes (:pointer size-t))
  (hmod CUmodule)
  (name (:pointer :char)))

(cffi:defcfun "cumodulegettexref" CUresult
  "\brief Returns a handle to a texture reference
 
  Returns in \p pTexRef the handle of the texture reference of name \p name
  in the module \p hmod. If no texture reference of that name exists,
  ::cuModuleGetTexRef() returns ::CUDA_ERROR_NOT_FOUND. This texture reference
  handle should not be destroyed, since it will be destroyed when the module
  is unloaded.
 
  \param pTexRef  - Returned texture reference
  \param hmod     - Module to retrieve texture reference from
  \param name     - Name of texture reference to retrieve
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetSurfRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload,
  ::cudaGetTextureReference"
  (ptexref (:pointer CUtexref))
  (hmod CUmodule)
  (name (:pointer :char)))

(cffi:defcfun "cumodulegetsurfref" CUresult
  "\brief Returns a handle to a surface reference
 
  Returns in \p pSurfRef the handle of the surface reference of name \p name
  in the module \p hmod. If no surface reference of that name exists,
  ::cuModuleGetSurfRef() returns ::CUDA_ERROR_NOT_FOUND.
 
  \param pSurfRef  - Returned surface reference
  \param hmod     - Module to retrieve surface reference from
  \param name     - Name of surface reference to retrieve
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_FOUND
  \notefnerr
 
  \sa ::cuModuleGetFunction,
  ::cuModuleGetGlobal,
  ::cuModuleGetTexRef,
  ::cuModuleLoad,
  ::cuModuleLoadData,
  ::cuModuleLoadDataEx,
  ::cuModuleLoadFatBinary,
  ::cuModuleUnload,
  ::cudaGetSurfaceReference"
  (psurfref (:pointer CUsurfref))
  (hmod CUmodule)
  (name (:pointer :char)))

(cffi:defcfun ("culinkcreate_v2" culinkcreate-v2) CUresult
  (numoptions :unsigned-int)
  (options (:pointer CUjit-option))
  (optionvalues (:pointer (:pointer :void)))
  (stateout (:pointer CUlinkState)))

(cffi:defcfun ("culinkadddata_v2" culinkadddata-v2) CUresult
  (state CUlinkState)
  (type CUjitInputType)
  (data (:pointer :void))
  (size size-t)
  (name (:pointer :char))
  (numoptions :unsigned-int)
  (options (:pointer CUjit-option))
  (optionvalues (:pointer (:pointer :void))))

(cffi:defcfun ("culinkaddfile_v2" culinkaddfile-v2) CUresult
  (state CUlinkState)
  (type CUjitInputType)
  (path (:pointer :char))
  (numoptions :unsigned-int)
  (options (:pointer CUjit-option))
  (optionvalues (:pointer (:pointer :void))))

(cffi:defcfun "culinkcomplete" CUresult
  "\brief Complete a pending linker invocation
 
  Completes the pending linker action and returns the cubin image for the linked
  device code, which can be used with ::cuModuleLoadData.  The cubin is owned by
  \p state, so it should be loaded before \p state is destroyed via ::cuLinkDestroy.
  This call does not destroy \p state.
 
  \param state    A pending linker invocation
  \param cubinOut On success, this will point to the output image
  \param sizeOut  Optional parameter to receive the size of the generated image
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY
 
  \sa ::cuLinkCreate,
  ::cuLinkAddData,
  ::cuLinkAddFile,
  ::cuLinkDestroy,
  ::cuModuleLoadData"
  (state CUlinkState)
  (cubinout (:pointer (:pointer :void)))
  (sizeout (:pointer size-t)))

(cffi:defcfun "culinkdestroy" CUresult
  "\brief Destroys state for a JIT linker invocation.
 
  \param state State object for the linker invocation
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_HANDLE
 
  \sa ::cuLinkCreate"
  (state CUlinkState))

(cffi:defcfun ("cumemgetinfo_v2" cumemgetinfo-v2) CUresult
  (free (:pointer size-t))
  (total (:pointer size-t)))

(cffi:defcfun ("cumemalloc_v2" cumemalloc-v2) CUresult
  (dptr (:pointer CUdeviceptr))
  (bytesize size-t))

(cffi:defcfun ("cumemallocpitch_v2" cumemallocpitch-v2) CUresult
  (dptr (:pointer CUdeviceptr))
  (ppitch (:pointer size-t))
  (widthinbytes size-t)
  (height size-t)
  (elementsizebytes :unsigned-int))

(cffi:defcfun ("cumemfree_v2" cumemfree-v2) CUresult
  (dptr CUdeviceptr))

(cffi:defcfun ("cumemgetaddressrange_v2" cumemgetaddressrange-v2) CUresult
  (pbase (:pointer CUdeviceptr))
  (psize (:pointer size-t))
  (dptr CUdeviceptr))

(cffi:defcfun ("cuMemAllocHost_v2" cuMemAllocHost_v2) CUresult
  (pp (:pointer (:pointer :void)))
  (bytesize size-t))

(cffi:defcfun "cumemfreehost" CUresult
  "\brief Frees page-locked host memory
 
  Frees the memory space pointed to by \p p, which must have been returned by
  a previous call to ::cuMemAllocHost().
 
  \param p - Pointer to memory to free
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  ::cudaFreeHost"
  (p (:pointer :void)))

(cffi:defcfun "cumemhostalloc" CUresult
  "\brief Allocates page-locked host memory
 
  Allocates \p bytesize bytes of host memory that is page-locked and accessible
  to the device. The driver tracks the virtual memory ranges allocated with
  this function and automatically accelerates calls to functions such as
  ::cuMemcpyHtoD(). Since the memory can be accessed directly by the device,
  it can be read or written with much higher bandwidth than pageable memory
  obtained with functions such as ::malloc(). Allocating excessive amounts of
  pinned memory may degrade system performance, since it reduces the amount
  of memory available to the system for paging. As a result, this function is
  best used sparingly to allocate staging areas for data exchange between
  host and device.
 
  The \p Flags parameter enables different options to be specified that
  affect the allocation, as follows.
 
  - ::CU_MEMHOSTALLOC_PORTABLE: The memory returned by this call will be
    considered as pinned memory by all CUDA contexts, not just the one that
    performed the allocation.
 
  - ::CU_MEMHOSTALLOC_DEVICEMAP: Maps the allocation into the CUDA address
    space. The device pointer to the memory may be obtained by calling
    ::cuMemHostGetDevicePointer().
 
  - ::CU_MEMHOSTALLOC_WRITECOMBINED: Allocates the memory as write-combined
    (WC). WC memory can be transferred across the PCI Express bus more
    quickly on some system configurations, but cannot be read efficiently by
    most CPUs. WC memory is a good option for buffers that will be written by
    the CPU and read by the GPU via mapped pinned memory or host->device
    transfers.
 
  All of these flags are orthogonal to one another: a developer may allocate
  memory that is portable, mapped andor write-combined with no restrictions.
 
  The CUDA context must have been created with the ::CU_CTX_MAP_HOST flag in
  order for the ::CU_MEMHOSTALLOC_DEVICEMAP flag to have any effect.
 
  The ::CU_MEMHOSTALLOC_DEVICEMAP flag may be specified on CUDA contexts for
  devices that do not support mapped pinned memory. The failure is deferred
  to ::cuMemHostGetDevicePointer() because the memory may be mapped into
  other CUDA contexts via the ::CU_MEMHOSTALLOC_PORTABLE flag.
 
  The memory allocated by this function must be freed with ::cuMemFreeHost().
 
  Note all host memory allocated using ::cuMemHostAlloc() will automatically
  be immediately accessible to all contexts on all devices which support unified
  addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING).
  Unless the flag ::CU_MEMHOSTALLOC_WRITECOMBINED is specified, the device pointer
  that may be used to access this host memory from those contexts is always equal
  to the returned host pointer \p pp.  If the flag ::CU_MEMHOSTALLOC_WRITECOMBINED
  is specified, then the function ::cuMemHostGetDevicePointer() must be used
  to query the device pointer, even if the context supports unified addressing.
  See \ref CUDA_UNIFIED for additional details.
 
  \param pp       - Returned host pointer to page-locked memory
  \param bytesize - Requested allocation size in bytes
  \param Flags    - Flags for allocation request
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  ::cudaHostAlloc"
  (pp (:pointer (:pointer :void)))
  (bytesize size-t)
  (flags :unsigned-int))

(cffi:defcfun ("cumemhostgetdevicepointer_v2" cumemhostgetdevicepointer-v2) CUresult
  (pdptr (:pointer CUdeviceptr))
  (p (:pointer :void))
  (flags :unsigned-int))

(cffi:defcfun "cumemhostgetflags" CUresult
  "\brief Passes back flags that were used for a pinned allocation
 
  Passes back the flags \p pFlags that were specified when allocating
  the pinned host buffer \p p allocated by ::cuMemHostAlloc.
 
  ::cuMemHostGetFlags() will fail if the pointer does not reside in
  an allocation performed by ::cuMemAllocHost() or ::cuMemHostAlloc().
 
  \param pFlags - Returned flags word
  \param p     - Host pointer
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cuMemAllocHost,
  ::cuMemHostAlloc,
  ::cudaHostGetFlags"
  (pflags (:pointer :unsigned-int))
  (p (:pointer :void)))

(cffi:defcfun "cumemallocmanaged" CUresult
  "\brief Allocates memory that will be automatically managed by the Unified Memory system
 
  Allocates \p bytesize bytes of managed memory on the device and returns in
  \p dptr a pointer to the allocated memory. If the device doesn't support
  allocating managed memory, ::CUDA_ERROR_NOT_SUPPORTED is returned. Support
  for managed memory can be queried using the device attribute
  ::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY. The allocated memory is suitably
  aligned for any kind of variable. The memory is not cleared. If \p bytesize
  is 0, ::cuMemAllocManaged returns ::CUDA_ERROR_INVALID_VALUE. The pointer
  is valid on the CPU and on all GPUs in the system that support managed memory.
  All accesses to this pointer must obey the Unified Memory programming model.
 
  \p flags specifies the default stream association for this allocation.
  \p flags must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST. If
  ::CU_MEM_ATTACH_GLOBAL is specified, then this memory is accessible from
  any stream on any device. If ::CU_MEM_ATTACH_HOST is specified, then the
  allocation should not be accessed from devices that have a zero value for the
  device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS; an explicit call to
  ::cuStreamAttachMemAsync will be required to enable access on such devices.
 
  If the association is later changed via ::cuStreamAttachMemAsync to
  a single stream, the default association as specifed during ::cuMemAllocManaged
  is restored when that stream is destroyed. For __managed__ variables, the
  default association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a
  stream is an asynchronous operation, and as a result, the change to default
  association won't happen until all work in the stream has completed.
 
  Memory allocated with ::cuMemAllocManaged should be released with ::cuMemFree.
 
  Device memory oversubscription is possible for GPUs that have a non-zero value for the
  device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Managed memory on
  such GPUs may be evicted from device memory to host memory at any time by the Unified
  Memory driver in order to make room for other allocations.
 
  In a multi-GPU system where all GPUs have a non-zero value for the device attribute
  ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, managed memory may not be populated when this
  API returns and instead may be populated on access. In such systems, managed memory can
  migrate to any processor's memory at any time. The Unified Memory driver will employ heuristics to
  maintain data locality and prevent excessive page faults to the extent possible. The application
  can also guide the driver about memory usage patterns via ::cuMemAdvise. The application
  can also explicitly migrate memory to a desired processor's memory via
  ::cuMemPrefetchAsync.
 
  In a multi-GPU system where all of the GPUs have a zero value for the device attribute
  ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS and all the GPUs have peer-to-peer support
  with each other, the physical storage for managed memory is created on the GPU which is active
  at the time ::cuMemAllocManaged is called. All other GPUs will reference the data at reduced
  bandwidth via peer mappings over the PCIe bus. The Unified Memory driver does not migrate
  memory among such GPUs.
 
  In a multi-GPU system where not all GPUs have peer-to-peer support with each other and
  where the value of the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  is zero for at least one of those GPUs, the location chosen for physical storage of managed
  memory is system-dependent.
  - On Linux, the location chosen will be device memory as long as the current set of active
  contexts are on devices that either have peer-to-peer support with each other or have a
  non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  If there is an active context on a GPU that does not have a non-zero value for that device
  attribute and it does not have peer-to-peer support with the other devices that have active
  contexts on them, then the location for physical storage will be 'zero-copy' or host memory.
  Note that this means that managed memory that is located in device memory is migrated to
  host memory if a new context is created on a GPU that doesn't have a non-zero value for
  the device attribute and does not support peer-to-peer with at least one of the other devices
  that has an active context. This in turn implies that context creation may fail if there is
  insufficient host memory to migrate all managed allocations.
  - On Windows, the physical storage is always created in 'zero-copy' or host memory.
  All GPUs will reference the data at reduced bandwidth over the PCIe bus. In these
  circumstances, use of the environment variable CUDA_VISIBLE_DEVICES is recommended to
  restrict CUDA to only use those GPUs that have peer-to-peer support.
  Alternatively, users can also set CUDA_MANAGED_FORCE_DEVICE_ALLOC to a
  non-zero value to force the driver to always use device memory for physical storage.
  When this environment variable is set to a non-zero value, all contexts created in
  that process on devices that support managed memory have to be peer-to-peer compatible
  with each other. Context creation will fail if a context is created on a device that
  supports managed memory and is not peer-to-peer compatible with any of the other
  managed memory supporting devices on which contexts were previously created, even if
  those contexts have been destroyed. These environment variables are described
  in the CUDA programming guide under the CUDA environment variables section.
  - On ARM, managed memory is not available on discrete gpu with Drive PX-2.
 
  \param dptr     - Returned device pointer
  \param bytesize - Requested allocation size in bytes
  \param flags    - Must be one of ::CU_MEM_ATTACH_GLOBAL or ::CU_MEM_ATTACH_HOST
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_NOT_SUPPORTED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  ::cuDeviceGetAttribute, ::cuStreamAttachMemAsync,
  ::cudaMallocManaged"
  (dptr (:pointer CUdeviceptr))
  (bytesize size-t)
  (flags :unsigned-int))

(cffi:defcfun "cudevicegetbypcibusid" CUresult
  "\brief Returns a handle to a compute device
 
  Returns in \p device a device handle given a PCI bus ID string.
 
  \param dev      - Returned device handle
 
  \param pciBusId - String in one of the following forms:
  [domain]:[bus]:[device].[function]
  [domain]:[bus]:[device]
  [bus]:[device].[function]
  where \p domain, \p bus, \p device, and \p function are all hexadecimal values
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGet,
  ::cuDeviceGetAttribute,
  ::cuDeviceGetPCIBusId,
  ::cudaDeviceGetByPCIBusId"
  (dev (:pointer CUdevice))
  (pcibusid (:pointer :char)))

(cffi:defcfun "cudevicegetpcibusid" CUresult
  "\brief Returns a PCI Bus Id string for the device
 
  Returns an ASCII string identifying the device \p dev in the NULL-terminated
  string pointed to by \p pciBusId. \p len specifies the maximum length of the
  string that may be returned.
 
  \param pciBusId - Returned identifier string for the device in the following format
  [domain]:[bus]:[device].[function]
  where \p domain, \p bus, \p device, and \p function are all hexadecimal values.
  pciBusId should be large enough to store 13 characters including the NULL-terminator.
 
  \param len      - Maximum length of string to store in \p name
 
  \param dev      - Device to get identifier string for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuDeviceGet,
  ::cuDeviceGetAttribute,
  ::cuDeviceGetByPCIBusId,
  ::cudaDeviceGetPCIBusId"
  (pcibusid (:pointer :char))
  (len :int)
  (dev CUdevice))

(cffi:defcfun "cuipcgeteventhandle" CUresult
  "\brief Gets an interprocess handle for a previously allocated event
 
  Takes as input a previously allocated event. This event must have been
  created with the ::CU_EVENT_INTERPROCESS and ::CU_EVENT_DISABLE_TIMING
  flags set. This opaque handle may be copied into other processes and
  opened with ::cuIpcOpenEventHandle to allow efficient hardware
  synchronization between GPU work in different processes.
 
  After the event has been opened in the importing process,
  ::cuEventRecord, ::cuEventSynchronize, ::cuStreamWaitEvent and
  ::cuEventQuery may be used in either process. Performing operations
  on the imported event after the exported event has been freed
  with ::cuEventDestroy will result in undefined behavior.
 
  IPC functionality is restricted to devices with support for unified
  addressing on Linux and Windows operating systems.
  IPC functionality on Windows is restricted to GPUs in TCC mode
 
  \param pHandle - Pointer to a user allocated CUipcEventHandle
                     in which to return the opaque event handle
  \param event   - Event allocated with ::CU_EVENT_INTERPROCESS and
                     ::CU_EVENT_DISABLE_TIMING flags.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_MAP_FAILED,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuEventCreate,
  ::cuEventDestroy,
  ::cuEventSynchronize,
  ::cuEventQuery,
  ::cuStreamWaitEvent,
  ::cuIpcOpenEventHandle,
  ::cuIpcGetMemHandle,
  ::cuIpcOpenMemHandle,
  ::cuIpcCloseMemHandle,
  ::cudaIpcGetEventHandle"
  (phandle (:pointer CUipcEventHandle))
  (event CUevent))

(cffi:defcfun "cuipcopeneventhandle" CUresult
  "\brief Opens an interprocess event handle for use in the current process
 
  Opens an interprocess event handle exported from another process with
  ::cuIpcGetEventHandle. This function returns a ::CUevent that behaves like
  a locally created event with the ::CU_EVENT_DISABLE_TIMING flag specified.
  This event must be freed with ::cuEventDestroy.
 
  Performing operations on the imported event after the exported event has
  been freed with ::cuEventDestroy will result in undefined behavior.
 
  IPC functionality is restricted to devices with support for unified
  addressing on Linux and Windows operating systems.
  IPC functionality on Windows is restricted to GPUs in TCC mode
 
  \param phEvent - Returns the imported event
  \param handle  - Interprocess handle to open
 
  \returns
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_MAP_FAILED,
  ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuEventCreate,
  ::cuEventDestroy,
  ::cuEventSynchronize,
  ::cuEventQuery,
  ::cuStreamWaitEvent,
  ::cuIpcGetEventHandle,
  ::cuIpcGetMemHandle,
  ::cuIpcOpenMemHandle,
  ::cuIpcCloseMemHandle,
  ::cudaIpcOpenEventHandle"
  (phevent (:pointer CUevent))
  (handle CUipcEventHandle))

(cffi:defcfun "cuipcgetmemhandle" CUresult
  "\brief Gets an interprocess memory handle for an existing device memory
  allocation
 
  Takes a pointer to the base of an existing device memory allocation created
  with ::cuMemAlloc and exports it for use in another process. This is a
  lightweight operation and may be called multiple times on an allocation
  without adverse effects.
 
  If a region of memory is freed with ::cuMemFree and a subsequent call
  to ::cuMemAlloc returns memory with the same device address,
  ::cuIpcGetMemHandle will return a unique handle for the
  new memory.
 
  IPC functionality is restricted to devices with support for unified
  addressing on Linux and Windows operating systems.
  IPC functionality on Windows is restricted to GPUs in TCC mode
 
  \param pHandle - Pointer to user allocated ::CUipcMemHandle to return
                     the handle in.
  \param dptr    - Base pointer to previously allocated device memory
 
  \returns
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_MAP_FAILED,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuMemAlloc,
  ::cuMemFree,
  ::cuIpcGetEventHandle,
  ::cuIpcOpenEventHandle,
  ::cuIpcOpenMemHandle,
  ::cuIpcCloseMemHandle,
  ::cudaIpcGetMemHandle"
  (phandle (:pointer CUipcMemHandle))
  (dptr CUdeviceptr))

(cffi:defcfun "cuipcopenmemhandle" CUresult
  "\brief Opens an interprocess memory handle exported from another process
  and returns a device pointer usable in the local process.
 
  Maps memory exported from another process with ::cuIpcGetMemHandle into
  the current device address space. For contexts on different devices
  ::cuIpcOpenMemHandle can attempt to enable peer access between the
  devices as if the user called ::cuCtxEnablePeerAccess. This behavior is
  controlled by the ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS flag.
  ::cuDeviceCanAccessPeer can determine if a mapping is possible.
 
  ::cuIpcOpenMemHandle can open handles to devices that may not be visible
  in the process calling the API.
 
  Contexts that may open ::CUipcMemHandles are restricted in the following way.
  ::CUipcMemHandles from each ::CUdevice in a given process may only be opened
  by one ::CUcontext per ::CUdevice per other process.
 
  Memory returned from ::cuIpcOpenMemHandle must be freed with
  ::cuIpcCloseMemHandle.
 
  Calling ::cuMemFree on an exported memory region before calling
  ::cuIpcCloseMemHandle in the importing context will result in undefined
  behavior.
 
  IPC functionality is restricted to devices with support for unified
  addressing on Linux and Windows operating systems.
  IPC functionality on Windows is restricted to GPUs in TCC mode
 
  \param pdptr  - Returned device pointer
  \param handle - ::CUipcMemHandle to open
  \param Flags  - Flags for this operation. Must be specified as ::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
 
  \returns
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_MAP_FAILED,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_TOO_MANY_PEERS,
  ::CUDA_ERROR_INVALID_VALUE
 
  \note No guarantees are made about the address returned in \p pdptr.
  In particular, multiple processes may not receive the same address for the same \p handle.
 
  \sa
  ::cuMemAlloc,
  ::cuMemFree,
  ::cuIpcGetEventHandle,
  ::cuIpcOpenEventHandle,
  ::cuIpcGetMemHandle,
  ::cuIpcCloseMemHandle,
  ::cuCtxEnablePeerAccess,
  ::cuDeviceCanAccessPeer,
  ::cudaIpcOpenMemHandle"
  (pdptr (:pointer CUdeviceptr))
  (handle CUipcMemHandle)
  (flags :unsigned-int))

(cffi:defcfun "cuipcclosememhandle" CUresult
  "\brief Close memory mapped with ::cuIpcOpenMemHandle
 
  Unmaps memory returnd by ::cuIpcOpenMemHandle. The original allocation
  in the exporting process as well as imported mappings in other processes
  will be unaffected.
 
  Any resources used to enable peer access will be freed if this is the
  last mapping using them.
 
  IPC functionality is restricted to devices with support for unified
  addressing on Linux and Windows operating systems.
  IPC functionality on Windows is restricted to GPUs in TCC mode
 
  \param dptr - Device pointer returned by ::cuIpcOpenMemHandle
 
  \returns
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_MAP_FAILED,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \sa
  ::cuMemAlloc,
  ::cuMemFree,
  ::cuIpcGetEventHandle,
  ::cuIpcOpenEventHandle,
  ::cuIpcGetMemHandle,
  ::cuIpcOpenMemHandle,
  ::cudaIpcCloseMemHandle"
  (dptr CUdeviceptr))

(cffi:defcfun ("cumemhostregister_v2" cumemhostregister-v2) CUresult
  (p (:pointer :void))
  (bytesize size-t)
  (flags :unsigned-int))

(cffi:defcfun "cumemhostunregister" CUresult
  "\brief Unregisters a memory range that was registered with cuMemHostRegister.
 
  Unmaps the memory range whose base address is specified by \p p, and makes
  it pageable again.
 
  The base address must be the same one specified to ::cuMemHostRegister().
 
  \param p - Host pointer to memory to unregister
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
  \notefnerr
 
  \sa
  ::cuMemHostRegister,
  ::cudaHostUnregister"
  (p (:pointer :void)))

(cffi:defcfun "cumemcpy" CUresult
  "\brief Copies memory
 
  Copies data between two pointers.
  \p dst and \p src are base pointers of the destination and source, respectively.
  \p ByteCount specifies the number of bytes to copy.
  Note that this function infers the type of the transfer (host to host, host to
    device, device to device, or device to host) from the pointer values.  This
    function is only allowed in contexts which support unified addressing.
 
  \param dst - Destination unified virtual address space pointer
  \param src - Source unified virtual address space pointer
  \param ByteCount - Size of memory copy in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_sync
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  ::cudaMemcpy,
  ::cudaMemcpyToSymbol,
  ::cudaMemcpyFromSymbol"
  (dst CUdeviceptr)
  (src CUdeviceptr)
  (bytecount size-t))

(cffi:defcfun "cumemcpypeer" CUresult
  "\brief Copies device memory between two contexts
 
  Copies from device memory in one context to device memory in another
  context. \p dstDevice is the base device pointer of the destination memory
  and \p dstContext is the destination context.  \p srcDevice is the base
  device pointer of the source memory and \p srcContext is the source pointer.
  \p ByteCount specifies the number of bytes to copy.
 
  \param dstDevice  - Destination device pointer
  \param dstContext - Destination context
  \param srcDevice  - Source device pointer
  \param srcContext - Source context
  \param ByteCount  - Size of memory copy in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_sync
 
  \sa ::cuMemcpyDtoD, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  ::cuMemcpy3DPeerAsync,
  ::cudaMemcpyPeer"
  (dstdevice CUdeviceptr)
  (dstcontext CUcontext)
  (srcdevice CUdeviceptr)
  (srccontext CUcontext)
  (bytecount size-t))

(cffi:defcfun ("cumemcpyhtod_v2" cumemcpyhtod-v2) CUresult
  (dstdevice CUdeviceptr)
  (srchost (:pointer :void))
  (bytecount size-t))

(cffi:defcfun ("cumemcpydtoh_v2" cumemcpydtoh-v2) CUresult
  (dsthost (:pointer :void))
  (srcdevice CUdeviceptr)
  (bytecount size-t))

(cffi:defcfun ("cuMemcpyDtoD_v2" cumemcpydtod-v2) CUresult
  (dstdevice CUdeviceptr)
  (srcdevice CUdeviceptr)
  (bytecount size-t))

(cffi:defcfun ("cumemcpydtoa_v2" cumemcpydtoa-v2) CUresult
  (dstarray CUarray)
  (dstoffset size-t)
  (srcdevice CUdeviceptr)
  (bytecount size-t))

(cffi:defcfun ("cumemcpyatod_v2" cumemcpyatod-v2) CUresult
  (dstdevice CUdeviceptr)
  (srcarray CUarray)
  (srcoffset size-t)
  (bytecount size-t))

(cffi:defcfun ("cumemcpyhtoa_v2" cumemcpyhtoa-v2) CUresult
  (dstarray CUarray)
  (dstoffset size-t)
  (srchost (:pointer :void))
  (bytecount size-t))

(cffi:defcfun ("cumemcpyatoh_v2" cumemcpyatoh-v2) CUresult
  (dsthost (:pointer :void))
  (srcarray CUarray)
  (srcoffset size-t)
  (bytecount size-t))

(cffi:defcfun ("cumemcpyatoa_v2" cumemcpyatoa-v2) CUresult
  (dstarray CUarray)
  (dstoffset size-t)
  (srcarray CUarray)
  (srcoffset size-t)
  (bytecount size-t))

(cffi:defcfun ("cumemcpy2d_v2" cumemcpy2d-v2) CUresult
  (pcopy (:pointer)))

(cffi:defcfun ("cumemcpy2dunaligned_v2" cumemcpy2dunaligned-v2) CUresult
  (pcopy (:pointer)))

(cffi:defcfun ("cumemcpy3d_v2" cumemcpy3d-v2) CUresult
  (pcopy (:pointer)))

(cffi:defcfun "cumemcpy3dpeer" CUresult
  "\brief Copies memory between contexts
 
  Perform a 3D memory copy according to the parameters specified in
  \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
  for documentation of its parameters.
 
  \param pCopy - Parameters for the memory copy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_sync
 
  \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  ::cuMemcpy3DPeerAsync,
  ::cudaMemcpy3DPeer"
  (pcopy (:pointer)))

(cffi:defcfun "cumemcpyasync" CUresult
  "\brief Copies memory asynchronously
 
  Copies data between two pointers.
  \p dst and \p src are base pointers of the destination and source, respectively.
  \p ByteCount specifies the number of bytes to copy.
  Note that this function infers the type of the transfer (host to host, host to
    device, device to device, or device to host) from the pointer values.  This
    function is only allowed in contexts which support unified addressing.
 
  \param dst       - Destination unified virtual address space pointer
  \param src       - Source unified virtual address space pointer
  \param ByteCount - Size of memory copy in bytes
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemcpyAsync,
  ::cudaMemcpyToSymbolAsync,
  ::cudaMemcpyFromSymbolAsync"
  (dst CUdeviceptr)
  (src CUdeviceptr)
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun "cumemcpypeerasync" CUresult
  "\brief Copies device memory between two contexts asynchronously.
 
  Copies from device memory in one context to device memory in another
  context. \p dstDevice is the base device pointer of the destination memory
  and \p dstContext is the destination context.  \p srcDevice is the base
  device pointer of the source memory and \p srcContext is the source pointer.
  \p ByteCount specifies the number of bytes to copy.
 
  \param dstDevice  - Destination device pointer
  \param dstContext - Destination context
  \param srcDevice  - Source device pointer
  \param srcContext - Source context
  \param ByteCount  - Size of memory copy in bytes
  \param hStream    - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpy3DPeer, ::cuMemcpyDtoDAsync,
  ::cuMemcpy3DPeerAsync,
  ::cudaMemcpyPeerAsync"
  (dstdevice CUdeviceptr)
  (dstcontext CUcontext)
  (srcdevice CUdeviceptr)
  (srccontext CUcontext)
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cumemcpyhtodasync_v2" cumemcpyhtodasync-v2) CUresult
  (dstdevice CUdeviceptr)
  (srchost (:pointer :void))
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cumemcpydtohasync_v2" cumemcpydtohasync-v2) CUresult
  (dsthost (:pointer :void))
  (srcdevice CUdeviceptr)
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cuMemcpyDtoDAsync_v2" cuMemcpyDtoDAsync_v2) CUresult
  (dstdevice CUdeviceptr)
  (srcdevice CUdeviceptr)
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cumemcpyhtoaasync_v2" cumemcpyhtoaasync-v2) CUresult
  (dstarray CUarray)
  (dstoffset size-t)
  (srchost (:pointer :void))
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cumemcpyatohasync_v2" cumemcpyatohasync-v2) CUresult
  (dsthost (:pointer :void))
  (srcarray CUarray)
  (srcoffset size-t)
  (bytecount size-t)
  (hstream CUstream))

(cffi:defcfun ("cumemcpy2dasync_v2" cumemcpy2dasync-v2) CUresult
  (pcopy (:pointer))
  (hstream CUstream))

(cffi:defcfun ("cumemcpy3dasync_v2" cumemcpy3dasync-v2) CUresult
  (pcopy (:pointer))
  (hstream CUstream))

(cffi:defcfun "cumemcpy3dpeerasync" CUresult
  "\brief Copies memory between contexts asynchronously.
 
  Perform a 3D memory copy according to the parameters specified in
  \p pCopy.  See the definition of the ::CUDA_MEMCPY3D_PEER structure
  for documentation of its parameters.
 
  \param pCopy - Parameters for the memory copy
  \param hStream - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuMemcpyDtoD, ::cuMemcpyPeer, ::cuMemcpyDtoDAsync, ::cuMemcpyPeerAsync,
  ::cuMemcpy3DPeerAsync,
  ::cudaMemcpy3DPeerAsync"
  (pcopy (:pointer))
  (hstream CUstream))

(cffi:defcfun ("cumemsetd8_v2" cumemsetd8-v2) CUresult
  (dstdevice CUdeviceptr)
  (uc :unsigned-char)
  (n size-t))

(cffi:defcfun ("cumemsetd16_v2" cumemsetd16-v2) CUresult
  (dstdevice CUdeviceptr)
  (us :unsigned-short)
  (n size-t))

(cffi:defcfun ("cumemsetd32_v2" cumemsetd32-v2) CUresult
  (dstdevice CUdeviceptr)
  (ui :unsigned-int)
  (n size-t))

(cffi:defcfun ("cumemsetd2d8_v2" cumemsetd2d8-v2) CUresult
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (uc :unsigned-char)
  (width size-t)
  (height size-t))

(cffi:defcfun ("cumemsetd2d16_v2" cumemsetd2d16-v2) CUresult
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (us :unsigned-short)
  (width size-t)
  (height size-t))

(cffi:defcfun ("cumemsetd2d32_v2" cumemsetd2d32-v2) CUresult
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (ui :unsigned-int)
  (width size-t)
  (height size-t))

(cffi:defcfun "cumemsetd8async" CUresult
  "\brief Sets device memory
 
  Sets the memory range of \p N 8-bit values to the specified value
  \p uc.
 
  \param dstDevice - Destination device pointer
  \param uc        - Value to set
  \param N         - Number of elements
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD16Async,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemsetAsync"
  (dstdevice CUdeviceptr)
  (uc :unsigned-char)
  (n size-t)
  (hstream CUstream))

(cffi:defcfun "cumemsetd16async" CUresult
  "\brief Sets device memory
 
  Sets the memory range of \p N 16-bit values to the specified value
  \p us. The \p dstDevice pointer must be two byte aligned.
 
  \param dstDevice - Destination device pointer
  \param us        - Value to set
  \param N         - Number of elements
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemsetAsync"
  (dstdevice CUdeviceptr)
  (us :unsigned-short)
  (n size-t)
  (hstream CUstream))

(cffi:defcfun "cumemsetd32async" CUresult
  "\brief Sets device memory
 
  Sets the memory range of \p N 32-bit values to the specified value
  \p ui. The \p dstDevice pointer must be four byte aligned.
 
  \param dstDevice - Destination device pointer
  \param ui        - Value to set
  \param N         - Number of elements
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async, ::cuMemsetD32,
  ::cudaMemsetAsync"
  (dstdevice CUdeviceptr)
  (ui :unsigned-int)
  (n size-t)
  (hstream CUstream))

(cffi:defcfun "cumemsetd2d8async" CUresult
  "\brief Sets device memory
 
  Sets the 2D memory range of \p Width 8-bit values to the specified value
  \p uc. \p Height specifies the number of rows to set, and \p dstPitch
  specifies the number of bytes between each row. This function performs
  fastest when the pitch is one that has been passed back by
  ::cuMemAllocPitch().
 
  \param dstDevice - Destination device pointer
  \param dstPitch  - Pitch of destination device pointer
  \param uc        - Value to set
  \param Width     - Width of row
  \param Height    - Number of rows
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemset2DAsync"
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (uc :unsigned-char)
  (width size-t)
  (height size-t)
  (hstream CUstream))

(cffi:defcfun "cumemsetd2d16async" CUresult
  "\brief Sets device memory
 
  Sets the 2D memory range of \p Width 16-bit values to the specified value
  \p us. \p Height specifies the number of rows to set, and \p dstPitch
  specifies the number of bytes between each row. The \p dstDevice pointer
  and \p dstPitch offset must be two byte aligned. This function performs
  fastest when the pitch is one that has been passed back by
  ::cuMemAllocPitch().
 
  \param dstDevice - Destination device pointer
  \param dstPitch  - Pitch of destination device pointer
  \param us        - Value to set
  \param Width     - Width of row
  \param Height    - Number of rows
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D32, ::cuMemsetD2D32Async,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemset2DAsync"
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (us :unsigned-short)
  (width size-t)
  (height size-t)
  (hstream CUstream))

(cffi:defcfun "cumemsetd2d32async" CUresult
  "\brief Sets device memory
 
  Sets the 2D memory range of \p Width 32-bit values to the specified value
  \p ui. \p Height specifies the number of rows to set, and \p dstPitch
  specifies the number of bytes between each row. The \p dstDevice pointer
  and \p dstPitch offset must be four byte aligned. This function performs
  fastest when the pitch is one that has been passed back by
  ::cuMemAllocPitch().
 
  \param dstDevice - Destination device pointer
  \param dstPitch  - Pitch of destination device pointer
  \param ui        - Value to set
  \param Width     - Width of row
  \param Height    - Number of rows
  \param hStream   - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
  \note_memset
  \note_null_stream
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayDestroy, ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D8Async,
  ::cuMemsetD2D16, ::cuMemsetD2D16Async, ::cuMemsetD2D32,
  ::cuMemsetD8, ::cuMemsetD8Async, ::cuMemsetD16, ::cuMemsetD16Async,
  ::cuMemsetD32, ::cuMemsetD32Async,
  ::cudaMemset2DAsync"
  (dstdevice CUdeviceptr)
  (dstpitch size-t)
  (ui :unsigned-int)
  (width size-t)
  (height size-t)
  (hstream CUstream))

(cffi:defcfun ("cuarraycreate_v2" cuarraycreate-v2) CUresult
  (phandle (:pointer))
  (pallocatearray (:pointer)))

(cffi:defcfun ("cuarraygetdescriptor_v2" cuarraygetdescriptor-v2) CUresult
  (parraydescriptor (:pointer))
  (harray CUarray))

(cffi:defcfun "cuarraydestroy" CUresult
  "\brief Destroys a CUDA array
 
  Destroys the CUDA array \p hArray.
 
  \param hArray - Array to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_ARRAY_IS_MAPPED,
  ::CUDA_ERROR_CONTEXT_IS_DESTROYED
  \notefnerr
 
  \sa ::cuArray3DCreate, ::cuArray3DGetDescriptor, ::cuArrayCreate,
  ::cuArrayGetDescriptor, ::cuMemAlloc, ::cuMemAllocHost,
  ::cuMemAllocPitch, ::cuMemcpy2D, ::cuMemcpy2DAsync, ::cuMemcpy2DUnaligned,
  ::cuMemcpy3D, ::cuMemcpy3DAsync, ::cuMemcpyAtoA, ::cuMemcpyAtoD,
  ::cuMemcpyAtoH, ::cuMemcpyAtoHAsync, ::cuMemcpyDtoA, ::cuMemcpyDtoD, ::cuMemcpyDtoDAsync,
  ::cuMemcpyDtoH, ::cuMemcpyDtoHAsync, ::cuMemcpyHtoA, ::cuMemcpyHtoAAsync,
  ::cuMemcpyHtoD, ::cuMemcpyHtoDAsync, ::cuMemFree, ::cuMemFreeHost,
  ::cuMemGetAddressRange, ::cuMemGetInfo, ::cuMemHostAlloc,
  ::cuMemHostGetDevicePointer, ::cuMemsetD2D8, ::cuMemsetD2D16,
  ::cuMemsetD2D32, ::cuMemsetD8, ::cuMemsetD16, ::cuMemsetD32,
  ::cudaFreeArray"
  (harray CUarray))

(cffi:defcfun ("cuarray3dcreate_v2" cuarray3dcreate-v2) CUresult
  (phandle (:pointer))
  (pallocatearray (:pointer)))

(cffi:defcfun ("cuarray3dgetdescriptor_v2" cuarray3dgetdescriptor-v2) CUresult
  (parraydescriptor (:pointer))
  (harray CUarray))

(cffi:defcfun "cumipmappedarraycreate" CUresult
  "\brief Creates a CUDA mipmapped array
 
  Creates a CUDA mipmapped array according to the ::CUDA_ARRAY3D_DESCRIPTOR structure
  \p pMipmappedArrayDesc and returns a handle to the new CUDA mipmapped array in \p pHandle.
  \p numMipmapLevels specifies the number of mipmap levels to be allocated. This value is
  clamped to the range [1, 1 + floor(log2(max(width, height, depth)))].
 
  The ::CUDA_ARRAY3D_DESCRIPTOR is defined as:
 
  \code
    typedef struct {
        unsigned int Width;
        unsigned int Height;
        unsigned int Depth;
        CUarray_format Format;
        unsigned int NumChannels;
        unsigned int Flags;
    } CUDA_ARRAY3D_DESCRIPTOR;
  \endcode
  where:
 
  - \p Width, \p Height, and \p Depth are the width, height, and depth of the
  CUDA array (in elements); the following types of CUDA arrays can be allocated:
      - A 1D mipmapped array is allocated if \p Height and \p Depth extents are both zero.
      - A 2D mipmapped array is allocated if only \p Depth extent is zero.
      - A 3D mipmapped array is allocated if all three extents are non-zero.
      - A 1D layered CUDA mipmapped array is allocated if only \p Height is zero and the
        ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 1D array. The number
        of layers is determined by the depth extent.
      - A 2D layered CUDA mipmapped array is allocated if all three extents are non-zero and
        the ::CUDA_ARRAY3D_LAYERED flag is set. Each layer is a 2D array. The number
        of layers is determined by the depth extent.
      - A cubemap CUDA mipmapped array is allocated if all three extents are non-zero and the
        ::CUDA_ARRAY3D_CUBEMAP flag is set. \p Width must be equal to \p Height, and
        \p Depth must be six. A cubemap is a special type of 2D layered CUDA array,
        where the six layers represent the six faces of a cube. The order of the six
        layers in memory is the same as that listed in ::CUarray_cubemap_face.
      - A cubemap layered CUDA mipmapped array is allocated if all three extents are non-zero,
        and both, ::CUDA_ARRAY3D_CUBEMAP and ::CUDA_ARRAY3D_LAYERED flags are set.
        \p Width must be equal to \p Height, and \p Depth must be a multiple of six.
        A cubemap layered CUDA array is a special type of 2D layered CUDA array that
        consists of a collection of cubemaps. The first six layers represent the first
        cubemap, the next six layers form the second cubemap, and so on.
 
  - ::Format specifies the format of the elements; ::CUarray_format is
  defined as:
  \code
    typedef enum CUarray_format_enum {
        CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
        CU_AD_FORMAT_UNSIGNED_INT16 = 0x02,
        CU_AD_FORMAT_UNSIGNED_INT32 = 0x03,
        CU_AD_FORMAT_SIGNED_INT8 = 0x08,
        CU_AD_FORMAT_SIGNED_INT16 = 0x09,
        CU_AD_FORMAT_SIGNED_INT32 = 0x0a,
        CU_AD_FORMAT_HALF = 0x10,
        CU_AD_FORMAT_FLOAT = 0x20
    } CUarray_format;
   \endcode
 
  - \p NumChannels specifies the number of packed components per CUDA array
  element; it may be 1, 2, or 4;
 
  - ::Flags may be set to
    - ::CUDA_ARRAY3D_LAYERED to enable creation of layered CUDA mipmapped arrays. If this flag is set,
      \p Depth specifies the number of layers, not the depth of a 3D array.
    - ::CUDA_ARRAY3D_SURFACE_LDST to enable surface references to be bound to individual mipmap levels of
      the CUDA mipmapped array. If this flag is not set, ::cuSurfRefSetArray will fail when attempting to
      bind a mipmap level of the CUDA mipmapped array to a surface reference.
     - ::CUDA_ARRAY3D_CUBEMAP to enable creation of mipmapped cubemaps. If this flag is set, \p Width must be
      equal to \p Height, and \p Depth must be six. If the ::CUDA_ARRAY3D_LAYERED flag is also set,
      then \p Depth must be a multiple of six.
    - ::CUDA_ARRAY3D_TEXTURE_GATHER to indicate that the CUDA mipmapped array will be used for texture gather.
      Texture gather can only be performed on 2D CUDA mipmapped arrays.
 
  \p Width, \p Height and \p Depth must meet certain size requirements as listed in the following table.
  All values are specified in elements. Note that for brevity's sake, the full name of the device attribute
  is not specified. For ex., TEXTURE1D_MIPMAPPED_WIDTH refers to the device attribute
  ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH.
 
  <table>
  <tr><td><b>CUDA array type<b><td>
  <td><b>Valid extents that must always be met<br>{(width range in elements), (height range),
  (depth range)}<b><td>
  <td><b>Valid extents with CUDA_ARRAY3D_SURFACE_LDST set<br>
  {(width range in elements), (height range), (depth range)}<b><td><tr>
  <tr><td>1D<td>
  <td><small>{ (1,TEXTURE1D_MIPMAPPED_WIDTH), 0, 0 }<small><td>
  <td><small>{ (1,SURFACE1D_WIDTH), 0, 0 }<small><td><tr>
  <tr><td>2D<td>
  <td><small>{ (1,TEXTURE2D_MIPMAPPED_WIDTH), (1,TEXTURE2D_MIPMAPPED_HEIGHT), 0 }<small><td>
  <td><small>{ (1,SURFACE2D_WIDTH), (1,SURFACE2D_HEIGHT), 0 }<small><td><tr>
  <tr><td>3D<td>
  <td><small>{ (1,TEXTURE3D_WIDTH), (1,TEXTURE3D_HEIGHT), (1,TEXTURE3D_DEPTH) }
  <br>OR<br>{ (1,TEXTURE3D_WIDTH_ALTERNATE), (1,TEXTURE3D_HEIGHT_ALTERNATE),
  (1,TEXTURE3D_DEPTH_ALTERNATE) }<small><td>
  <td><small>{ (1,SURFACE3D_WIDTH), (1,SURFACE3D_HEIGHT),
  (1,SURFACE3D_DEPTH) }<small><td><tr>
  <tr><td>1D Layered<td>
  <td><small>{ (1,TEXTURE1D_LAYERED_WIDTH), 0,
  (1,TEXTURE1D_LAYERED_LAYERS) }<small><td>
  <td><small>{ (1,SURFACE1D_LAYERED_WIDTH), 0,
  (1,SURFACE1D_LAYERED_LAYERS) }<small><td><tr>
  <tr><td>2D Layered<td>
  <td><small>{ (1,TEXTURE2D_LAYERED_WIDTH), (1,TEXTURE2D_LAYERED_HEIGHT),
  (1,TEXTURE2D_LAYERED_LAYERS) }<small><td>
  <td><small>{ (1,SURFACE2D_LAYERED_WIDTH), (1,SURFACE2D_LAYERED_HEIGHT),
  (1,SURFACE2D_LAYERED_LAYERS) }<small><td><tr>
  <tr><td>Cubemap<td>
  <td><small>{ (1,TEXTURECUBEMAP_WIDTH), (1,TEXTURECUBEMAP_WIDTH), 6 }<small><td>
  <td><small>{ (1,SURFACECUBEMAP_WIDTH),
  (1,SURFACECUBEMAP_WIDTH), 6 }<small><td><tr>
  <tr><td>Cubemap Layered<td>
  <td><small>{ (1,TEXTURECUBEMAP_LAYERED_WIDTH), (1,TEXTURECUBEMAP_LAYERED_WIDTH),
  (1,TEXTURECUBEMAP_LAYERED_LAYERS) }<small><td>
  <td><small>{ (1,SURFACECUBEMAP_LAYERED_WIDTH), (1,SURFACECUBEMAP_LAYERED_WIDTH),
  (1,SURFACECUBEMAP_LAYERED_LAYERS) }<small><td><tr>
  <table>
 
 
  \param pHandle             - Returned mipmapped array
  \param pMipmappedArrayDesc - mipmapped array descriptor
  \param numMipmapLevels     - Number of mipmap levels
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cuMipmappedArrayDestroy,
  ::cuMipmappedArrayGetLevel,
  ::cuArrayCreate,
  ::cudaMallocMipmappedArray"
  (phandle (:pointer))
  (pmipmappedarraydesc (:pointer))
  (nummipmaplevels :unsigned-int))

(cffi:defcfun "cumipmappedarraygetlevel" CUresult
  "\brief Gets a mipmap level of a CUDA mipmapped array
 
  Returns in \p pLevelArray a CUDA array that represents a single mipmap level
  of the CUDA mipmapped array \p hMipmappedArray.
 
  If \p level is greater than the maximum number of levels in this mipmapped array,
  ::CUDA_ERROR_INVALID_VALUE is returned.
 
  \param pLevelArray     - Returned mipmap level CUDA array
  \param hMipmappedArray - CUDA mipmapped array
  \param level           - Mipmap level
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa
  ::cuMipmappedArrayCreate,
  ::cuMipmappedArrayDestroy,
  ::cuArrayCreate,
  ::cudaGetMipmappedArrayLevel"
  (plevelarray (:pointer CUarray))
  (hmipmappedarray CUmipmappedArray)
  (level :unsigned-int))

(cffi:defcfun "cumipmappedarraydestroy" CUresult
  "\brief Destroys a CUDA mipmapped array
 
  Destroys the CUDA mipmapped array \p hMipmappedArray.
 
  \param hMipmappedArray - Mipmapped array to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_ARRAY_IS_MAPPED,
  ::CUDA_ERROR_CONTEXT_IS_DESTROYED
  \notefnerr
 
  \sa
  ::cuMipmappedArrayCreate,
  ::cuMipmappedArrayGetLevel,
  ::cuArrayCreate,
  ::cudaFreeMipmappedArray"
  (hmipmappedarray CUmipmappedArray))

(cffi:defcfun "cuPointerGetAttribute" CUresult
  "\brief Returns information about a pointer
 
  The supported attributes are:
 
  - ::CU_POINTER_ATTRIBUTE_CONTEXT:
 
       Returns in \p data the ::CUcontext in which \p ptr was allocated or
       registered.
       The type of \p data must be ::CUcontext .
 
       If \p ptr was not allocated by, mapped by, or registered with
       a ::CUcontext which uses unified virtual addressing then
       ::CUDA_ERROR_INVALID_VALUE is returned.
 
  - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
 
       Returns in \p data the physical memory type of the memory that
       \p ptr addresses as a ::CUmemorytype enumerated value.
       The type of \p data must be unsigned int.
 
       If \p ptr addresses device memory then \p data is set to
       ::CU_MEMORYTYPE_DEVICE.  The particular ::CUdevice on which the
       memory resides is the ::CUdevice of the ::CUcontext returned by the
       ::CU_POINTER_ATTRIBUTE_CONTEXT attribute of \p ptr.
 
       If \p ptr addresses host memory then \p data is set to
       ::CU_MEMORYTYPE_HOST.
 
       If \p ptr was not allocated by, mapped by, or registered with
       a ::CUcontext which uses unified virtual addressing then
       ::CUDA_ERROR_INVALID_VALUE is returned.
 
       If the current ::CUcontext does not support unified virtual
       addressing then ::CUDA_ERROR_INVALID_CONTEXT is returned.
 
  - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER:
 
       Returns in \p data the device pointer value through which
       \p ptr may be accessed by kernels running in the current
       ::CUcontext.
       The type of \p data must be CUdeviceptr .
 
       If there exists no device pointer value through which
       kernels running in the current ::CUcontext may access
       \p ptr then ::CUDA_ERROR_INVALID_VALUE is returned.
 
       If there is no current ::CUcontext then
       ::CUDA_ERROR_INVALID_CONTEXT is returned.
 
       Except in the exceptional disjoint addressing cases discussed
       below, the value returned in \p data will equal the input
       value \p ptr.
 
  - ::CU_POINTER_ATTRIBUTE_HOST_POINTER:
 
       Returns in \p data the host pointer value through which
       \p ptr may be accessed by by the host program.
       The type of \p data must be void .
       If there exists no host pointer value through which
       the host program may directly access \p ptr then
       ::CUDA_ERROR_INVALID_VALUE is returned.
 
       Except in the exceptional disjoint addressing cases discussed
       below, the value returned in \p data will equal the input
       value \p ptr.
 
  - ::CU_POINTER_ATTRIBUTE_P2P_TOKENS:
 
       Returns in \p data two tokens for use with the nv-p2p.h Linux
       kernel interface. \p data must be a struct of type
       CUDA_POINTER_ATTRIBUTE_P2P_TOKENS.
 
       \p ptr must be a pointer to memory obtained from :cuMemAlloc().
       Note that p2pToken and vaSpaceToken are only valid for the
       lifetime of the source allocation. A subsequent allocation at
       the same address may return completely different tokens.
       Querying this attribute has a side effect of setting the attribute
       ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS for the region of memory that
       \p ptr points to.
 
  - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 
       A boolean attribute which when set, ensures that synchronous memory operations
       initiated on the region of memory that \p ptr points to will always synchronize.
       See further documentation in the section titled `API synchronization behavior`
       to learn more about cases when synchronous memory operations can
       exhibit asynchronous behavior.
 
  - ::CU_POINTER_ATTRIBUTE_BUFFER_ID:
 
       Returns in \p data a buffer ID which is guaranteed to be unique within the process.
       \p data must point to an unsigned long long.
 
       \p ptr must be a pointer to memory obtained from a CUDA memory allocation API.
       Every memory allocation from any of the CUDA memory allocation APIs will
       have a unique ID over a process lifetime. Subsequent allocations do not reuse IDs
       from previous freed allocations. IDs are only unique within a single process.
 
 
  - ::CU_POINTER_ATTRIBUTE_IS_MANAGED:
 
       Returns in \p data a boolean that indicates whether the pointer points to
       managed memory or not.
 
  - ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
 
       Returns in \p data an integer representing a device ordinal of a device against
       which the memory was allocated or registered.
 
  \par
 
  Note that for most allocations in the unified virtual address space
  the host and device pointer for accessing the allocation will be the
  same.  The exceptions to this are
   - user memory registered using ::cuMemHostRegister
   - host memory allocated using ::cuMemHostAlloc with the
     ::CU_MEMHOSTALLOC_WRITECOMBINED flag
  For these types of allocation there will exist separate, disjoint host
  and device addresses for accessing the allocation.  In particular
   - The host address will correspond to an invalid unmapped device address
     (which will result in an exception if accessed from the device)
   - The device address will correspond to an invalid unmapped host address
     (which will result in an exception if accessed from the host).
  For these types of allocations, querying ::CU_POINTER_ATTRIBUTE_HOST_POINTER
  and ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER may be used to retrieve the host
  and device addresses from either address.
 
  \param data      - Returned pointer attribute value
  \param attribute - Pointer attribute to query
  \param ptr       - Pointer
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuPointerSetAttribute,
  ::cuMemAlloc,
  ::cuMemFree,
  ::cuMemAllocHost,
  ::cuMemFreeHost,
  ::cuMemHostAlloc,
  ::cuMemHostRegister,
  ::cuMemHostUnregister,
  ::cudaPointerGetAttributes"
  (data (:pointer :void))
  (attribute CUpointer-attribute)
  (ptr CUdeviceptr))

(cffi:defcfun "cumemprefetchasync" CUresult
  "\brief Prefetches memory to the specified destination device
 
  Prefetches memory to the specified destination device.  \p devPtr is the
  base device pointer of the memory to be prefetched and \p dstDevice is the
  destination device. \p count specifies the number of bytes to copy. \p hStream
  is the stream in which the operation is enqueued. The memory range must refer
  to managed memory allocated via ::cuMemAllocManaged or declared via __managed__ variables.
 
  Passing in CU_DEVICE_CPU for \p dstDevice will prefetch the data to host memory. If
  \p dstDevice is a GPU, then the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  must be non-zero. Additionally, \p hStream must be associated with a device that has a
  non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
 
  The start address and end address of the memory range will be rounded down and rounded up
  respectively to be aligned to CPU page size before the prefetch operation is enqueued
  in the stream.
 
  If no physical memory has been allocated for this region, then this memory region
  will be populated and mapped on the destination device. If there's insufficient
  memory to prefetch the desired region, the Unified Memory driver may evict pages from other
  ::cuMemAllocManaged allocations to host memory in order to make room. Device memory
  allocated using ::cuMemAlloc or ::cuArrayCreate will not be evicted.
 
  By default, any mappings to the previous location of the migrated pages are removed and
  mappings for the new location are only setup on \p dstDevice. The exact behavior however
  also depends on the settings applied to this memory range via ::cuMemAdvise as described
  below:
 
  If ::CU_MEM_ADVISE_SET_READ_MOSTLY was set on any subset of this memory range,
  then that subset will create a read-only copy of the pages on \p dstDevice.
 
  If ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION was called on any subset of this memory
  range, then the pages will be migrated to \p dstDevice even if \p dstDevice is not the
  preferred location of any pages in the memory range.
 
  If ::CU_MEM_ADVISE_SET_ACCESSED_BY was called on any subset of this memory range,
  then mappings to those pages from all the appropriate processors are updated to
  refer to the new location if establishing such a mapping is possible. Otherwise,
  those mappings are cleared.
 
  Note that this API is not required for functionality and only serves to improve performance
  by allowing the application to migrate data to a suitable location before it is accessed.
  Memory accesses to this range are always coherent and are allowed even when the data is
  actively being migrated.
 
  Note that this function is asynchronous with respect to the host and all work
  on other devices.
 
  \param devPtr    - Pointer to be prefetched
  \param count     - Size in bytes
  \param dstDevice - Destination device to prefetch to
  \param hStream    - Stream to enqueue prefetch operation
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
  ::cuMemcpy3DPeerAsync, ::cuMemAdvise,
  ::cudaMemPrefetchAsync"
  (devptr CUdeviceptr)
  (count size-t)
  (dstdevice CUdevice)
  (hstream CUstream))

(cffi:defcfun "cumemadvise" CUresult
  "\brief Advise about the usage of a given memory range
 
  Advise the Unified Memory subsystem about the usage pattern for the memory range
  starting at \p devPtr with a size of \p count bytes. The start address and end address of the memory
  range will be rounded down and rounded up respectively to be aligned to CPU page size before the
  advice is applied. The memory range must refer to managed memory allocated via ::cuMemAllocManaged
  or declared via __managed__ variables. The memory range could also refer to system-allocated pageable
  memory provided it represents a valid, host-accessible region of memory and all additional constraints
  imposed by \p advice as outlined below are also satisfied. Specifying an invalid system-allocated pageable
  memory range results in an error being returned.
 
  The \p advice parameter can take the following values:
  - ::CU_MEM_ADVISE_SET_READ_MOSTLY: This implies that the data is mostly going to be read
  from and only occasionally written to. Any read accesses from any processor to this region will create a
  read-only copy of at least the accessed pages in that processor's memory. Additionally, if ::cuMemPrefetchAsync
  is called on this region, it will create a read-only copy of the data on the destination processor.
  If any processor writes to this region, all copies of the corresponding page will be invalidated
  except for the one where the write occurred. The \p device argument is ignored for this advice.
  Note that for a page to be read-duplicated, the accessing processor must either be the CPU or a GPU
  that has a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  Also, if a context is created on a device that does not have the device attribute
  ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS set, then read-duplication will not occur until
  all such contexts are destroyed.
  If the memory region refers to valid system-allocated pageable memory, then the accessing device must
  have a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS for a read-only
  copy to be created on that device. Note however that if the accessing device also has a non-zero value for the
  device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, then setting this advice
  will not create a read-only copy when that device accesses this memory region.
 
  - ::CU_MEM_ADVISE_UNSET_READ_MOSTLY:  Undoes the effect of ::CU_MEM_ADVISE_SET_READ_MOSTLY and also prevents the
  Unified Memory driver from attempting heuristic read-duplication on the memory range. Any read-duplicated
  copies of the data will be collapsed into a single copy. The location for the collapsed
  copy will be the preferred location if the page has a preferred location and one of the read-duplicated
  copies was resident at that location. Otherwise, the location chosen is arbitrary.
 
  - ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION: This advice sets the preferred location for the
  data to be the memory belonging to \p device. Passing in CU_DEVICE_CPU for \p device sets the
  preferred location as host memory. If \p device is a GPU, then it must have a non-zero value for the
  device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. Setting the preferred location
  does not cause data to migrate to that location immediately. Instead, it guides the migration policy
  when a fault occurs on that memory region. If the data is already in its preferred location and the
  faulting processor can establish a mapping without requiring the data to be migrated, then
  data migration will be avoided. On the other hand, if the data is not in its preferred location
  or if a direct mapping cannot be established, then it will be migrated to the processor accessing
  it. It is important to note that setting the preferred location does not prevent data prefetching
  done using ::cuMemPrefetchAsync.
  Having a preferred location can override the page thrash detection and resolution logic in the Unified
  Memory driver. Normally, if a page is detected to be constantly thrashing between for example host and device
  memory, the page may eventually be pinned to host memory by the Unified Memory driver. But
  if the preferred location is set as device memory, then the page will continue to thrash indefinitely.
  If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
  policies associated with that advice will override the policies of this advice, unless read accesses from
  \p device will not result in a read-only copy being created on that device as outlined in description for
  the advice ::CU_MEM_ADVISE_SET_READ_MOSTLY.
  If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
  value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
  a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
  then this call has no effect. Note however that this behavior may change in the future.
 
  - ::CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION: Undoes the effect of ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION
  and changes the preferred location to none.
 
  - ::CU_MEM_ADVISE_SET_ACCESSED_BY: This advice implies that the data will be accessed by \p device.
  Passing in ::CU_DEVICE_CPU for \p device will set the advice for the CPU. If \p device is a GPU, then
  the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS must be non-zero.
  This advice does not cause data migration and has no impact on the location of the data per se. Instead,
  it causes the data to always be mapped in the specified processor's page tables, as long as the
  location of the data permits a mapping to be established. If the data gets migrated for any reason,
  the mappings are updated accordingly.
  This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
  Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
  data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
  over to the other GPUs is not as important because the accesses are infrequent and the overhead of
  migration may be too high. But preventing faults can still help improve performance, and so having
  a mapping set up in advance is useful. Note that on CPU access of this data, the data may be migrated
  to host memory because the CPU typically cannot access device memory directly. Any GPU that had the
  ::CU_MEM_ADVISE_SET_ACCESSED_BY flag set for this data will now have its mapping updated to point to the
  page in host memory.
  If ::CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory region or any subset of it, then the
  policies associated with that advice will override the policies of this advice. Additionally, if the
  preferred location of this memory region or any subset of it is also \p device, then the policies
  associated with ::CU_MEM_ADVISE_SET_PREFERRED_LOCATION will override the policies of this advice.
  If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
  value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
  a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
  then this call has no effect.
 
  - ::CU_MEM_ADVISE_UNSET_ACCESSED_BY: Undoes the effect of ::CU_MEM_ADVISE_SET_ACCESSED_BY. Any mappings to
  the data from \p device may be removed at any time causing accesses to result in non-fatal page faults.
  If the memory region refers to valid system-allocated pageable memory, then \p device must have a non-zero
  value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS. Additionally, if \p device has
  a non-zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
  then this call has no effect.
 
  \param devPtr - Pointer to memory to set the advice for
  \param count  - Size in bytes of the memory range
  \param advice - Advice to be applied for the specified memory range
  \param device - Device to apply the advice for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuMemcpy, ::cuMemcpyPeer, ::cuMemcpyAsync,
  ::cuMemcpy3DPeerAsync, ::cuMemPrefetchAsync,
  ::cudaMemAdvise"
  (devptr CUdeviceptr)
  (count size-t)
  (advice CUmem-advise)
  (device CUdevice))

(cffi:defcfun "cumemrangegetattribute" CUresult
  "\brief Query an attribute of a given memory range
 
  Query an attribute about the memory range starting at \p devPtr with a size of \p count bytes. The
  memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
  __managed__ variables.
 
  The \p attribute parameter can take the following values:
  - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY: If this attribute is specified, \p data will be interpreted
  as a 32-bit integer, and \p dataSize must be 4. The result returned will be 1 if all pages in the given
  memory range have read-duplication enabled, or 0 otherwise.
  - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION: If this attribute is specified, \p data will be
  interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be a GPU device
  id if all pages in the memory range have that GPU as their preferred location, or it will be CU_DEVICE_CPU
  if all pages in the memory range have the CPU as their preferred location, or it will be CU_DEVICE_INVALID
  if either all the pages don't have the same preferred location or some of the pages don't have a
  preferred location at all. Note that the actual location of the pages in the memory range at the time of
  the query may be different from the preferred location.
  - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY: If this attribute is specified, \p data will be interpreted
  as an array of 32-bit integers, and \p dataSize must be a non-zero multiple of 4. The result returned
  will be a list of device ids that had ::CU_MEM_ADVISE_SET_ACCESSED_BY set for that entire memory range.
  If any device does not have that advice set for the entire memory range, that device will not be included.
  If \p data is larger than the number of devices that have that advice set for that memory range,
  CU_DEVICE_INVALID will be returned in all the extra space provided. For ex., if \p dataSize is 12
  (i.e. \p data has 3 elements) and only device 0 has the advice set, then the result returned will be
  { 0, CU_DEVICE_INVALID, CU_DEVICE_INVALID }. If \p data is smaller than the number of devices that have
  that advice set, then only as many devices will be returned as can fit in the array. There is no
  guarantee on which specific devices will be returned, however.
  - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION: If this attribute is specified, \p data will be
  interpreted as a 32-bit integer, and \p dataSize must be 4. The result returned will be the last location
  to which all pages in the memory range were prefetched explicitly via ::cuMemPrefetchAsync. This will either be
  a GPU id or CU_DEVICE_CPU depending on whether the last location for prefetch was a GPU or the CPU
  respectively. If any page in the memory range was never explicitly prefetched or if all pages were not
  prefetched to the same location, CU_DEVICE_INVALID will be returned. Note that this simply returns the
  last location that the applicaton requested to prefetch the memory range to. It gives no indication as to
  whether the prefetch operation to that location has completed or even begun.
 
  \param data      - A pointers to a memory location where the result
                     of each attribute query will be written to.
  \param dataSize  - Array containing the size of data
  \param attribute - The attribute to query
  \param devPtr    - Start of the range to query
  \param count     - Size of the range to query
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
  \note_async
  \note_null_stream
 
  \sa ::cuMemRangeGetAttributes, ::cuMemPrefetchAsync,
  ::cuMemAdvise,
  ::cudaMemRangeGetAttribute"
  (data (:pointer :void))
  (datasize size-t)
  (attribute CUmem-range-attribute)
  (devptr CUdeviceptr)
  (count size-t))

(cffi:defcfun "cumemrangegetattributes" CUresult
  "\brief Query attributes of a given memory range.
 
  Query attributes of the memory range starting at \p devPtr with a size of \p count bytes. The
  memory range must refer to managed memory allocated via ::cuMemAllocManaged or declared via
  __managed__ variables. The \p attributes array will be interpreted to have \p numAttributes
  entries. The \p dataSizes array will also be interpreted to have \p numAttributes entries.
  The results of the query will be stored in \p data.
 
  The list of supported attributes are given below. Please refer to ::cuMemRangeGetAttribute for
  attribute descriptions and restrictions.
 
  - ::CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
  - ::CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
  - ::CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
  - ::CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
 
  \param data          - A two-dimensional array containing pointers to memory
                         locations where the result of each attribute query will be written to.
  \param dataSizes     - Array containing the sizes of each result
  \param attributes    - An array of attributes to query
                         (numAttributes and the number of attributes in this array should match)
  \param numAttributes - Number of attributes to query
  \param devPtr        - Start of the range to query
  \param count         - Size of the range to query
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa ::cuMemRangeGetAttribute, ::cuMemAdvise
  ::cuMemPrefetchAsync,
  ::cudaMemRangeGetAttributes"
  (data (:pointer (:pointer :void)))
  (datasizes (:pointer size-t))
  (attributes (:pointer CUmem-range-attribute))
  (numattributes size-t)
  (devptr CUdeviceptr)
  (count size-t))

(cffi:defcfun "cupointersetattribute" CUresult
  "\brief Set attributes on a previously allocated memory region
 
  The supported attributes are:
 
  - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
 
       A boolean attribute that can either be set (1) or unset (0). When set,
       the region of memory that \p ptr points to is guaranteed to always synchronize
       memory operations that are synchronous. If there are some previously initiated
       synchronous memory operations that are pending when this attribute is set, the
       function does not return until those memory operations are complete.
       See further documentation in the section titled `API synchronization behavior`
       to learn more about cases when synchronous memory operations can
       exhibit asynchronous behavior.
       \p value will be considered as a pointer to an unsigned integer to which this attribute is to be set.
 
  \param value     - Pointer to memory containing the value to be set
  \param attribute - Pointer attribute to set
  \param ptr       - Pointer to a memory region allocated using CUDA memory allocation APIs
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa ::cuPointerGetAttribute,
  ::cuPointerGetAttributes,
  ::cuMemAlloc,
  ::cuMemFree,
  ::cuMemAllocHost,
  ::cuMemFreeHost,
  ::cuMemHostAlloc,
  ::cuMemHostRegister,
  ::cuMemHostUnregister"
  (value (:pointer :void))
  (attribute CUpointer-attribute)
  (ptr CUdeviceptr))

(cffi:defcfun "cuPointerGetAttributes" CUresult
  "\brief Returns information about a pointer.
 
  The supported attributes are (refer to ::cuPointerGetAttribute for attribute descriptions and restrictions):
 
  - ::CU_POINTER_ATTRIBUTE_CONTEXT
  - ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE
  - ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER
  - ::CU_POINTER_ATTRIBUTE_HOST_POINTER
  - ::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
  - ::CU_POINTER_ATTRIBUTE_BUFFER_ID
  - ::CU_POINTER_ATTRIBUTE_IS_MANAGED
  - ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
 
  \param numAttributes - Number of attributes to query
  \param attributes    - An array of attributes to query
                       (numAttributes and the number of attributes in this array should match)
  \param data          - A two-dimensional array containing pointers to memory
                       locations where the result of each attribute query will be written to.
  \param ptr           - Pointer to query
 
  Unlike ::cuPointerGetAttribute, this function will not return an error when the \p ptr
  encountered is not a valid CUDA pointer. Instead, the attributes are assigned default NULL values
  and CUDA_SUCCESS is returned.
 
  If \p ptr was not allocated by, mapped by, or registered with a ::CUcontext which uses UVA
  (Unified Virtual Addressing), ::CUDA_ERROR_INVALID_CONTEXT is returned.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuPointerGetAttribute,
  ::cuPointerSetAttribute,
  ::cudaPointerGetAttributes"
  (numattributes :unsigned-int)
  (attributes (:pointer CUpointer-attribute))
  (data (:pointer (:pointer :void)))
  (ptr CUdeviceptr))

(cffi:defcfun "custreamcreate" CUresult
  "\brief Create a stream
 
  Creates a stream and returns a handle in \p phStream.  The \p Flags argument
  determines behaviors of the stream.  Valid values for \p Flags are:
  - ::CU_STREAM_DEFAULT: Default stream creation flag.
  - ::CU_STREAM_NON_BLOCKING: Specifies that work running in the created
    stream may run concurrently with work in stream 0 (the NULL stream), and that
    the created stream should perform no implicit synchronization with stream 0.
 
  \param phStream - Returned newly created stream
  \param Flags    - Parameters for stream creation
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa ::cuStreamDestroy,
  ::cuStreamCreateWithPriority,
  ::cuStreamGetPriority,
  ::cuStreamGetFlags,
  ::cuStreamWaitEvent,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamAddCallback,
  ::cudaStreamCreate,
  ::cudaStreamCreateWithFlags"
  (phstream (:pointer CUstream))
  (flags :unsigned-int))

(cffi:defcfun "custreamcreatewithpriority" CUresult
  "\brief Create a stream with the given priority
 
  Creates a stream with the specified priority and returns a handle in \p phStream.
  This API alters the scheduler priority of work in the stream. Work in a higher
  priority stream may preempt work already executing in a low priority stream.
 
  \p priority follows a convention where lower numbers represent higher priorities.
  '0' represents default priority. The range of meaningful numerical priorities can
  be queried using ::cuCtxGetStreamPriorityRange. If the specified priority is
  outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
  it will automatically be clamped to the lowest or the highest number in the range.
 
  \param phStream    - Returned newly created stream
  \param flags       - Flags for stream creation. See ::cuStreamCreate for a list of
                       valid flags
  \param priority    - Stream priority. Lower numbers represent higher priorities.
                       See ::cuCtxGetStreamPriorityRange for more information about
                       meaningful stream priorities that can be passed.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \note Stream priorities are supported only on GPUs
  with compute capability 3.5 or higher.
 
  \note In the current implementation, only compute kernels launched in
  priority streams are affected by the stream's priority. Stream priorities have
  no effect on host-to-device and device-to-host memory operations.
 
  \sa ::cuStreamDestroy,
  ::cuStreamCreate,
  ::cuStreamGetPriority,
  ::cuCtxGetStreamPriorityRange,
  ::cuStreamGetFlags,
  ::cuStreamWaitEvent,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamAddCallback,
  ::cudaStreamCreateWithPriority"
  (phstream (:pointer CUstream))
  (flags :unsigned-int)
  (priority :int))

(cffi:defcfun "custreamgetpriority" CUresult
  "\brief Query the priority of a given stream
 
  Query the priority of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
  and return the priority in \p priority. Note that if the stream was created with a
  priority outside the numerical range returned by ::cuCtxGetStreamPriorityRange,
  this function returns the clamped priority.
  See ::cuStreamCreateWithPriority for details about priority clamping.
 
  \param hStream    - Handle to the stream to be queried
  \param priority   - Pointer to a signed integer in which the stream's priority is returned
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa ::cuStreamDestroy,
  ::cuStreamCreate,
  ::cuStreamCreateWithPriority,
  ::cuCtxGetStreamPriorityRange,
  ::cuStreamGetFlags,
  ::cudaStreamGetPriority"
  (hstream CUstream)
  (priority (:pointer :int)))

(cffi:defcfun "custreamgetflags" CUresult
  "\brief Query the flags of a given stream
 
  Query the flags of a stream created using ::cuStreamCreate or ::cuStreamCreateWithPriority
  and return the flags in \p flags.
 
  \param hStream    - Handle to the stream to be queried
  \param flags      - Pointer to an unsigned integer in which the stream's flags are returned
                      The value returned in \p flags is a logical 'OR' of all flags that
                      were used while creating this stream. See ::cuStreamCreate for the list
                      of valid flags
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa ::cuStreamDestroy,
  ::cuStreamCreate,
  ::cuStreamGetPriority,
  ::cudaStreamGetFlags"
  (hstream CUstream)
  (flags (:pointer :unsigned-int)))

(cffi:defcfun "custreamgetctx" CUresult
  "\brief Query the context associated with a stream
 
  Returns the CUDA context that the stream is associated with.
 
  The stream handle \p hStream can refer to any of the following:
  <ul>
    <li>a stream created via any of the CUDA driver APIs such as ::cuStreamCreate
    and ::cuStreamCreateWithPriority, or their runtime API equivalents such as
    ::cudaStreamCreate, ::cudaStreamCreateWithFlags and ::cudaStreamCreateWithPriority.
    The returned context is the context that was active in the calling thread when the
    stream was created. Passing an invalid handle will result in undefined behavior.<li>
    <li>any of the special streams such as the NULL stream, ::CU_STREAM_LEGACY and
    ::CU_STREAM_PER_THREAD. The runtime API equivalents of these are also accepted,
    which are NULL, ::cudaStreamLegacy and ::cudaStreamPerThread respectively.
    Specifying any of the special handles will return the context current to the
    calling thread. If no context is current to the calling thread,
    ::CUDA_ERROR_INVALID_CONTEXT is returned.<li>
  <ul>
 
  \param hStream - Handle to the stream to be queried
  \param pctx    - Returned context associated with the stream
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  \notefnerr
 
  \sa ::cuStreamDestroy,
  ::cuStreamCreateWithPriority,
  ::cuStreamGetPriority,
  ::cuStreamGetFlags,
  ::cuStreamWaitEvent,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamAddCallback,
  ::cudaStreamCreate,
  ::cudaStreamCreateWithFlags"
  (hstream CUstream)
  (pctx (:pointer CUcontext)))

(cffi:defcfun "custreamwaitevent" CUresult
  "\brief Make a compute stream wait on an event
 
  Makes all future work submitted to \p hStream wait for all work captured in
  \p hEvent.  See ::cuEventRecord() for details on what is captured by an event.
  The synchronization will be performed efficiently on the device when applicable.
  \p hEvent may be from a different context or device than \p hStream.
 
  \param hStream - Stream to wait
  \param hEvent  - Event to wait on (may not be NULL)
  \param Flags   - Parameters for the operation (must be 0)
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuEventRecord,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamAddCallback,
  ::cuStreamDestroy,
  ::cudaStreamWaitEvent"
  (hstream CUstream)
  (hevent CUevent)
  (flags :unsigned-int))

(cffi:defcfun "custreamaddcallback" CUresult
  "\brief Add a callback to a compute stream
 
  \note This function is slated for eventual deprecation and removal. If
  you do not require the callback to execute in case of a device error,
  consider using ::cuLaunchHostFunc. Additionally, this function is not
  supported with ::cuStreamBeginCapture and ::cuStreamEndCapture, unlike
  ::cuLaunchHostFunc.
 
  Adds a callback to be called on the host after all currently enqueued
  items in the stream have completed.  For each
  cuStreamAddCallback call, the callback will be executed exactly once.
  The callback will block later work in the stream until it is finished.
 
  The callback may be passed ::CUDA_SUCCESS or an error code.  In the event
  of a device error, all subsequently executed callbacks will receive an
  appropriate ::CUresult.
 
  Callbacks must not make any CUDA API calls.  Attempting to use a CUDA API
  will result in ::CUDA_ERROR_NOT_PERMITTED.  Callbacks must not perform any
  synchronization that may depend on outstanding device work or other callbacks
  that are not mandated to run earlier.  Callbacks without a mandated order
  (in independent streams) execute in undefined order and may be serialized.
 
  For the purposes of Unified Memory, callback execution makes a number of
  guarantees:
  <ul>
    <li>The callback stream is considered idle for the duration of the
    callback.  Thus, for example, a callback may always use memory attached
    to the callback stream.<li>
    <li>The start of execution of a callback has the same effect as
    synchronizing an event recorded in the same stream immediately prior to
    the callback.  It thus synchronizes streams which have been joined
    prior to the callback.<li>
    <li>Adding device work to any stream does not have the effect of making
    the stream active until all preceding host functions and stream callbacks
    have executed.  Thus, for
    example, a callback might use global attached memory even if work has
    been added to another stream, if the work has been ordered behind the
    callback with an event.<li>
    <li>Completion of a callback does not cause a stream to become
    active except as described above.  The callback stream will remain idle
    if no device work follows the callback, and will remain idle across
    consecutive callbacks without device work in between.  Thus, for example,
    stream synchronization can be done by signaling from a callback at the
    end of the stream.<li>
  <ul>
 
  \param hStream  - Stream to add callback to
  \param callback - The function to call once preceding stream operations are complete
  \param userData - User specified data to be passed to the callback function
  \param flags    - Reserved for future use, must be 0
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamWaitEvent,
  ::cuStreamDestroy,
  ::cuMemAllocManaged,
  ::cuStreamAttachMemAsync,
  ::cuStreamLaunchHostFunc,
  ::cudaStreamAddCallback"
  (hstream CUstream)
  (callback CUstreamCallback)
  (userdata (:pointer :void))
  (flags :unsigned-int))

(cffi:defcfun ("custreambegincapture_v2" custreambegincapture-v2) CUresult
  (hstream CUstream)
  (mode CUstreamCaptureMode))

(cffi:defcfun "cuthreadexchangestreamcapturemode" CUresult
  "\brief Swaps the stream capture interaction mode for a thread
 
  Sets the calling thread's stream capture interaction mode to the value contained
  in \p mode, and overwrites \p mode with the previous mode for the thread. To
  facilitate deterministic behavior across function or module boundaries, callers
  are encouraged to use this API in a push-pop fashion: \code
     CUstreamCaptureMode mode = desiredMode;
     cuThreadExchangeStreamCaptureMode(&mode);
     ...
     cuThreadExchangeStreamCaptureMode(&mode);  restore previous mode
  \endcode
 
  During stream capture (see ::cuStreamBeginCapture), some actions, such as a call
  to ::cudaMalloc, may be unsafe. In the case of ::cudaMalloc, the operation is
  not enqueued asynchronously to a stream, and is not observed by stream capture.
  Therefore, if the sequence of operations captured via ::cuStreamBeginCapture
  depended on the allocation being replayed whenever the graph is launched, the
  captured graph would be invalid.
 
  Therefore, stream capture places restrictions on API calls that can be made within
  or concurrently to a ::cuStreamBeginCapture-::cuStreamEndCapture sequence. This
  behavior can be controlled via this API and flags to ::cuStreamBeginCapture.
 
  A thread's mode is one of the following:
  - \p CU_STREAM_CAPTURE_MODE_GLOBAL: This is the default mode. If the local thread has
    an ongoing capture sequence that was not initiated with
    \p CU_STREAM_CAPTURE_MODE_RELAXED at \p cuStreamBeginCapture, or if any other thread
    has a concurrent capture sequence initiated with \p CU_STREAM_CAPTURE_MODE_GLOBAL,
    this thread is prohibited from potentially unsafe API calls.
  - \p CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: If the local thread has an ongoing capture
    sequence not initiated with \p CU_STREAM_CAPTURE_MODE_RELAXED, it is prohibited
    from potentially unsafe API calls. Concurrent capture sequences in other threads
    are ignored.
  - \p CU_STREAM_CAPTURE_MODE_RELAXED: The local thread is not prohibited from potentially
    unsafe API calls. Note that the thread is still prohibited from API calls which
    necessarily conflict with stream capture, for example, attempting ::cuEventQuery
    on an event that was last recorded inside a capture sequence.
 
  \param mode - Pointer to mode value to swap with the current mode
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cuStreamBeginCapture"
  (mode (:pointer CUstreamCaptureMode)))

(cffi:defcfun "custreamendcapture" CUresult
  "\brief Ends capture on a stream, returning the captured graph
 
  End capture on \p hStream, returning the captured graph via \p phGraph.
  Capture must have been initiated on \p hStream via a call to ::cuStreamBeginCapture.
  If capture was invalidated, due to a violation of the rules of stream capture, then
  a NULL graph will be returned.
 
  If the \p mode argument to ::cuStreamBeginCapture was not
  ::CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as
  ::cuStreamBeginCapture.
 
  \param hStream - Stream to query
  \param phGraph - The captured graph
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD
  \notefnerr
 
  \sa
  ::cuStreamCreate,
  ::cuStreamBeginCapture,
  ::cuStreamIsCapturing"
  (hstream CUstream)
  (phgraph (:pointer CUgraph)))

(cffi:defcfun "custreamiscapturing" CUresult
  "\brief Returns a stream's capture status
 
  Return the capture status of \p hStream via \p captureStatus. After a successful
  call, \p captureStatus will contain one of the following:
  - ::CU_STREAM_CAPTURE_STATUS_NONE: The stream is not capturing.
  - ::CU_STREAM_CAPTURE_STATUS_ACTIVE: The stream is capturing.
  - ::CU_STREAM_CAPTURE_STATUS_INVALIDATED: The stream was capturing but an error
    has invalidated the capture sequence. The capture sequence must be terminated
    with ::cuStreamEndCapture on the stream where it was initiated in order to
    continue using \p hStream.
 
  Note that, if this is called on ::CU_STREAM_LEGACY (the null stream) while
  a blocking stream in the same context is capturing, it will return
  ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT and \p captureStatus is unspecified
  after the call. The blocking stream capture is not invalidated.
 
  When a blocking stream is capturing, the legacy stream is in an
  unusable state until the blocking stream capture is terminated. The legacy
  stream is not supported for stream capture, but attempted use would have an
  implicit dependency on the capturing stream(s).
 
  \param hStream       - Stream to query
  \param captureStatus - Returns the stream's capture status
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
  \notefnerr
 
  \sa
  ::cuStreamCreate,
  ::cuStreamBeginCapture,
  ::cuStreamEndCapture"
  (hstream CUstream)
  (capturestatus (:pointer CUstreamCaptureStatus)))

(cffi:defcfun "custreamgetcaptureinfo" CUresult
  "\brief Query capture status of a stream
 
  Query the capture status of a stream and and get an id for
  the capture sequence, which is unique over the lifetime of the process.
 
  If called on ::CU_STREAM_LEGACY (the null stream) while a stream not created
  with ::CU_STREAM_NON_BLOCKING is capturing, returns ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT.
 
  A valid id is returned only if both of the following are true:
  - the call returns CUDA_SUCCESS
  - captureStatus is set to ::CU_STREAM_CAPTURE_STATUS_ACTIVE
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT
  \notefnerr
 
  \sa
  ::cuStreamBeginCapture,
  ::cuStreamIsCapturing"
  (hstream CUstream)
  (capturestatus (:pointer CUstreamCaptureStatus))
  (id (:pointer cuuint64-t)))

(cffi:defcfun "custreamattachmemasync" CUresult
  "\brief Attach memory to a stream asynchronously
 
  Enqueues an operation in \p hStream to specify stream association of
  \p length bytes of memory starting from \p dptr. This function is a
  stream-ordered operation, meaning that it is dependent on, and will
  only take effect when, previous work in stream has completed. Any
  previous association is automatically replaced.
 
  \p dptr must point to one of the following types of memories:
  - managed memory declared using the __managed__ keyword or allocated with
    ::cuMemAllocManaged.
  - a valid host-accessible region of system-allocated pageable memory. This
    type of memory may only be specified if the device associated with the
    stream reports a non-zero value for the device attribute
    ::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS.
 
  For managed allocations, \p length must be either zero or the entire
  allocation's size. Both indicate that the entire allocation's stream
  association is being changed. Currently, it is not possible to change stream
  association for a portion of a managed allocation.
 
  For pageable host allocations, \p length must be non-zero.
 
  The stream association is specified using \p flags which must be
  one of ::CUmemAttach_flags.
  If the ::CU_MEM_ATTACH_GLOBAL flag is specified, the memory can be accessed
  by any stream on any device.
  If the ::CU_MEM_ATTACH_HOST flag is specified, the program makes a guarantee
  that it won't access the memory on the device from any stream on a device that
  has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS.
  If the ::CU_MEM_ATTACH_SINGLE flag is specified and \p hStream is associated with
  a device that has a zero value for the device attribute ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
  the program makes a guarantee that it will only access the memory on the device
  from \p hStream. It is illegal to attach singly to the NULL stream, because the
  NULL stream is a virtual global stream and not a specific stream. An error will
  be returned in this case.
 
  When memory is associated with a single stream, the Unified Memory system will
  allow CPU access to this memory region so long as all operations in \p hStream
  have completed, regardless of whether other streams are active. In effect,
  this constrains exclusive ownership of the managed memory region by
  an active GPU to per-stream activity instead of whole-GPU activity.
 
  Accessing memory on the device from streams that are not associated with
  it will produce undefined results. No error checking is performed by the
  Unified Memory system to ensure that kernels launched into other streams
  do not access this region.
 
  It is a program's responsibility to order calls to ::cuStreamAttachMemAsync
  via events, synchronization or other means to ensure legal access to memory
  at all times. Data visibility and coherency will be changed appropriately
  for all kernels which follow a stream-association change.
 
  If \p hStream is destroyed while data is associated with it, the association is
  removed and the association reverts to the default visibility of the allocation
  as specified at ::cuMemAllocManaged. For __managed__ variables, the default
  association is always ::CU_MEM_ATTACH_GLOBAL. Note that destroying a stream is an
  asynchronous operation, and as a result, the change to default association won't
  happen until all work in the stream has completed.
 
  \param hStream - Stream in which to enqueue the attach operation
  \param dptr    - Pointer to memory (must be a pointer to managed memory or
                   to a valid host-accessible region of system-allocated
                   pageable memory)
  \param length  - Length of memory
  \param flags   - Must be one of ::CUmemAttach_flags
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamWaitEvent,
  ::cuStreamDestroy,
  ::cuMemAllocManaged,
  ::cudaStreamAttachMemAsync"
  (hstream CUstream)
  (dptr CUdeviceptr)
  (length size-t)
  (flags :unsigned-int))

(cffi:defcfun "custreamquery" CUresult
  "\brief Determine status of a compute stream
 
  Returns ::CUDA_SUCCESS if all operations in the stream specified by
  \p hStream have completed, or ::CUDA_ERROR_NOT_READY if not.
 
  For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
  is equivalent to having called ::cuStreamSynchronize().
 
  \param hStream - Stream to query status of
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_READY
  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuStreamWaitEvent,
  ::cuStreamDestroy,
  ::cuStreamSynchronize,
  ::cuStreamAddCallback,
  ::cudaStreamQuery"
  (hstream CUstream))

(cffi:defcfun "cuStreamSynchronize" CUresult
  "\brief Wait until a stream's tasks are completed
 
  Waits until the device has completed all operations in the stream specified
  by \p hStream. If the context was created with the
  ::CU_CTX_SCHED_BLOCKING_SYNC flag, the CPU thread will block until the
  stream is finished with all of its tasks.
 
  \param hStream - Stream to wait for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE

  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuStreamDestroy,
  ::cuStreamWaitEvent,
  ::cuStreamQuery,
  ::cuStreamAddCallback,
  ::cudaStreamSynchronize"
  (hstream CUstream))

(cffi:defcfun ("cuStreamDestroy_v2" custreamdestroy-v2) CUresult
  (hstream CUstream))

(cffi:defcfun "cuEventCreate" CUresult
  "\brief Creates an event
 
  Creates an event phEvent for the current context with the flags specified via
  \p Flags. Valid flags include:
  - ::CU_EVENT_DEFAULT: Default event creation flag.
  - ::CU_EVENT_BLOCKING_SYNC: Specifies that the created event should use blocking
    synchronization.  A CPU thread that uses ::cuEventSynchronize() to wait on
    an event created with this flag will block until the event has actually
    been recorded.
  - ::CU_EVENT_DISABLE_TIMING: Specifies that the created event does not need
    to record timing data.  Events created with this flag specified and
    the ::CU_EVENT_BLOCKING_SYNC flag not specified will provide the best
    performance when used with ::cuStreamWaitEvent() and ::cuEventQuery().
  - ::CU_EVENT_INTERPROCESS: Specifies that the created event may be used as an
    interprocess event by ::cuIpcGetEventHandle(). ::CU_EVENT_INTERPROCESS must
    be specified along with ::CU_EVENT_DISABLE_TIMING.
 
  \param phEvent - Returns newly created event
  \param Flags   - Event creation flags
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \notefnerr
 
  \sa
  ::cuEventRecord,
  ::cuEventQuery,
  ::cuEventSynchronize,
  ::cuEventDestroy,
  ::cuEventElapsedTime,
  ::cudaEventCreate,
  ::cudaEventCreateWithFlags"
  (phevent (:pointer CUevent))
  (flags :unsigned-int))

(cffi:defcfun "cuEventRecord" CUresult
  "\brief Records an event
 
  Captures in \p hEvent the contents of \p hStream at the time of this call.
  \p hEvent and \p hStream must be from the same context.
  Calls such as ::cuEventQuery() or ::cuStreamWaitEvent() will then
  examine or wait for completion of the work that was captured. Uses of
  \p hStream after this call do not modify \p hEvent. See note on default
  stream behavior for what is captured in the default case.
 
  ::cuEventRecord() can be called multiple times on the same event and
  will overwrite the previously captured state. Other APIs such as
  ::cuStreamWaitEvent() use the most recently captured state at the time
  of the API call, and are not affected by later calls to
  ::cuEventRecord(). Before the first call to ::cuEventRecord(), an
  event represents an empty set of work, so for example ::cuEventQuery()
  would return ::CUDA_SUCCESS.
 
  \param hEvent  - Event to record
  \param hStream - Stream to record event for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \note_null_stream
  \notefnerr
 
  \sa ::cuEventCreate,
  ::cuEventQuery,
  ::cuEventSynchronize,
  ::cuStreamWaitEvent,
  ::cuEventDestroy,
  ::cuEventElapsedTime,
  ::cudaEventRecord"
  (hevent CUevent)
  (hstream CUstream))

(cffi:defcfun "cueventquery" CUresult
  "\brief Queries an event's status
 
  Queries the status of all work currently captured by \p hEvent. See
  ::cuEventRecord() for details on what is captured by an event.
 
  Returns ::CUDA_SUCCESS if all captured work has been completed, or
  ::CUDA_ERROR_NOT_READY if any captured work is incomplete.
 
  For the purposes of Unified Memory, a return value of ::CUDA_SUCCESS
  is equivalent to having called ::cuEventSynchronize().
 
  \param hEvent - Event to query
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_READY
  \notefnerr
 
  \sa ::cuEventCreate,
  ::cuEventRecord,
  ::cuEventSynchronize,
  ::cuEventDestroy,
  ::cuEventElapsedTime,
  ::cudaEventQuery"
  (hevent CUevent))

(cffi:defcfun "cuEventSynchronize" CUresult
  "\brief Waits for an event to complete
 
  Waits until the completion of all work currently captured in \p hEvent.
  See ::cuEventRecord() for details on what is captured by an event.
 
  Waiting for an event that was created with the ::CU_EVENT_BLOCKING_SYNC
  flag will cause the calling CPU thread to block until the event has
  been completed by the device.  If the ::CU_EVENT_BLOCKING_SYNC flag has
  not been set, then the CPU thread will busy-wait until the event has
  been completed by the device.
 
  \param hEvent - Event to wait for
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuEventCreate,
  ::cuEventRecord,
  ::cuEventQuery,
  ::cuEventDestroy,
  ::cuEventElapsedTime,
  ::cudaEventSynchronize"
  (hevent CUevent))

(cffi:defcfun ("cuEventDestroy_v2" cueventdestroy-v2) CUresult
  (hevent CUevent))

(cffi:defcfun "cueventelapsedtime" CUresult
  "\brief Computes the elapsed time between two events
 
  Computes the elapsed time between two events (in milliseconds with a
  resolution of around 0.5 microseconds).
 
  If either event was last recorded in a non-NULL stream, the resulting time
  may be greater than expected (even if both used the same stream handle). This
  happens because the ::cuEventRecord() operation takes place asynchronously
  and there is no guarantee that the measured latency is actually just between
  the two events. Any number of other different stream operations could execute
  in between the two measured events, thus altering the timing in a significant
  way.
 
  If ::cuEventRecord() has not been called on either event then
  ::CUDA_ERROR_INVALID_HANDLE is returned. If ::cuEventRecord() has been called
  on both events but one or both of them has not yet been completed (that is,
  ::cuEventQuery() would return ::CUDA_ERROR_NOT_READY on at least one of the
  events), ::CUDA_ERROR_NOT_READY is returned. If either event was created with
  the ::CU_EVENT_DISABLE_TIMING flag, then this function will return
  ::CUDA_ERROR_INVALID_HANDLE.
 
  \param pMilliseconds - Time between \p hStart and \p hEnd in ms
  \param hStart        - Starting event
  \param hEnd          - Ending event
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_READY
  \notefnerr
 
  \sa ::cuEventCreate,
  ::cuEventRecord,
  ::cuEventQuery,
  ::cuEventSynchronize,
  ::cuEventDestroy,
  ::cudaEventElapsedTime"
  (pmilliseconds (:pointer :float))
  (hstart CUevent)
  (hend CUevent))

(cffi:defcfun "cuimportexternalmemory" CUresult
  "\brief Imports an external memory object
 
  Imports an externally allocated memory object and returns
  a handle to that in \p extMem_out.
 
  The properties of the handle being imported must be described in
  \p memHandleDesc. The ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC structure
  is defined as follows:
 
  \code
        typedef struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
            CUexternalMemoryHandleType type;
            union {
                int fd;
                struct {
                    void handle;
                    const void name;
                } win32;
            } handle;
            unsigned long long size;
            unsigned int flags;
        } CUDA_EXTERNAL_MEMORY_HANDLE_DESC;
  \endcode
 
  where ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type specifies the type
  of handle being imported. ::CUexternalMemoryHandleType is
  defined as:
 
  \code
        typedef enum CUexternalMemoryHandleType_enum {
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD        = 1,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32     = 2,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP       = 4,
            CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE   = 5
        } CUexternalMemoryHandleType;
  \endcode
 
  If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
  ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD, then
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::fd must be a valid
  file descriptor referencing a memory object. Ownership of
  the file descriptor is transferred to the CUDA driver when the
  handle is imported successfully. Performing any operations on the
  file descriptor after it is imported results in undefined behavior.
 
  If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
  ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
  of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
  NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
  is not NULL, then it must represent a valid shared NT handle that
  references a memory object. Ownership of this handle is
  not transferred to CUDA after the import operation, so the
  application must release the handle using the appropriate system
  call. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
  is not NULL, then it must point to a NULL-terminated array of
  UTF-16 characters that refers to a memory object.
 
  If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
  ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT, then
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle must
  be non-NULL and
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
  must be NULL. The handle specified must be a globally shared KMT
  handle. This handle does not hold a reference to the underlying
  object, and thus will be invalid when all references to the
  memory object are destroyed.
 
  If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
  ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP, then exactly one
  of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
  NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
  is not NULL, then it must represent a valid shared NT handle that
  is returned by ID3DDevice::CreateSharedHandle when referring to a
  ID3D12Heap object. This handle holds a reference to the underlying
  object. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
  is not NULL, then it must point to a NULL-terminated array of
  UTF-16 characters that refers to a ID3D12Heap object.
 
  If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::type is
  ::CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE, then exactly one
  of ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle and
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name must not be
  NULL. If ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::handle
  is not NULL, then it must represent a valid shared NT handle that
  is returned by ID3DDevice::CreateSharedHandle when referring to a
  ID3D12Resource object. This handle holds a reference to the
  underlying object. If
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::handle::win32::name
  is not NULL, then it must point to a NULL-terminated array of
  UTF-16 characters that refers to a ID3D12Resource object.
 
  The size of the memory object must be specified in
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::size.
 
  Specifying the flag ::CUDA_EXTERNAL_MEMORY_DEDICATED in
  ::CUDA_EXTERNAL_MEMORY_HANDLE_DESC::flags indicates that the
  resource is a dedicated resource. The definition of what a
  dedicated resource is outside the scope of this extension.
 
  \param extMem_out    - Returned handle to an external memory object
  \param memHandleDesc - Memory import handle descriptor
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \note If the Vulkan memory imported into CUDA is mapped on the CPU then the
  application must use vkInvalidateMappedMemoryRangesvkFlushMappedMemoryRanges
  as well as appropriate Vulkan pipeline barriers to maintain coherence between
  CPU and GPU. For more information on these APIs, please refer to `Synchronization
  and Cache Control` chapter from Vulkan specification.
 
  \sa ::cuDestroyExternalMemory,
  ::cuExternalMemoryGetMappedBuffer,
  ::cuExternalMemoryGetMappedMipmappedArray"
  (extmem-out (:pointer CUexternalMemory))
  (memhandledesc (:pointer CUDA-EXTERNAL-MEMORY-HANDLE-DESC)))

(cffi:defcfun "cuexternalmemorygetmappedbuffer" CUresult
  "\brief Maps a buffer onto an imported memory object
 
  Maps a buffer onto an imported memory object and returns a device
  pointer in \p devPtr.
 
  The properties of the buffer being mapped must be described in
  \p bufferDesc. The ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC structure is
  defined as follows:
 
  \code
        typedef struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
            unsigned long long offset;
            unsigned long long size;
            unsigned int flags;
        } CUDA_EXTERNAL_MEMORY_BUFFER_DESC;
  \endcode
 
  where ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::offset is the offset in
  the memory object where the buffer's base address is.
  ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::size is the size of the buffer.
  ::CUDA_EXTERNAL_MEMORY_BUFFER_DESC::flags must be zero.
 
  The offset and size have to be suitably aligned to match the
  requirements of the external API. Mapping two buffers whose ranges
  overlap may or may not result in the same virtual address being
  returned for the overlapped portion. In such cases, the application
  must ensure that all accesses to that region from the GPU are
  volatile. Otherwise writes made via one address are not guaranteed
  to be visible via the other address, even if they're issued by the
  same thread. It is recommended that applications map the combined
  range instead of mapping separate buffers and then apply the
  appropriate offsets to the returned pointer to derive the
  individual buffers.
 
  The returned pointer \p devPtr must be freed using ::cuMemFree.
 
  \param devPtr     - Returned device pointer to buffer
  \param extMem     - Handle to external memory object
  \param bufferDesc - Buffer descriptor
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuImportExternalMemory
  ::cuDestroyExternalMemory,
  ::cuExternalMemoryGetMappedMipmappedArray"
  (devptr (:pointer CUdeviceptr))
  (extmem CUexternalMemory)
  (bufferdesc (:pointer)))

(cffi:defcfun "cuexternalmemorygetmappedmipmappedarray" CUresult
  ""
  (mipmap (:pointer CUmipmappedArray))
  (extmem CUexternalMemory)
  (mipmapdesc (:pointer)))

(cffi:defcfun "cudestroyexternalmemory" CUresult
  "\brief Destroys an external memory object.
 
  Destroys the specified external memory object. Any existing buffers
  and CUDA mipmapped arrays mapped onto this object must no longer be
  used and must be explicitly freed using ::cuMemFree and
  ::cuMipmappedArrayDestroy respectively.
 
  \param extMem - External memory object to be destroyed
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuImportExternalMemory
  ::cuExternalMemoryGetMappedBuffer,
  ::cuExternalMemoryGetMappedMipmappedArray"
  (extmem CUexternalMemory))

(cffi:defcfun "cuimportexternalsemaphore" CUresult
  "\brief Imports an external semaphore
 
  Imports an externally allocated synchronization object and returns
  a handle to that in \p extSem_out.
 
  The properties of the handle being imported must be described in
  \p semHandleDesc. The ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC is
  defined as follows:
 
  \code
        typedef struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
            CUexternalSemaphoreHandleType type;
            union {
                int fd;
                struct {
                    void handle;
                    const void name;
                } win32;
            } handle;
            unsigned int flags;
        } CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC;
  \endcode
 
  where ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type specifies the type of
  handle being imported. ::CUexternalSemaphoreHandleType is defined
  as:
 
  \code
        typedef enum CUexternalSemaphoreHandleType_enum {
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD        = 1,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32     = 2,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
            CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE      = 4
        } CUexternalSemaphoreHandleType;
  \endcode
 
  If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD, then
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::fd must be a valid
  file descriptor referencing a synchronization object. Ownership of
  the file descriptor is transferred to the CUDA driver when the
  handle is imported successfully. Performing any operations on the
  file descriptor after it is imported results in undefined behavior.
 
  If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32, then exactly one
  of ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
  NULL. If
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
  is not NULL, then it must represent a valid shared NT handle that
  references a synchronization object. Ownership of this handle is
  not transferred to CUDA after the import operation, so the
  application must release the handle using the appropriate system
  call. If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
  is not NULL, then it must name a valid synchronization object.
 
  If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT, then
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle must
  be non-NULL and
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
  must be NULL. The handle specified must be a globally shared KMT
  handle. This handle does not hold a reference to the underlying
  object, and thus will be invalid when all references to the
  synchronization object are destroyed.
 
  If ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::type is
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE, then exactly one
  of ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle and
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name must not be
  NULL. If
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::handle
  is not NULL, then it must represent a valid shared NT handle that
  is returned by ID3DDevice::CreateSharedHandle when referring to a
  ID3D12Fence object. This handle holds a reference to the underlying
  object. If
  ::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC::handle::win32::name
  is not NULL, then it must name a valid synchronization object that
  refers to a valid ID3D12Fence object.
 
  \param extSem_out    - Returned handle to an external semaphore
  \param semHandleDesc - Semaphore import handle descriptor
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuDestroyExternalSemaphore,
  ::cuSignalExternalSemaphoresAsync,
  ::cuWaitExternalSemaphoresAsync"
  (extsem-out (:pointer CUexternalSemaphore))
  (semhandledesc (:pointer CUDA-EXTERNAL-SEMAPHORE-HANDLE-DESC)))

(cffi:defcfun "cusignalexternalsemaphoresasync" CUresult
  "\brief Signals a set of external semaphore objects
 
  Enqueues a signal operation on a set of externally allocated
  semaphore object in the specified stream. The operations will be
  executed when all prior operations in the stream complete.
 
  The exact semantics of signaling a semaphore depends on the type of
  the object.
 
  If the semaphore object is any one of the following types:
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
  then signaling the semaphore will set it to the signaled state.
 
  If the semaphore object is of the type
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE, then the
  semaphore will be set to the value specified in
  ::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS::params::fence::value.
 
  \param extSemArray - Set of external semaphores to be signaled
  \param paramsArray - Array of semaphore parameters
  \param numExtSems  - Number of semaphores to signal
  \param stream     - Stream to enqueue the signal operations in
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuImportExternalSemaphore,
  ::cuDestroyExternalSemaphore,
  ::cuWaitExternalSemaphoresAsync"
  (extsemarray (:pointer CUexternalSemaphore))
  (paramsarray (:pointer CUDA-EXTERNAL-SEMAPHORE-SIGNAL-PARAMS))
  (numextsems :unsigned-int)
  (stream CUstream))

(cffi:defcfun "cuwaitexternalsemaphoresasync" CUresult
  "\brief Waits on a set of external semaphore objects
 
  Enqueues a wait operation on a set of externally allocated
  semaphore object in the specified stream. The operations will be
  executed when all prior operations in the stream complete.
 
  The exact semantics of waiting on a semaphore depends on the type
  of the object.
 
  If the semaphore object is any one of the following types:
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD,
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32,
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
  then waiting on the semaphore will wait until the semaphore reaches
  the signaled state. The semaphore will then be reset to the
  unsignaled state. Therefore for every signal operation, there can
  only be one wait operation.
 
  If the semaphore object is of the type
  ::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE, then waiting on
  the semaphore will wait until the value of the semaphore is
  greater than or equal to
  ::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS::params::fence::value.
 
  \param extSemArray - External semaphores to be waited on
  \param paramsArray - Array of semaphore parameters
  \param numExtSems  - Number of semaphores to wait on
  \param stream      - Stream to enqueue the wait operations in
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuImportExternalSemaphore,
  ::cuDestroyExternalSemaphore,
  ::cuSignalExternalSemaphoresAsync"
  (extsemarray (:pointer CUexternalSemaphore))
  (paramsarray (:pointer CUDA-EXTERNAL-SEMAPHORE-WAIT-PARAMS))
  (numextsems :unsigned-int)
  (stream CUstream))

(cffi:defcfun "cudestroyexternalsemaphore" CUresult
  "\brief Destroys an external semaphore
 
  Destroys an external semaphore object and releases any references
  to the underlying resource. Any outstanding signals or waits must
  have completed before the semaphore is destroyed.
 
  \param extSem - External semaphore to be destroyed
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_HANDLE
  \notefnerr
 
  \sa ::cuImportExternalSemaphore,
  ::cuSignalExternalSemaphoresAsync,
  ::cuWaitExternalSemaphoresAsync"
  (extsem CUexternalSemaphore))

(cffi:defcfun "custreamwaitvalue32" CUresult
  "\brief Wait on a memory location
 
  Enqueues a synchronization of the stream on the given memory location. Work
  ordered after the operation will block until the given condition on the
  memory is satisfied. By default, the condition is to wait for
  (int32_t)(addr - value) >= 0, a cyclic greater-or-equal.
  Other condition types can be specified via \p flags.
 
  If the memory was registered via ::cuMemHostRegister(), the device pointer
  should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
  be used with managed memory (::cuMemAllocManaged).
 
  Support for this can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.
 
  Support for CU_STREAM_WAIT_VALUE_NOR can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR.
 
  \param stream The stream to synchronize on the memory location.
  \param addr The memory location to wait on.
  \param value The value to compare with the memory location.
  \param flags See ::CUstreamWaitValue_flags.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \notefnerr
 
  \sa ::cuStreamWaitValue64,
  ::cuStreamWriteValue32,
  ::cuStreamWriteValue64
  ::cuStreamBatchMemOp,
  ::cuMemHostRegister,
  ::cuStreamWaitEvent"
  (stream CUstream)
  (addr CUdeviceptr)
  (value cuuint32-t)
  (flags :unsigned-int))

(cffi:defcfun "custreamwaitvalue64" CUresult
  "\brief Wait on a memory location
 
  Enqueues a synchronization of the stream on the given memory location. Work
  ordered after the operation will block until the given condition on the
  memory is satisfied. By default, the condition is to wait for
  (int64_t)(addr - value) >= 0, a cyclic greater-or-equal.
  Other condition types can be specified via \p flags.
 
  If the memory was registered via ::cuMemHostRegister(), the device pointer
  should be obtained with ::cuMemHostGetDevicePointer().
 
  Support for this can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 
  \param stream The stream to synchronize on the memory location.
  \param addr The memory location to wait on.
  \param value The value to compare with the memory location.
  \param flags See ::CUstreamWaitValue_flags.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \notefnerr
 
  \sa ::cuStreamWaitValue32,
  ::cuStreamWriteValue32,
  ::cuStreamWriteValue64,
  ::cuStreamBatchMemOp,
  ::cuMemHostRegister,
  ::cuStreamWaitEvent"
  (stream CUstream)
  (addr CUdeviceptr)
  (value cuuint64-t)
  (flags :unsigned-int))

(cffi:defcfun "custreamwritevalue32" CUresult
  "\brief Write a value to memory
 
  Write a value to memory. Unless the ::CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
  flag is passed, the write is preceded by a system-wide memory fence,
  equivalent to a __threadfence_system() but scoped to the stream
  rather than a CUDA thread.
 
  If the memory was registered via ::cuMemHostRegister(), the device pointer
  should be obtained with ::cuMemHostGetDevicePointer(). This function cannot
  be used with managed memory (::cuMemAllocManaged).
 
  Support for this can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS.
 
  \param stream The stream to do the write in.
  \param addr The device address to write to.
  \param value The value to write.
  \param flags See ::CUstreamWriteValue_flags.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \notefnerr
 
  \sa ::cuStreamWriteValue64,
  ::cuStreamWaitValue32,
  ::cuStreamWaitValue64,
  ::cuStreamBatchMemOp,
  ::cuMemHostRegister,
  ::cuEventRecord"
  (stream CUstream)
  (addr CUdeviceptr)
  (value cuuint32-t)
  (flags :unsigned-int))

(cffi:defcfun "custreamwritevalue64" CUresult
  "\brief Write a value to memory
 
  Write a value to memory. Unless the ::CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
  flag is passed, the write is preceded by a system-wide memory fence,
  equivalent to a __threadfence_system() but scoped to the stream
  rather than a CUDA thread.
 
  If the memory was registered via ::cuMemHostRegister(), the device pointer
  should be obtained with ::cuMemHostGetDevicePointer().
 
  Support for this can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS.
 
  \param stream The stream to do the write in.
  \param addr The device address to write to.
  \param value The value to write.
  \param flags See ::CUstreamWriteValue_flags.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \notefnerr
 
  \sa ::cuStreamWriteValue32,
  ::cuStreamWaitValue32,
  ::cuStreamWaitValue64,
  ::cuStreamBatchMemOp,
  ::cuMemHostRegister,
  ::cuEventRecord"
  (stream CUstream)
  (addr CUdeviceptr)
  (value cuuint64-t)
  (flags :unsigned-int))

(cffi:defcfun "custreambatchmemop" CUresult
  "\brief Batch operations to synchronize the stream via memory operations
 
  This is a batch version of ::cuStreamWaitValue32() and ::cuStreamWriteValue32().
  Batching operations may avoid some performance overhead in both the API call
  and the device execution versus adding them to the stream in separate API
  calls. The operations are enqueued in the order they appear in the array.
 
  See ::CUstreamBatchMemOpType for the full set of supported operations, and
  ::cuStreamWaitValue32(), ::cuStreamWaitValue64(), ::cuStreamWriteValue32(),
  and ::cuStreamWriteValue64() for details of specific operations.
 
  Basic support for this can be queried with ::cuDeviceGetAttribute() and
  ::CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS. See related APIs for details
  on querying support for specific operations.
 
  \param stream The stream to enqueue the operations in.
  \param count The number of operations in the array. Must be less than 256.
  \param paramArray The types and parameters of the individual operations.
  \param flags Reserved for future expansion; must be 0.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \notefnerr
 
  \sa ::cuStreamWaitValue32,
  ::cuStreamWaitValue64,
  ::cuStreamWriteValue32,
  ::cuStreamWriteValue64,
  ::cuMemHostRegister"
  (stream CUstream)
  (count :unsigned-int)
  (paramarray (:pointer))
  (flags :unsigned-int))

(cffi:defcfun "cufuncgetattribute" CUresult
  "\brief Returns information about a function
 
  Returns in \p pi the integer value of the attribute \p attrib on the kernel
  given by \p hfunc. The supported attributes are:
  - ::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: The maximum number of threads
    per block, beyond which a launch of the function would fail. This number
    depends on both the function and the device on which the function is
    currently loaded.
  - ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: The size in bytes of
    statically-allocated shared memory per block required by this function.
    This does not include dynamically-allocated shared memory requested by
    the user at runtime.
  - ::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: The size in bytes of user-allocated
    constant memory required by this function.
  - ::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: The size in bytes of local memory
    used by each thread of this function.
  - ::CU_FUNC_ATTRIBUTE_NUM_REGS: The number of registers used by each thread
    of this function.
  - ::CU_FUNC_ATTRIBUTE_PTX_VERSION: The PTX virtual architecture version for
    which the function was compiled. This value is the major PTX version  10
    + the minor PTX version, so a PTX version 1.3 function would return the
    value 13. Note that this may return the undefined value of 0 for cubins
    compiled prior to CUDA 3.0.
  - ::CU_FUNC_ATTRIBUTE_BINARY_VERSION: The binary architecture version for
    which the function was compiled. This value is the major binary
    version  10 + the minor binary version, so a binary version 1.3 function
    would return the value 13. Note that this will return a value of 10 for
    legacy cubins that do not have a properly-encoded binary architecture
    version.
  - ::CU_FUNC_CACHE_MODE_CA: The attribute to indicate whether the function has
    been compiled with user specified option -Xptxas --dlcm=ca set .
  - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: The maximum size in bytes of
    dynamically-allocated shared memory.
  - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: Preferred shared memory-L1
    cache split ratio in percent of total shared memory.
 
  \param pi     - Returned attribute value
  \param attrib - Attribute requested
  \param hfunc  - Function to query attribute of
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncSetCacheConfig,
  ::cuLaunchKernel,
  ::cudaFuncGetAttributes
  ::cudaFuncSetAttribute"
  (pi (:pointer :int))
  (attrib CUfunction-attribute)
  (hfunc CUfunction))

(cffi:defcfun "cufuncsetattribute" CUresult
  "\brief Sets information about a function
 
  This call sets the value of a specified attribute \p attrib on the kernel given
  by \p hfunc to an integer value specified by \p val
  This function returns CUDA_SUCCESS if the new value of the attribute could be
  successfully set. If the set fails, this call will return an error.
  Not all attributes can have values set. Attempting to set a value on a read-only
  attribute will result in an error (CUDA_ERROR_INVALID_VALUE)
 
  Supported attributes for the cuFuncSetAttribute call are:
  - ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: This maximum size in bytes of
    dynamically-allocated shared memory. The value should contain the requested
    maximum size of dynamically-allocated shared memory. The sum of this value and
    the function attribute ::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES cannot exceed the
    device attribute ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
    The maximal size of requestable dynamic shared memory may differ by GPU
    architecture.
  - ::CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: On devices where the L1
    cache and shared memory use the same hardware resources, this sets the shared memory
    carveout preference, in percent of the total shared memory.
    See ::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    This is only a hint, and the driver can choose a different ratio if required to execute the function.
 
  \param hfunc  - Function to query attribute of
  \param attrib - Attribute requested
  \param value   - The value to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncSetCacheConfig,
  ::cuLaunchKernel,
  ::cudaFuncGetAttributes
  ::cudaFuncSetAttribute"
  (hfunc CUfunction)
  (attrib CUfunction-attribute)
  (value :int))

(cffi:defcfun "cufuncsetcacheconfig" CUresult
  "\brief Sets the preferred cache configuration for a device function
 
  On devices where the L1 cache and shared memory use the same hardware
  resources, this sets through \p config the preferred cache configuration for
  the device function \p hfunc. This is only a preference. The driver will use
  the requested configuration if possible, but it is free to choose a different
  configuration if required to execute \p hfunc.  Any context-wide preference
  set via ::cuCtxSetCacheConfig() will be overridden by this per-function
  setting unless the per-function setting is ::CU_FUNC_CACHE_PREFER_NONE. In
  that case, the current context-wide setting will be used.
 
  This setting does nothing on devices where the size of the L1 cache and
  shared memory are fixed.
 
  Launching a kernel with a different preference than the most recent
  preference setting may insert a device-side synchronization point.
 
 
  The supported cache configurations are:
  - ::CU_FUNC_CACHE_PREFER_NONE: no preference for shared memory or L1 (default)
  - ::CU_FUNC_CACHE_PREFER_SHARED: prefer larger shared memory and smaller L1 cache
  - ::CU_FUNC_CACHE_PREFER_L1: prefer larger L1 cache and smaller shared memory
  - ::CU_FUNC_CACHE_PREFER_EQUAL: prefer equal sized L1 cache and shared memory
 
  \param hfunc  - Kernel to configure cache for
  \param config - Requested cache configuration
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cuLaunchKernel,
  ::cudaFuncSetCacheConfig"
  (hfunc CUfunction)
  (config CUfunc-cache))

(cffi:defcfun "cufuncsetsharedmemconfig" CUresult
  "\brief Sets the shared memory configuration for a device function.
 
  On devices with configurable shared memory banks, this function will
  force all subsequent launches of the specified device function to have
  the given shared memory bank size configuration. On any given launch of the
  function, the shared memory configuration of the device will be temporarily
  changed if needed to suit the function's preferred configuration. Changes in
  shared memory configuration between subsequent launches of functions,
  may introduce a device side synchronization point.
 
  Any per-function setting of shared memory bank size set via
  ::cuFuncSetSharedMemConfig will override the context wide setting set with
  ::cuCtxSetSharedMemConfig.
 
  Changing the shared memory bank size will not increase shared memory usage
  or affect occupancy of kernels, but may have major effects on performance.
  Larger bank sizes will allow for greater potential bandwidth to shared memory,
  but will change what kinds of accesses to shared memory will result in bank
  conflicts.
 
  This function will do nothing on devices with fixed shared memory bank size.
 
  The supported bank configurations are:
  - ::CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE: use the context's shared memory
    configuration when launching this function.
  - ::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE: set shared memory bank width to
    be natively four bytes when launching this function.
  - ::CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: set shared memory bank width to
    be natively eight bytes when launching this function.
 
  \param hfunc  - kernel to be given a shared memory config
  \param config - requested shared memory configuration
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuCtxGetSharedMemConfig,
  ::cuCtxSetSharedMemConfig,
  ::cuFuncGetAttribute,
  ::cuLaunchKernel,
  ::cudaFuncSetSharedMemConfig"
  (hfunc CUfunction)
  (config CUsharedconfig))

(cffi:defcfun "culaunchkernel" CUresult
  "\brief Launches a CUDA function
 
  Invokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
  grid of blocks. Each block contains \p blockDimX x \p blockDimY x
  \p blockDimZ threads.
 
  \p sharedMemBytes sets the amount of dynamic shared memory that will be
  available to each thread block.
 
  Kernel parameters to \p f can be specified in one of two ways:
 
  1) Kernel parameters can be specified via \p kernelParams.  If \p f
  has N parameters, then \p kernelParams needs to be an array of N
  pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
  must point to a region of memory from which the actual kernel
  parameter will be copied.  The number of kernel parameters and their
  offsets and sizes do not need to be specified as that information is
  retrieved directly from the kernel's image.
 
  2) Kernel parameters can also be packaged by the application into
  a single buffer that is passed in via the \p extra parameter.
  This places the burden on the application of knowing each kernel
  parameter's size and alignmentpadding within the buffer.  Here is
  an example of using the \p extra parameter in this manner:
  \code
    size_t argBufferSize;
    char argBuffer[256];

     populate argBuffer and argBufferSize

    void config[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
        CU_LAUNCH_PARAM_BUFFER_SIZE,    &argBufferSize,
        CU_LAUNCH_PARAM_END
    };
    status = cuLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, s, NULL, config);
  \endcode
 
  The \p extra parameter exists to allow ::cuLaunchKernel to take
  additional less commonly used arguments.  \p extra specifies a list of
  names of extra settings and their corresponding values.  Each extra
  setting name is immediately followed by the corresponding value.  The
  list must be terminated with either NULL or ::CU_LAUNCH_PARAM_END.
 
  - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
    array;
  - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
    value in \p extra will be a pointer to a buffer containing all
    the kernel parameters for launching kernel \p f;
  - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
    value in \p extra will be a pointer to a size_t containing the
    size of the buffer specified with ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 
  The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel
  parameters are specified with both \p kernelParams and \p extra
  (i.e. both \p kernelParams and \p extra are non-NULL).
 
  Calling ::cuLaunchKernel() sets persistent function state that is
  the same as function state set through the following deprecated APIs:
   ::cuFuncSetBlockShape(),
   ::cuFuncSetSharedSize(),
   ::cuParamSetSize(),
   ::cuParamSeti(),
   ::cuParamSetf(),
   ::cuParamSetv().
 
  When the kernel \p f is launched via ::cuLaunchKernel(), the previous
  block shape, shared size and parameter info associated with \p f
  is overwritten.
 
  Note that to use ::cuLaunchKernel(), the kernel \p f must either have
  been compiled with toolchain version 3.2 or later so that it will
  contain kernel parameter information, or have no kernel parameters.
  If either of these conditions is not met, then ::cuLaunchKernel() will
  return ::CUDA_ERROR_INVALID_IMAGE.
 
  \param f              - Kernel to launch
  \param gridDimX       - Width of grid in blocks
  \param gridDimY       - Height of grid in blocks
  \param gridDimZ       - Depth of grid in blocks
  \param blockDimX      - X dimension of each thread block
  \param blockDimY      - Y dimension of each thread block
  \param blockDimZ      - Z dimension of each thread block
  \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
  \param hStream        - Stream identifier
  \param kernelParams   - Array of pointers to kernel parameters
  \param extra          - Extra options
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_IMAGE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  \note_null_stream
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cudaLaunchKernel"
  (f CUfunction)
  (griddimx :unsigned-int)
  (griddimy :unsigned-int)
  (griddimz :unsigned-int)
  (blockdimx :unsigned-int)
  (blockdimy :unsigned-int)
  (blockdimz :unsigned-int)
  (sharedmembytes :unsigned-int)
  (hstream CUstream)
  (kernelparams (:pointer (:pointer :void)))
  (extra (:pointer (:pointer :void))))

(cffi:defcfun "culaunchcooperativekernel" CUresult
  "\brief Launches a CUDA function where thread blocks can cooperate and synchronize as they execute
 
  Invokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
  grid of blocks. Each block contains \p blockDimX x \p blockDimY x
  \p blockDimZ threads.
 
  \p sharedMemBytes sets the amount of dynamic shared memory that will be
  available to each thread block.
 
  The device on which this kernel is invoked must have a non-zero value for
  the device attribute ::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
 
  The total number of blocks launched cannot exceed the maximum number of blocks per
  multiprocessor as returned by ::cuOccupancyMaxActiveBlocksPerMultiprocessor (or
  ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
  as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
 
  The kernel cannot make use of CUDA dynamic parallelism.
 
  Kernel parameters must be specified via \p kernelParams.  If \p f
  has N parameters, then \p kernelParams needs to be an array of N
  pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
  must point to a region of memory from which the actual kernel
  parameter will be copied.  The number of kernel parameters and their
  offsets and sizes do not need to be specified as that information is
  retrieved directly from the kernel's image.
 
  Calling ::cuLaunchCooperativeKernel() sets persistent function state that is
  the same as function state set through ::cuLaunchKernel API
 
  When the kernel \p f is launched via ::cuLaunchCooperativeKernel(), the previous
  block shape, shared size and parameter info associated with \p f
  is overwritten.
 
  Note that to use ::cuLaunchCooperativeKernel(), the kernel \p f must either have
  been compiled with toolchain version 3.2 or later so that it will
  contain kernel parameter information, or have no kernel parameters.
  If either of these conditions is not met, then ::cuLaunchCooperativeKernel() will
  return ::CUDA_ERROR_INVALID_IMAGE.
 
  \param f              - Kernel to launch
  \param gridDimX       - Width of grid in blocks
  \param gridDimY       - Height of grid in blocks
  \param gridDimZ       - Depth of grid in blocks
  \param blockDimX      - X dimension of each thread block
  \param blockDimY      - Y dimension of each thread block
  \param blockDimZ      - Z dimension of each thread block
  \param sharedMemBytes - Dynamic shared-memory size per thread block in bytes
  \param hStream        - Stream identifier
  \param kernelParams   - Array of pointers to kernel parameters
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_IMAGE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  \note_null_stream
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cuLaunchCooperativeKernelMultiDevice,
  ::cudaLaunchCooperativeKernel"
  (f CUfunction)
  (griddimx :unsigned-int)
  (griddimy :unsigned-int)
  (griddimz :unsigned-int)
  (blockdimx :unsigned-int)
  (blockdimy :unsigned-int)
  (blockdimz :unsigned-int)
  (sharedmembytes :unsigned-int)
  (hstream CUstream)
  (kernelparams (:pointer (:pointer :void))))

(cffi:defcfun "culaunchcooperativekernelmultidevice" CUresult
  "\brief Launches CUDA functions on multiple devices where thread blocks can cooperate and synchronize as they execute
 
  Invokes kernels as specified in the \p launchParamsList array where each element
  of the array specifies all the parameters required to perform a single kernel launch.
  These kernels can cooperate and synchronize as they execute. The size of the array is
  specified by \p numDevices.
 
  No two kernels can be launched on the same device. All the devices targeted by this
  multi-device launch must be identical. All devices must have a non-zero value for the
  device attribute ::CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH.
 
  All kernels launched must be identical with respect to the compiled code. Note that
  any __device__, __constant__ or __managed__ variables present in the module that owns
  the kernel launched on each device, are independently instantiated on every device.
  It is the application's responsiblity to ensure these variables are initialized and
  used appropriately.
 
  The size of the grids as specified in blocks, the size of the blocks themselves
  and the amount of shared memory used by each thread block must also match across
  all launched kernels.
 
  The streams used to launch these kernels must have been created via either ::cuStreamCreate
  or ::cuStreamCreateWithPriority. The NULL stream or ::CU_STREAM_LEGACY or ::CU_STREAM_PER_THREAD
  cannot be used.
 
  The total number of blocks launched per kernel cannot exceed the maximum number of blocks
  per multiprocessor as returned by ::cuOccupancyMaxActiveBlocksPerMultiprocessor (or
  ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
  as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT. Since the
  total number of blocks launched per device has to match across all devices, the maximum
  number of blocks that can be launched per device will be limited by the device with the
  least number of multiprocessors.
 
  The kernels cannot make use of CUDA dynamic parallelism.
 
  The ::CUDA_LAUNCH_PARAMS structure is defined as:
  \code
        typedef struct CUDA_LAUNCH_PARAMS_st
        {
            CUfunction function;
            unsigned int gridDimX;
            unsigned int gridDimY;
            unsigned int gridDimZ;
            unsigned int blockDimX;
            unsigned int blockDimY;
            unsigned int blockDimZ;
            unsigned int sharedMemBytes;
            CUstream hStream;
            void kernelParams;
        } CUDA_LAUNCH_PARAMS;
  \endcode
  where:
  - ::CUDA_LAUNCH_PARAMS::function specifies the kernel to be launched. All functions must
    be identical with respect to the compiled code.
  - ::CUDA_LAUNCH_PARAMS::gridDimX is the width of the grid in blocks. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::gridDimY is the height of the grid in blocks. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::gridDimZ is the depth of the grid in blocks. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::blockDimX is the X dimension of each thread block. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::blockDimX is the Y dimension of each thread block. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::blockDimZ is the Z dimension of each thread block. This must match across
    all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::sharedMemBytes is the dynamic shared-memory size per thread block in bytes.
    This must match across all kernels launched.
  - ::CUDA_LAUNCH_PARAMS::hStream is the handle to the stream to perform the launch in. This cannot
    be the NULL stream or ::CU_STREAM_LEGACY or ::CU_STREAM_PER_THREAD. The CUDA context associated
    with this stream must match that associated with ::CUDA_LAUNCH_PARAMS::function.
  - ::CUDA_LAUNCH_PARAMS::kernelParams is an array of pointers to kernel parameters. If
    ::CUDA_LAUNCH_PARAMS::function has N parameters, then ::CUDA_LAUNCH_PARAMS::kernelParams
    needs to be an array of N pointers. Each of ::CUDA_LAUNCH_PARAMS::kernelParams[0] through
    ::CUDA_LAUNCH_PARAMS::kernelParams[N-1] must point to a region of memory from which the actual
    kernel parameter will be copied. The number of kernel parameters and their offsets and sizes
    do not need to be specified as that information is retrieved directly from the kernel's image.
 
  By default, the kernel won't begin execution on any GPU until all prior work in all the specified
  streams has completed. This behavior can be overridden by specifying the flag
  ::CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC. When this flag is specified, each kernel
  will only wait for prior work in the stream corresponding to that GPU to complete before it begins
  execution.
 
  Similarly, by default, any subsequent work pushed in any of the specified streams will not begin
  execution until the kernels on all GPUs have completed. This behavior can be overridden by specifying
  the flag ::CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC. When this flag is specified,
  any subsequent work pushed in any of the specified streams will only wait for the kernel launched
  on the GPU corresponding to that stream to complete before it begins execution.
 
  Calling ::cuLaunchCooperativeKernelMultiDevice() sets persistent function state that is
  the same as function state set through ::cuLaunchKernel API when called individually for each
  element in \p launchParamsList.
 
  When kernels are launched via ::cuLaunchCooperativeKernelMultiDevice(), the previous
  block shape, shared size and parameter info associated with each ::CUDA_LAUNCH_PARAMS::function
  in \p launchParamsList is overwritten.
 
  Note that to use ::cuLaunchCooperativeKernelMultiDevice(), the kernels must either have
  been compiled with toolchain version 3.2 or later so that it will
  contain kernel parameter information, or have no kernel parameters.
  If either of these conditions is not met, then ::cuLaunchCooperativeKernelMultiDevice() will
  return ::CUDA_ERROR_INVALID_IMAGE.
 
  \param launchParamsList - List of launch parameters, one per device
  \param numDevices       - Size of the \p launchParamsList array
  \param flags            - Flags to control launch behavior
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_IMAGE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  \note_null_stream
  \notefnerr
 
  \sa ::cuCtxGetCacheConfig,
  ::cuCtxSetCacheConfig,
  ::cuFuncSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cuLaunchCooperativeKernel,
  ::cudaLaunchCooperativeKernelMultiDevice"
  (launchparamslist (:pointer CUDA-LAUNCH-PARAMS))
  (numdevices :unsigned-int)
  (flags :unsigned-int))

(cffi:defcfun "culaunchhostfunc" CUresult
  "\brief Enqueues a host function call in a stream
 
  Enqueues a host function to run in a stream.  The function will be called
  after currently enqueued work and will block work added after it.
 
  The host function must not make any CUDA API calls.  Attempting to use a
  CUDA API may result in ::CUDA_ERROR_NOT_PERMITTED, but this is not required.
  The host function must not perform any synchronization that may depend on
  outstanding CUDA work not mandated to run earlier.  Host functions without a
  mandated order (such as in independent streams) execute in undefined order
  and may be serialized.
 
  For the purposes of Unified Memory, execution makes a number of guarantees:
  <ul>
    <li>The stream is considered idle for the duration of the function's
    execution.  Thus, for example, the function may always use memory attached
    to the stream it was enqueued in.<li>
    <li>The start of execution of the function has the same effect as
    synchronizing an event recorded in the same stream immediately prior to
    the function.  It thus synchronizes streams which have been joined
    prior to the function.<li>
    <li>Adding device work to any stream does not have the effect of making
    the stream active until all preceding host functions and stream callbacks
    have executed.  Thus, for
    example, a function might use global attached memory even if work has
    been added to another stream, if the work has been ordered behind the
    function call with an event.<li>
    <li>Completion of the function does not cause a stream to become
    active except as described above.  The stream will remain idle
    if no device work follows the function, and will remain idle across
    consecutive host functions or stream callbacks without device work in
    between.  Thus, for example,
    stream synchronization can be done by signaling from a host function at the
    end of the stream.<li>
  <ul>
 
  Note that, in contrast to ::cuStreamAddCallback, the function will not be
  called in the event of an error in the CUDA context.
 
  \param hStream  - Stream to enqueue function call in
  \param fn       - The function to call once preceding stream operations are complete
  \param userData - User-specified data to be passed to the function
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_SUPPORTED
  \note_null_stream
  \notefnerr
 
  \sa ::cuStreamCreate,
  ::cuStreamQuery,
  ::cuStreamSynchronize,
  ::cuStreamWaitEvent,
  ::cuStreamDestroy,
  ::cuMemAllocManaged,
  ::cuStreamAttachMemAsync,
  ::cuStreamAddCallback"
  (hstream CUstream)
  (fn CUhostFn)
  (userdata (:pointer :void)))

(cffi:defcfun "cufuncsetblockshape" CUresult
  "\brief Sets the block-dimensions for the function
 
  \deprecated
 
  Specifies the \p x, \p y, and \p z dimensions of the thread blocks that are
  created when the kernel given by \p hfunc is launched.
 
  \param hfunc - Kernel to specify dimensions of
  \param x     - X dimension
  \param y     - Y dimension
  \param z     - Z dimension
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetSharedSize,
  ::cuFuncSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSeti,
  ::cuParamSetf,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (x :int)
  (y :int)
  (z :int))

(cffi:defcfun "cufuncsetsharedsize" CUresult
  "\brief Sets the dynamic shared-memory size for the function
 
  \deprecated
 
  Sets through \p bytes the amount of dynamic shared memory that will be
  available to each thread block when the kernel given by \p hfunc is launched.
 
  \param hfunc - Kernel to specify dynamic shared-memory size for
  \param bytes - Dynamic shared-memory size per thread in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetCacheConfig,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSeti,
  ::cuParamSetf,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (bytes :unsigned-int))

(cffi:defcfun "cuparamsetsize" CUresult
  "\brief Sets the parameter size for the function
 
  \deprecated
 
  Sets through \p numbytes the total size in bytes needed by the function
  parameters of the kernel corresponding to \p hfunc.
 
  \param hfunc    - Kernel to set parameter size for
  \param numbytes - Size of parameter list in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetf,
  ::cuParamSeti,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (numbytes :unsigned-int))

(cffi:defcfun "cuparamseti" CUresult
  "\brief Adds an integer parameter to the function's argument list
 
  \deprecated
 
  Sets an integer parameter that will be specified the next time the
  kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
 
  \param hfunc  - Kernel to add parameter to
  \param offset - Offset to add parameter to argument list
  \param value  - Value of parameter
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSetf,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (offset :int)
  (value :unsigned-int))

(cffi:defcfun "cuparamsetf" CUresult
  "\brief Adds a floating-point parameter to the function's argument list
 
  \deprecated
 
  Sets a floating-point parameter that will be specified the next time the
  kernel corresponding to \p hfunc will be invoked. \p offset is a byte offset.
 
  \param hfunc  - Kernel to add parameter to
  \param offset - Offset to add parameter to argument list
  \param value  - Value of parameter
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSeti,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (offset :int)
  (value :float))

(cffi:defcfun "cuparamsetv" CUresult
  "\brief Adds arbitrary data to the function's argument list
 
  \deprecated
 
  Copies an arbitrary amount of data (specified in \p numbytes) from \p ptr
  into the parameter space of the kernel corresponding to \p hfunc. \p offset
  is a byte offset.
 
  \param hfunc    - Kernel to add data to
  \param offset   - Offset to add data to argument list
  \param ptr      - Pointer to arbitrary data
  \param numbytes - Size of data to copy in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSetf,
  ::cuParamSeti,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (hfunc CUfunction)
  (offset :int)
  (ptr (:pointer :void))
  (numbytes :unsigned-int))

(cffi:defcfun "culaunch" CUresult
  "\brief Launches a CUDA function
 
  \deprecated
 
  Invokes the kernel \p f on a 1 x 1 x 1 grid of blocks. The block
  contains the number of threads specified by a previous call to
  ::cuFuncSetBlockShape().
 
  \param f - Kernel to launch
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSetf,
  ::cuParamSeti,
  ::cuParamSetv,
  ::cuLaunchGrid,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (f CUfunction))

(cffi:defcfun "culaunchgrid" CUresult
  "\brief Launches a CUDA function
 
  \deprecated
 
  Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
  blocks. Each block contains the number of threads specified by a previous
  call to ::cuFuncSetBlockShape().
 
  \param f           - Kernel to launch
  \param grid_width  - Width of grid in blocks
  \param grid_height - Height of grid in blocks
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSetf,
  ::cuParamSeti,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGridAsync,
  ::cuLaunchKernel"
  (f CUfunction)
  (grid-width :int)
  (grid-height :int))

(cffi:defcfun "culaunchgridasync" CUresult
  "\brief Launches a CUDA function
 
  \deprecated
 
  Invokes the kernel \p f on a \p grid_width x \p grid_height grid of
  blocks. Each block contains the number of threads specified by a previous
  call to ::cuFuncSetBlockShape().
 
  \param f           - Kernel to launch
  \param grid_width  - Width of grid in blocks
  \param grid_height - Height of grid in blocks
  \param hStream     - Stream identifier
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_LAUNCH_FAILED,
  ::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
  ::CUDA_ERROR_LAUNCH_TIMEOUT,
  ::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
  ::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED
 
  \note In certain cases where cubins are created with no ABI (i.e., using \p ptxas \p --abi-compile \p no),
        this function may serialize kernel launches. In order to force the CUDA driver to retain
        asynchronous behavior, set the ::CU_CTX_LMEM_RESIZE_TO_MAX flag during context creation (see ::cuCtxCreate).
 
  \note_null_stream
  \notefnerr
 
  \sa ::cuFuncSetBlockShape,
  ::cuFuncSetSharedSize,
  ::cuFuncGetAttribute,
  ::cuParamSetSize,
  ::cuParamSetf,
  ::cuParamSeti,
  ::cuParamSetv,
  ::cuLaunch,
  ::cuLaunchGrid,
  ::cuLaunchKernel"
  (f CUfunction)
  (grid-width :int)
  (grid-height :int)
  (hstream CUstream))

(cffi:defcfun "cuparamsettexref" CUresult
  "\brief Adds a texture-reference to the function's argument list
 
  \deprecated
 
  Makes the CUDA array or linear memory bound to the texture reference
  \p hTexRef available to a device program as a texture. In this version of
  CUDA, the texture-reference must be obtained via ::cuModuleGetTexRef() and
  the \p texunit parameter must be set to ::CU_PARAM_TR_DEFAULT.
 
  \param hfunc   - Kernel to add texture-reference to
  \param texunit - Texture unit (must be ::CU_PARAM_TR_DEFAULT)
  \param hTexRef - Texture-reference to add to argument list
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr"
  (hfunc CUfunction)
  (texunit :int)
  (htexref CUtexref))

(cffi:defcfun "cugraphcreate" CUresult
  "\brief Creates a graph
 
  Creates an empty graph, which is returned via \p phGraph.
 
  \param phGraph - Returns newly created graph
  \param flags   - Graph creation flags, must be 0
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode,
  ::cuGraphInstantiate,
  ::cuGraphDestroy,
  ::cuGraphGetNodes,
  ::cuGraphGetRootNodes,
  ::cuGraphGetEdges,
  ::cuGraphClone"
  (phgraph (:pointer CUgraph))
  (flags :unsigned-int))

(cffi:defcfun "cugraphaddkernelnode" CUresult
  "\brief Creates a kernel execution node and adds it to a graph
 
  Creates a new kernel execution node and adds it to \p hGraph with \p numDependencies
  dependencies specified via \p dependencies and arguments specified in \p nodeParams.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  The CUDA_KERNEL_NODE_PARAMS structure is defined as:
 
  \code
   typedef struct CUDA_KERNEL_NODE_PARAMS_st {
       CUfunction func;
       unsigned int gridDimX;
       unsigned int gridDimY;
       unsigned int gridDimZ;
       unsigned int blockDimX;
       unsigned int blockDimY;
       unsigned int blockDimZ;
       unsigned int sharedMemBytes;
       void kernelParams;
       void extra;
   } CUDA_KERNEL_NODE_PARAMS;
  \endcode
 
  When the graph is launched, the node will invoke kernel \p func on a (\p gridDimX x
  \p gridDimY x \p gridDimZ) grid of blocks. Each block contains
  (\p blockDimX x \p blockDimY x \p blockDimZ) threads.
 
  \p sharedMemBytes sets the amount of dynamic shared memory that will be
  available to each thread block.
 
  Kernel parameters to \p func can be specified in one of two ways:
 
  1) Kernel parameters can be specified via \p kernelParams. If the kernel has N
  parameters, then \p kernelParams needs to be an array of N pointers. Each pointer,
  from \p kernelParams[0] to \p kernelParams[N-1], points to the region of memory from which the actual
  parameter will be copied. The number of kernel parameters and their offsets and sizes do not need
  to be specified as that information is retrieved directly from the kernel's image.
 
  2) Kernel parameters can also be packaged by the application into a single buffer that is passed in
  via \p extra. This places the burden on the application of knowing each kernel
  parameter's size and alignmentpadding within the buffer. The \p extra parameter exists
  to allow this function to take additional less commonly used arguments. \p extra specifies
  a list of names of extra settings and their corresponding values. Each extra setting name is
  immediately followed by the corresponding value. The list must be terminated with either NULL or
  CU_LAUNCH_PARAM_END.
 
  - ::CU_LAUNCH_PARAM_END, which indicates the end of the \p extra
    array;
  - ::CU_LAUNCH_PARAM_BUFFER_POINTER, which specifies that the next
    value in \p extra will be a pointer to a buffer
    containing all the kernel parameters for launching kernel
    \p func;
  - ::CU_LAUNCH_PARAM_BUFFER_SIZE, which specifies that the next
    value in \p extra will be a pointer to a size_t
    containing the size of the buffer specified with
    ::CU_LAUNCH_PARAM_BUFFER_POINTER;
 
  The error ::CUDA_ERROR_INVALID_VALUE will be returned if kernel parameters are specified with both
  \p kernelParams and \p extra (i.e. both \p kernelParams and
  \p extra are non-NULL).
 
  The \p kernelParams or \p extra array, as well as the argument values it points to,
  are copied during this call.
 
  \note Kernels launched using graphs must not use texture and surface references. Reading or
        writing through any texture or surface reference is undefined behavior.
        This restriction does not apply to texture and surface objects.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
  \param nodeParams      - Parameters for the GPU execution node
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchKernel,
  ::cuGraphKernelNodeGetParams,
  ::cuGraphKernelNodeSetParams,
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t)
  (nodeparams (:pointer CUDA-KERNEL-NODE-PARAMS)))

(cffi:defcfun "cugraphkernelnodegetparams" CUresult
  "\brief Returns a kernel node's parameters
 
  Returns the parameters of kernel node \p hNode in \p nodeParams.
  The \p kernelParams or \p extra array returned in \p nodeParams,
  as well as the argument values it points to, are owned by the node.
  This memory remains valid until the node is destroyed or its
  parameters are modified, and should not be modified
  directly. Use ::cuGraphKernelNodeSetParams to update the
  parameters of this node.
 
  The params will contain either \p kernelParams or \p extra,
  according to which of these was most recently set on the node.
 
  \param hNode      - Node to get the parameters for
  \param nodeParams - Pointer to return the parameters
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchKernel,
  ::cuGraphAddKernelNode,
  ::cuGraphKernelNodeSetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer CUDA-KERNEL-NODE-PARAMS)))

(cffi:defcfun "cugraphkernelnodesetparams" CUresult
  "\brief Sets a kernel node's parameters
 
  Sets the parameters of kernel node \p hNode to \p nodeParams.
 
  \param hNode      - Node to set the parameters for
  \param nodeParams - Parameters to copy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchKernel,
  ::cuGraphAddKernelNode,
  ::cuGraphKernelNodeGetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer CUDA-KERNEL-NODE-PARAMS)))

(cffi:defcfun "cugraphaddmemcpynode" CUresult
  "\brief Creates a memcpy node and adds it to a graph
 
  Creates a new memcpy node and adds it to \p hGraph with \p numDependencies
  dependencies specified via \p dependencies.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  When the graph is launched, the node will perform the memcpy described by \p copyParams.
  See ::cuMemcpy3D() for a description of the structure and its restrictions.
 
  Memcpy nodes have some additional restrictions with regards to managed memory, if the
  system contains at least one device which has a zero value for the device attribute
  ::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS. If one or more of the operands refer
  to managed memory, then using the memory type ::CU_MEMORYTYPE_UNIFIED is disallowed
  for those operand(s). The managed memory will be treated as residing on either the
  host or the device, depending on which memory type is specified.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
  \param copyParams      - Parameters for the memory copy
  \param ctx             - Context on which to run the node
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemcpy3D,
  ::cuGraphMemcpyNodeGetParams,
  ::cuGraphMemcpyNodeSetParams,
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemsetNode"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t)
  (copyparams (:pointer ))
  (ctx CUcontext))

(cffi:defcfun "cugraphmemcpynodegetparams" CUresult
  "\brief Returns a memcpy node's parameters
 
  Returns the parameters of memcpy node \p hNode in \p nodeParams.
 
  \param hNode      - Node to get the parameters for
  \param nodeParams - Pointer to return the parameters
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemcpy3D,
  ::cuGraphAddMemcpyNode,
  ::cuGraphMemcpyNodeSetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer)))

(cffi:defcfun "cugraphmemcpynodesetparams" CUresult
  "\brief Sets a memcpy node's parameters
 
  Sets the parameters of memcpy node \p hNode to \p nodeParams.
 
  \param hNode      - Node to set the parameters for
  \param nodeParams - Parameters to copy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemcpy3D,
  ::cuGraphAddMemcpyNode,
  ::cuGraphMemcpyNodeGetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer)))

(cffi:defcfun "cugraphaddmemsetnode" CUresult
  "\brief Creates a memset node and adds it to a graph
 
  Creates a new memset node and adds it to \p hGraph with \p numDependencies
  dependencies specified via \p dependencies.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  The element size must be 1, 2, or 4 bytes.
  When the graph is launched, the node will perform the memset described by \p memsetParams.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
  \param memsetParams    - Parameters for the memory set
  \param ctx             - Context on which to run the node
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_CONTEXT
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemsetD2D32,
  ::cuGraphMemsetNodeGetParams,
  ::cuGraphMemsetNodeSetParams,
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t)
  (memsetparams (:pointer))
  (ctx CUcontext))

(cffi:defcfun "cugraphmemsetnodegetparams" CUresult
  "\brief Returns a memset node's parameters
 
  Returns the parameters of memset node \p hNode in \p nodeParams.
 
  \param hNode      - Node to get the parameters for
  \param nodeParams - Pointer to return the parameters
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemsetD2D32,
  ::cuGraphAddMemsetNode,
  ::cuGraphMemsetNodeSetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer)))

(cffi:defcfun "cugraphmemsetnodesetparams" CUresult
  "\brief Sets a memset node's parameters
 
  Sets the parameters of memset node \p hNode to \p nodeParams.
 
  \param hNode      - Node to set the parameters for
  \param nodeParams - Parameters to copy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuMemsetD2D32,
  ::cuGraphAddMemsetNode,
  ::cuGraphMemsetNodeGetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer)))

(cffi:defcfun "cugraphaddhostnode" CUresult
  "\brief Creates a host execution node and adds it to a graph
 
  Creates a new CPU execution node and adds it to \p hGraph with \p numDependencies
  dependencies specified via \p dependencies and arguments specified in \p nodeParams.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  When the graph is launched, the node will invoke the specified CPU function.
  Host nodes are not supported under MPS with pre-Volta GPUs.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
  \param nodeParams      - Parameters for the host node
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_NOT_SUPPORTED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchHostFunc,
  ::cuGraphHostNodeGetParams,
  ::cuGraphHostNodeSetParams,
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t)
  (nodeparams (:pointer CUDA-HOST-NODE-PARAMS)))

(cffi:defcfun "cugraphhostnodegetparams" CUresult
  "\brief Returns a host node's parameters
 
  Returns the parameters of host node \p hNode in \p nodeParams.
 
  \param hNode      - Node to get the parameters for
  \param nodeParams - Pointer to return the parameters
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchHostFunc,
  ::cuGraphAddHostNode,
  ::cuGraphHostNodeSetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer CUDA-HOST-NODE-PARAMS)))

(cffi:defcfun "cugraphhostnodesetparams" CUresult
  "\brief Sets a host node's parameters
 
  Sets the parameters of host node \p hNode to \p nodeParams.
 
  \param hNode      - Node to set the parameters for
  \param nodeParams - Parameters to copy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuLaunchHostFunc,
  ::cuGraphAddHostNode,
  ::cuGraphHostNodeGetParams"
  (hnode CUgraphNode)
  (nodeparams (:pointer CUDA-HOST-NODE-PARAMS)))

(cffi:defcfun "cugraphaddchildgraphnode" CUresult
  "\brief Creates a child graph node and adds it to a graph
 
  Creates a new node which executes an embedded graph, and adds it to \p hGraph with
  \p numDependencies dependencies specified via \p dependencies.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  The node executes an embedded child graph. The child graph is cloned in this call.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
  \param childGraph      - The graph to clone into this node
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphChildGraphNodeGetGraph,
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode,
  ::cuGraphClone"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t)
  (childgraph CUgraph))

(cffi:defcfun "cugraphchildgraphnodegetgraph" CUresult
  "\brief Gets a handle to the embedded graph of a child graph node
 
  Gets a handle to the embedded graph in a child graph node. This call
  does not clone the graph. Changes to the graph will be reflected in
  the node, and the node retains ownership of the graph.
 
  \param hNode   - Node to get the embedded graph for
  \param phGraph - Location to store a handle to the graph
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphAddChildGraphNode,
  ::cuGraphNodeFindInClone"
  (hnode CUgraphNode)
  (phgraph (:pointer CUgraph)))

(cffi:defcfun "cugraphaddemptynode" CUresult
  "\brief Creates an empty node and adds it to a graph
 
  Creates a new node which performs no operation, and adds it to \p hGraph with
  \p numDependencies dependencies specified via \p dependencies.
  It is possible for \p numDependencies to be 0, in which case the node will be placed
  at the root of the graph. \p dependencies may not have any duplicate entries.
  A handle to the new node will be returned in \p phGraphNode.
 
  An empty node performs no operation during execution, but can be used for
  transitive ordering. For example, a phased execution graph with 2 groups of n
  nodes with a barrier between them can be represented using an empty node and
  2n dependency edges, rather than no empty node and n^2 dependency edges.
 
  \param phGraphNode     - Returns newly created node
  \param hGraph          - Graph to which to add the node
  \param dependencies    - Dependencies of the node
  \param numDependencies - Number of dependencies
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate,
  ::cuGraphDestroyNode,
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode"
  (phgraphnode (:pointer CUgraphNode))
  (hgraph CUgraph)
  (dependencies (:pointer CUgraphNode))
  (numdependencies size-t))

(cffi:defcfun "cugraphclone" CUresult
  "\brief Clones a graph
 
  This function creates a copy of \p originalGraph and returns it in \p  phGraphClone.
  All parameters are copied into the cloned graph. The original graph may be modified
  after this call without affecting the clone.
 
  Child graph nodes in the original graph are recursively copied into the clone.
 
  \param phGraphClone  - Returns newly created cloned graph
  \param originalGraph - Graph to clone
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_OUT_OF_MEMORY
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate,
  ::cuGraphNodeFindInClone"
  (phgraphclone (:pointer CUgraph))
  (originalgraph CUgraph))

(cffi:defcfun "cugraphnodefindinclone" CUresult
  "\brief Finds a cloned version of a node
 
  This function returns the node in \p hClonedGraph corresponding to \p hOriginalNode
  in the original graph.
 
  \p hClonedGraph must have been cloned from \p hOriginalGraph via ::cuGraphClone.
  \p hOriginalNode must have been in \p hOriginalGraph at the time of the call to
  ::cuGraphClone, and the corresponding cloned node in \p hClonedGraph must not have
  been removed. The cloned node is then returned via \p phClonedNode.
 
  \param phNode  - Returns handle to the cloned node
  \param hOriginalNode - Handle to the original node
  \param hClonedGraph - Cloned graph to query
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphClone"
  (phnode (:pointer CUgraphNode))
  (horiginalnode CUgraphNode)
  (hclonedgraph CUgraph))

(cffi:defcfun "cugraphnodegettype" CUresult
  "\brief Returns a node's type
 
  Returns the node type of \p hNode in \p type.
 
  \param hNode - Node to query
  \param type  - Pointer to return the node type
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphGetNodes,
  ::cuGraphGetRootNodes,
  ::cuGraphChildGraphNodeGetGraph,
  ::cuGraphKernelNodeGetParams,
  ::cuGraphKernelNodeSetParams,
  ::cuGraphHostNodeGetParams,
  ::cuGraphHostNodeSetParams,
  ::cuGraphMemcpyNodeGetParams,
  ::cuGraphMemcpyNodeSetParams,
  ::cuGraphMemsetNodeGetParams,
  ::cuGraphMemsetNodeSetParams"
  (hnode CUgraphNode)
  (type (:pointer CUgraphNodeType)))

(cffi:defcfun "cugraphgetnodes" CUresult
  "\brief Returns a graph's nodes
 
  Returns a list of \p hGraph's nodes. \p nodes may be NULL, in which case this
  function will return the number of nodes in \p numNodes. Otherwise,
  \p numNodes entries will be filled in. If \p numNodes is higher than the actual
  number of nodes, the remaining entries in \p nodes will be set to NULL, and the
  number of nodes actually obtained will be returned in \p numNodes.
 
  \param hGraph   - Graph to query
  \param nodes    - Pointer to return the nodes
  \param numNodes - See description
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate,
  ::cuGraphGetRootNodes,
  ::cuGraphGetEdges,
  ::cuGraphNodeGetType,
  ::cuGraphNodeGetDependencies,
  ::cuGraphNodeGetDependentNodes"
  (hgraph CUgraph)
  (nodes (:pointer CUgraphNode))
  (numnodes (:pointer size-t)))

(cffi:defcfun "cugraphgetrootnodes" CUresult
  "\brief Returns a graph's root nodes
 
  Returns a list of \p hGraph's root nodes. \p rootNodes may be NULL, in which case this
  function will return the number of root nodes in \p numRootNodes. Otherwise,
  \p numRootNodes entries will be filled in. If \p numRootNodes is higher than the actual
  number of root nodes, the remaining entries in \p rootNodes will be set to NULL, and the
  number of nodes actually obtained will be returned in \p numRootNodes.
 
  \param hGraph       - Graph to query
  \param rootNodes    - Pointer to return the root nodes
  \param numRootNodes - See description
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate,
  ::cuGraphGetNodes,
  ::cuGraphGetEdges,
  ::cuGraphNodeGetType,
  ::cuGraphNodeGetDependencies,
  ::cuGraphNodeGetDependentNodes"
  (hgraph CUgraph)
  (rootnodes (:pointer CUgraphNode))
  (numrootnodes (:pointer size-t)))

(cffi:defcfun "cugraphgetedges" CUresult
  "\brief Returns a graph's dependency edges
 
  Returns a list of \p hGraph's dependency edges. Edges are returned via corresponding
  indices in \p from and \p to; that is, the node in \p to[i] has a dependency on the
  node in \p from[i]. \p from and \p to may both be NULL, in which
  case this function only returns the number of edges in \p numEdges. Otherwise,
  \p numEdges entries will be filled in. If \p numEdges is higher than the actual
  number of edges, the remaining entries in \p from and \p to will be set to NULL, and
  the number of edges actually returned will be written to \p numEdges.
 
  \param hGraph   - Graph to get the edges from
  \param from     - Location to return edge endpoints
  \param to       - Location to return edge endpoints
  \param numEdges - See description
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphGetNodes,
  ::cuGraphGetRootNodes,
  ::cuGraphAddDependencies,
  ::cuGraphRemoveDependencies,
  ::cuGraphNodeGetDependencies,
  ::cuGraphNodeGetDependentNodes"
  (hgraph CUgraph)
  (from (:pointer CUgraphNode))
  (to (:pointer CUgraphNode))
  (numedges (:pointer size-t)))

(cffi:defcfun "cugraphnodegetdependencies" CUresult
  "\brief Returns a node's dependencies
 
  Returns a list of \p node's dependencies. \p dependencies may be NULL, in which case this
  function will return the number of dependencies in \p numDependencies. Otherwise,
  \p numDependencies entries will be filled in. If \p numDependencies is higher than the actual
  number of dependencies, the remaining entries in \p dependencies will be set to NULL, and the
  number of nodes actually obtained will be returned in \p numDependencies.
 
  \param hNode           - Node to query
  \param dependencies    - Pointer to return the dependencies
  \param numDependencies - See description
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphNodeGetDependentNodes,
  ::cuGraphGetNodes,
  ::cuGraphGetRootNodes,
  ::cuGraphGetEdges,
  ::cuGraphAddDependencies,
  ::cuGraphRemoveDependencies"
  (hnode CUgraphNode)
  (dependencies (:pointer CUgraphNode))
  (numdependencies (:pointer size-t)))

(cffi:defcfun "cugraphnodegetdependentnodes" CUresult
  "\brief Returns a node's dependent nodes
 
  Returns a list of \p node's dependent nodes. \p dependentNodes may be NULL, in which
  case this function will return the number of dependent nodes in \p numDependentNodes.
  Otherwise, \p numDependentNodes entries will be filled in. If \p numDependentNodes is
  higher than the actual number of dependent nodes, the remaining entries in
  \p dependentNodes will be set to NULL, and the number of nodes actually obtained will
  be returned in \p numDependentNodes.
 
  \param hNode             - Node to query
  \param dependentNodes    - Pointer to return the dependent nodes
  \param numDependentNodes - See description
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphNodeGetDependencies,
  ::cuGraphGetNodes,
  ::cuGraphGetRootNodes,
  ::cuGraphGetEdges,
  ::cuGraphAddDependencies,
  ::cuGraphRemoveDependencies"
  (hnode CUgraphNode)
  (dependentnodes (:pointer CUgraphNode))
  (numdependentnodes (:pointer size-t)))

(cffi:defcfun "cugraphadddependencies" CUresult
  "\brief Adds dependency edges to a graph
 
  The number of dependencies to be added is defined by \p numDependencies
  Elements in \p from and \p to at corresponding indices define a dependency.
  Each node in \p from and \p to must belong to \p hGraph.
 
  If \p numDependencies is 0, elements in \p from and \p to will be ignored.
  Specifying an existing dependency will return an error.
 
  \param hGraph - Graph to which dependencies are added
  \param from - Array of nodes that provide the dependencies
  \param to - Array of dependent nodes
  \param numDependencies - Number of dependencies to be added
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphRemoveDependencies,
  ::cuGraphGetEdges,
  ::cuGraphNodeGetDependencies,
  ::cuGraphNodeGetDependentNodes"
  (hgraph CUgraph)
  (from (:pointer CUgraphNode))
  (to (:pointer CUgraphNode))
  (numdependencies size-t))

(cffi:defcfun "cugraphremovedependencies" CUresult
  "\brief Removes dependency edges from a graph
 
  The number of \p dependencies to be removed is defined by \p numDependencies.
  Elements in \p from and \p to at corresponding indices define a dependency.
  Each node in \p from and \p to must belong to \p hGraph.
 
  If \p numDependencies is 0, elements in \p from and \p to will be ignored.
  Specifying a non-existing dependency will return an error.
 
  \param hGraph - Graph from which to remove dependencies
  \param from - Array of nodes that provide the dependencies
  \param to - Array of dependent nodes
  \param numDependencies - Number of dependencies to be removed
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphAddDependencies,
  ::cuGraphGetEdges,
  ::cuGraphNodeGetDependencies,
  ::cuGraphNodeGetDependentNodes"
  (hgraph CUgraph)
  (from (:pointer CUgraphNode))
  (to (:pointer CUgraphNode))
  (numdependencies size-t))

(cffi:defcfun "cugraphdestroynode" CUresult
  "\brief Remove a node from the graph
 
  Removes \p hNode from its graph. This operation also severs any dependencies of other nodes
  on \p hNode and vice versa.
 
  \param hNode  - Node to remove
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphAddChildGraphNode,
  ::cuGraphAddEmptyNode,
  ::cuGraphAddKernelNode,
  ::cuGraphAddHostNode,
  ::cuGraphAddMemcpyNode,
  ::cuGraphAddMemsetNode"
  (hnode CUgraphNode))

(cffi:defcfun "cugraphinstantiate" CUresult
  "\brief Creates an executable graph from a graph
 
  Instantiates \p hGraph as an executable graph. The graph is validated for any
  structural constraints or intra-node constraints which were not previously
  validated. If instantiation is successful, a handle to the instantiated graph
  is returned in \p graphExec.
 
  If there are any errors, diagnostic information may be returned in \p errorNode and
  \p logBuffer. This is the primary way to inspect instantiation errors. The output
  will be null terminated unless the diagnostics overflow
  the buffer. In this case, they will be truncated, and the last byte can be
  inspected to determine if truncation occurred.
 
  \param phGraphExec - Returns instantiated graph
  \param hGraph      - Graph to instantiate
  \param phErrorNode - In case of an instantiation error, this may be modified to
                       indicate a node contributing to the error
  \param logBuffer   - A character buffer to store diagnostic messages
  \param bufferSize  - Size of the log buffer in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate,
  ::cuGraphLaunch,
  ::cuGraphExecDestroy"
  (phgraphexec (:pointer CUgraphExec))
  (hgraph CUgraph)
  (pherrornode (:pointer CUgraphNode))
  (logbuffer (:pointer :char))
  (buffersize size-t))

(cffi:defcfun "cugraphexeckernelnodesetparams" CUresult
  "\brief Sets the parameters for a kernel node in the given graphExec
 
  Sets the parameters of a kernel node in an executable graph \p hGraphExec.
  The node is identified by the corresponding node \p hNode in the
  non-executable graph, from which the executable graph was instantiated.
 
  \p hNode must not have been removed from the original graph. The \p func field
  of \p nodeParams cannot be modified and must match the original value.
  All other values can be modified.
 
  The modifications take effect at the next launch of \p hGraphExec. Already
  enqueued or running launches of \p hGraphExec are not affected by this call.
  \p hNode is also not modified by this call.
 
  \param hGraphExec  - The executable graph in which to set the specified node
  \param hNode       - kernel node from the graph from which graphExec was instantiated
  \param nodeParams  - Updated Parameters to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_INVALID_VALUE,
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphAddKernelNode,
  ::cuGraphKernelNodeSetParams,
  ::cuGraphInstantiate"
  (hgraphexec CUgraphExec)
  (hnode CUgraphNode)
  (nodeparams (:pointer CUDA-KERNEL-NODE-PARAMS)))

(cffi:defcfun "cugraphlaunch" CUresult
  "\brief Launches an executable graph in a stream
 
  Executes \p hGraphExec in \p hStream. Only one instance of \p hGraphExec may be executing
  at a time. Each launch is ordered behind both any previous work in \p hStream
  and any previous launches of \p hGraphExec. To execute a graph concurrently, it must be
  instantiated multiple times into multiple executable graphs.
 
  \param hGraphExec - Executable graph to launch
  \param hStream    - Stream in which to launch the graph
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphInstantiate,
  ::cuGraphExecDestroy"
  (hgraphexec CUgraphExec)
  (hstream CUstream))

(cffi:defcfun "cugraphexecdestroy" CUresult
  "\brief Destroys an executable graph
 
  Destroys the executable graph specified by \p hGraphExec, as well
  as all of its executable nodes. If the executable graph is
  in-flight, it will not be terminated, but rather freed
  asynchronously on completion.
 
  \param hGraphExec - Executable graph to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphInstantiate,
  ::cuGraphLaunch"
  (hgraphexec CUgraphExec))

(cffi:defcfun "cugraphdestroy" CUresult
  "\brief Destroys a graph
 
  Destroys the graph specified by \p hGraph, as well as all of its nodes.
 
  \param hGraph - Graph to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_VALUE
  \note_graph_thread_safety
  \notefnerr
 
  \sa
  ::cuGraphCreate"
  (hgraph CUgraph))

(cffi:defcfun "cuoccupancymaxactiveblockspermultiprocessor" CUresult
  "\brief Returns occupancy of a function
 
  Returns in \p numBlocks the number of the maximum active blocks per
  streaming multiprocessor.
 
  \param numBlocks       - Returned occupancy
  \param func            - Kernel for which occupancy is calculated
  \param blockSize       - Block size the kernel is intended to be launched with
  \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cudaOccupancyMaxActiveBlocksPerMultiprocessor"
  (numblocks (:pointer :int))
  (func CUfunction)
  (blocksize :int)
  (dynamicsmemsize size-t))

(cffi:defcfun "cuoccupancymaxactiveblockspermultiprocessorwithflags" CUresult
  "\brief Returns occupancy of a function
 
  Returns in \p numBlocks the number of the maximum active blocks per
  streaming multiprocessor.
 
  The \p Flags parameter controls how special cases are handled. The
  valid flags are:
 
  - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
    ::cuOccupancyMaxActiveBlocksPerMultiprocessor;
 
  - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
    default behavior on platform where global caching affects
    occupancy. On such platforms, if caching is enabled, but
    per-block SM resource usage would result in zero occupancy, the
    occupancy calculator will calculate the occupancy as if caching
    is disabled. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE makes
    the occupancy calculator to return 0 in such cases. More information
    can be found about this feature in the Unified L1Texture Cache
    section of the Maxwell tuning guide.
 
  \param numBlocks       - Returned occupancy
  \param func            - Kernel for which occupancy is calculated
  \param blockSize       - Block size the kernel is intended to be launched with
  \param dynamicSMemSize - Per-block dynamic shared memory usage intended, in bytes
  \param flags           - Requested behavior for the occupancy calculator
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"
  (numblocks (:pointer :int))
  (func CUfunction)
  (blocksize :int)
  (dynamicsmemsize size-t)
  (flags :unsigned-int))

(cffi:defcfun "cuoccupancymaxpotentialblocksize" CUresult
  "\brief Suggest a launch configuration with reasonable occupancy
 
  Returns in \p blockSize a reasonable block size that can achieve
  the maximum occupancy (or, the maximum number of active warps with
  the fewest blocks per multiprocessor), and in \p minGridSize the
  minimum grid size to achieve the maximum occupancy.
 
  If \p blockSizeLimit is 0, the configurator will use the maximum
  block size permitted by the device  function instead.
 
  If per-block dynamic shared memory allocation is not needed, the
  user should leave both \p blockSizeToDynamicSMemSize and \p
  dynamicSMemSize as 0.
 
  If per-block dynamic shared memory allocation is needed, then if
  the dynamic shared memory size is constant regardless of block
  size, the size should be passed through \p dynamicSMemSize, and \p
  blockSizeToDynamicSMemSize should be NULL.
 
  Otherwise, if the per-block dynamic shared memory size varies with
  different block sizes, the user needs to provide a unary function
  through \p blockSizeToDynamicSMemSize that computes the dynamic
  shared memory needed by \p func for any given block size. \p
  dynamicSMemSize is ignored. An example signature is:
 
  \code
      Take block size, returns dynamic shared memory needed
     size_t blockToSmem(int blockSize);
  \endcode
 
  \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
  \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
  \param func        - Kernel for which launch configuration is calculated
  \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
  \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
  \param blockSizeLimit  - The maximum block size \p func is designed to handle
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cudaOccupancyMaxPotentialBlockSize"
  (mingridsize (:pointer :int))
  (blocksize (:pointer :int))
  (func CUfunction)
  (blocksizetodynamicsmemsize :int)
  (dynamicsmemsize size-t)
  (blocksizelimit :int))

(cffi:defcfun "cuoccupancymaxpotentialblocksizewithflags" CUresult
  "\brief Suggest a launch configuration with reasonable occupancy
 
  An extended version of ::cuOccupancyMaxPotentialBlockSize. In
  addition to arguments passed to ::cuOccupancyMaxPotentialBlockSize,
  ::cuOccupancyMaxPotentialBlockSizeWithFlags also takes a \p Flags
  parameter.
 
  The \p Flags parameter controls how special cases are handled. The
  valid flags are:
 
  - ::CU_OCCUPANCY_DEFAULT, which maintains the default behavior as
    ::cuOccupancyMaxPotentialBlockSize;
 
  - ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE, which suppresses the
    default behavior on platform where global caching affects
    occupancy. On such platforms, the launch configurations that
    produces maximal occupancy might not support global
    caching. Setting ::CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
    guarantees that the the produced launch configuration is global
    caching compatible at a potential cost of occupancy. More information
    can be found about this feature in the Unified L1Texture Cache
    section of the Maxwell tuning guide.
 
  \param minGridSize - Returned minimum grid size needed to achieve the maximum occupancy
  \param blockSize   - Returned maximum block size that can achieve the maximum occupancy
  \param func        - Kernel for which launch configuration is calculated
  \param blockSizeToDynamicSMemSize - A function that calculates how much per-block dynamic shared memory \p func uses based on the block size
  \param dynamicSMemSize - Dynamic shared memory usage intended, in bytes
  \param blockSizeLimit  - The maximum block size \p func is designed to handle
  \param flags       - Options
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cudaOccupancyMaxPotentialBlockSizeWithFlags"
  (mingridsize (:pointer :int))
  (blocksize (:pointer :int))
  (func CUfunction)
  (blocksizetodynamicsmemsize :int)
  (dynamicsmemsize size-t)
  (blocksizelimit :int)
  (flags :unsigned-int))

(cffi:defcfun "cutexrefsetarray" CUresult
  "\brief Binds an array as a texture reference
 
  \deprecated
 
  Binds the CUDA array \p hArray to the texture reference \p hTexRef. Any
  previous address or CUDA array state associated with the texture reference
  is superseded by this function. \p Flags must be set to
  ::CU_TRSA_OVERRIDE_FORMAT. Any CUDA array previously bound to \p hTexRef is
  unbound.
 
  \param hTexRef - Texture reference to bind
  \param hArray  - Array to bind
  \param Flags   - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToArray"
  (htexref CUtexref)
  (harray CUarray)
  (flags :unsigned-int))

(cffi:defcfun "cutexrefsetmipmappedarray" CUresult
  "\brief Binds a mipmapped array to a texture reference
 
  \deprecated
 
  Binds the CUDA mipmapped array \p hMipmappedArray to the texture reference \p hTexRef.
  Any previous address or CUDA array state associated with the texture reference
  is superseded by this function. \p Flags must be set to ::CU_TRSA_OVERRIDE_FORMAT.
  Any CUDA array previously bound to \p hTexRef is unbound.
 
  \param hTexRef         - Texture reference to bind
  \param hMipmappedArray - Mipmapped array to bind
  \param Flags           - Options (must be ::CU_TRSA_OVERRIDE_FORMAT)
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (hmipmappedarray CUmipmappedArray)
  (flags :unsigned-int))

(cffi:defcfun ("cutexrefsetaddress_v2" cutexrefsetaddress-v2) CUresult
  (byteoffset (:pointer size-t))
  (htexref CUtexref)
  (dptr CUdeviceptr)
  (bytes size-t))

(cffi:defcfun ("cutexrefsetaddress2d_v3" cutexrefsetaddress2d-v3) CUresult
  (htexref CUtexref)
  (desc (:pointer))
  (dptr CUdeviceptr)
  (pitch size-t))

(cffi:defcfun "cutexrefsetformat" CUresult
  "\brief Sets the format for a texture reference
 
  \deprecated
 
  Specifies the format of the data to be read by the texture reference
  \p hTexRef. \p fmt and \p NumPackedComponents are exactly analogous to the
  ::Format and ::NumChannels members of the ::CUDA_ARRAY_DESCRIPTOR structure:
  They specify the format of each component and the number of components per
  array element.
 
  \param hTexRef             - Texture reference
  \param fmt                 - Format to set
  \param NumPackedComponents - Number of components per array element
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaCreateChannelDesc,
  ::cudaBindTexture,
  ::cudaBindTexture2D,
  ::cudaBindTextureToArray,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (fmt CUarray-format)
  (numpackedcomponents :int))

(cffi:defcfun "cutexrefsetaddressmode" CUresult
  "\brief Sets the addressing mode for a texture reference
 
  \deprecated
 
  Specifies the addressing mode \p am for the given dimension \p dim of the
  texture reference \p hTexRef. If \p dim is zero, the addressing mode is
  applied to the first parameter of the functions used to fetch from the
  texture; if \p dim is 1, the second, and so on. ::CUaddress_mode is defined
  as:
  \code
   typedef enum CUaddress_mode_enum {
      CU_TR_ADDRESS_MODE_WRAP = 0,
      CU_TR_ADDRESS_MODE_CLAMP = 1,
      CU_TR_ADDRESS_MODE_MIRROR = 2,
      CU_TR_ADDRESS_MODE_BORDER = 3
   } CUaddress_mode;
  \endcode
 
  Note that this call has no effect if \p hTexRef is bound to linear memory.
  Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES, is not set, the only
  supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
 
  \param hTexRef - Texture reference
  \param dim     - Dimension
  \param am      - Addressing mode to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTexture,
  ::cudaBindTexture2D,
  ::cudaBindTextureToArray,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (dim :int)
  (am CUaddress-mode))

(cffi:defcfun "cutexrefsetfiltermode" CUresult
  "\brief Sets the filtering mode for a texture reference
 
  \deprecated
 
  Specifies the filtering mode \p fm to be used when reading memory through
  the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
 
  \code
   typedef enum CUfilter_mode_enum {
      CU_TR_FILTER_MODE_POINT = 0,
      CU_TR_FILTER_MODE_LINEAR = 1
   } CUfilter_mode;
  \endcode
 
  Note that this call has no effect if \p hTexRef is bound to linear memory.
 
  \param hTexRef - Texture reference
  \param fm      - Filtering mode to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToArray"
  (htexref CUtexref)
  (fm CUfilter-mode))

(cffi:defcfun "cutexrefsetmipmapfiltermode" CUresult
  "\brief Sets the mipmap filtering mode for a texture reference
 
  \deprecated
 
  Specifies the mipmap filtering mode \p fm to be used when reading memory through
  the texture reference \p hTexRef. ::CUfilter_mode_enum is defined as:
 
  \code
   typedef enum CUfilter_mode_enum {
      CU_TR_FILTER_MODE_POINT = 0,
      CU_TR_FILTER_MODE_LINEAR = 1
   } CUfilter_mode;
  \endcode
 
  Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 
  \param hTexRef - Texture reference
  \param fm      - Filtering mode to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (fm CUfilter-mode))

(cffi:defcfun "cutexrefsetmipmaplevelbias" CUresult
  "\brief Sets the mipmap level bias for a texture reference
 
  \deprecated
 
  Specifies the mipmap level bias \p bias to be added to the specified mipmap level when
  reading memory through the texture reference \p hTexRef.
 
  Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 
  \param hTexRef - Texture reference
  \param bias    - Mipmap level bias
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (bias :float))

(cffi:defcfun "cutexrefsetmipmaplevelclamp" CUresult
  "\brief Sets the mipmap minmax mipmap level clamps for a texture reference
 
  \deprecated
 
  Specifies the minmax mipmap level clamps, \p minMipmapLevelClamp and \p maxMipmapLevelClamp
  respectively, to be used when reading memory through the texture reference
  \p hTexRef.
 
  Note that this call has no effect if \p hTexRef is not bound to a mipmapped array.
 
  \param hTexRef        - Texture reference
  \param minMipmapLevelClamp - Mipmap min level clamp
  \param maxMipmapLevelClamp - Mipmap max level clamp
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (minmipmaplevelclamp :float)
  (maxmipmaplevelclamp :float))

(cffi:defcfun "cutexrefsetmaxanisotropy" CUresult
  "\brief Sets the maximum anisotropy for a texture reference
 
  \deprecated
 
  Specifies the maximum anisotropy \p maxAniso to be used when reading memory through
  the texture reference \p hTexRef.
 
  Note that this call has no effect if \p hTexRef is bound to linear memory.
 
  \param hTexRef  - Texture reference
  \param maxAniso - Maximum anisotropy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTextureToArray,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (maxaniso :unsigned-int))

(cffi:defcfun "cutexrefsetbordercolor" CUresult
  "\brief Sets the border color for a texture reference
 
  \deprecated
 
  Specifies the value of the RGBA color via the \p pBorderColor to the texture reference
  \p hTexRef. The color value supports only float type and holds color components in
  the following sequence:
  pBorderColor[0] holds 'R' component
  pBorderColor[1] holds 'G' component
  pBorderColor[2] holds 'B' component
  pBorderColor[3] holds 'A' component
 
  Note that the color values can be set only when the Address mode is set to
  CU_TR_ADDRESS_MODE_BORDER using ::cuTexRefSetAddressMode.
  Applications using integer border color values have to reinterpret_cast their values to float.
 
  \param hTexRef       - Texture reference
  \param pBorderColor  - RGBA color
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddressMode,
  ::cuTexRefGetAddressMode, ::cuTexRefGetBorderColor,
  ::cudaBindTexture,
  ::cudaBindTexture2D,
  ::cudaBindTextureToArray,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (pbordercolor (:pointer :float)))

(cffi:defcfun "cutexrefsetflags" CUresult
  "\brief Sets the flags for a texture reference
 
  \deprecated
 
  Specifies optional flags via \p Flags to specify the behavior of data
  returned through the texture reference \p hTexRef. The valid flags are:
 
  - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of
    having the texture promote integer data to floating point data in the
    range [0, 1]. Note that texture with 32-bit integer format
    would not be promoted, regardless of whether or not this
    flag is specified;
  - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the
    default behavior of having the texture coordinates range
    from [0, Dim) where Dim is the width or height of the CUDA
    array. Instead, the texture coordinates [0, 1.0) reference
    the entire breadth of the array dimension;
 
  \param hTexRef - Texture reference
  \param Flags   - Optional flags to set
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat,
  ::cudaBindTexture,
  ::cudaBindTexture2D,
  ::cudaBindTextureToArray,
  ::cudaBindTextureToMipmappedArray"
  (htexref CUtexref)
  (flags :unsigned-int))

(cffi:defcfun ("cutexrefgetaddress_v2" cutexrefgetaddress-v2) CUresult
  (pdptr (:pointer CUdeviceptr))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetarray" CUresult
  "\brief Gets the array bound to a texture reference
 
  \deprecated
 
  Returns in \p phArray the CUDA array bound to the texture reference
  \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
  is not bound to any CUDA array.
 
  \param phArray - Returned array
  \param hTexRef - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pharray (:pointer CUarray))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetmipmappedarray" CUresult
  "\brief Gets the mipmapped array bound to a texture reference
 
  \deprecated
 
  Returns in \p phMipmappedArray the CUDA mipmapped array bound to the texture
  reference \p hTexRef, or returns ::CUDA_ERROR_INVALID_VALUE if the texture reference
  is not bound to any CUDA mipmapped array.
 
  \param phMipmappedArray - Returned mipmapped array
  \param hTexRef          - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (phmipmappedarray (:pointer CUmipmappedArray))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetaddressmode" CUresult
  "\brief Gets the addressing mode used by a texture reference
 
  \deprecated
 
  Returns in \p pam the addressing mode corresponding to the
  dimension \p dim of the texture reference \p hTexRef. Currently, the only
  valid value for \p dim are 0 and 1.
 
  \param pam     - Returned addressing mode
  \param hTexRef - Texture reference
  \param dim     - Dimension
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pam (:pointer CUaddress-mode))
  (htexref CUtexref)
  (dim :int))

(cffi:defcfun "cutexrefgetfiltermode" CUresult
  "\brief Gets the filter-mode used by a texture reference
 
  \deprecated
 
  Returns in \p pfm the filtering mode of the texture reference
  \p hTexRef.
 
  \param pfm     - Returned filtering mode
  \param hTexRef - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pfm (:pointer CUfilter-mode))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetformat" CUresult
  "\brief Gets the format used by a texture reference
 
  \deprecated
 
  Returns in \p pFormat and \p pNumChannels the format and number
  of components of the CUDA array bound to the texture reference \p hTexRef.
  If \p pFormat or \p pNumChannels is NULL, it will be ignored.
 
  \param pFormat      - Returned format
  \param pNumChannels - Returned number of components
  \param hTexRef      - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags"
  (pformat (:pointer CUarray-format))
  (pnumchannels (:pointer :int))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetmipmapfiltermode" CUresult
  "\brief Gets the mipmap filtering mode for a texture reference
 
  \deprecated
 
  Returns the mipmap filtering mode in \p pfm that's used when reading memory through
  the texture reference \p hTexRef.
 
  \param pfm     - Returned mipmap filtering mode
  \param hTexRef - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pfm (:pointer CUfilter-mode))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetmipmaplevelbias" CUresult
  "\brief Gets the mipmap level bias for a texture reference
 
  \deprecated
 
  Returns the mipmap level bias in \p pBias that's added to the specified mipmap
  level when reading memory through the texture reference \p hTexRef.
 
  \param pbias   - Returned mipmap level bias
  \param hTexRef - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pbias (:pointer :float))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetmipmaplevelclamp" CUresult
  "\brief Gets the minmax mipmap level clamps for a texture reference
 
  \deprecated
 
  Returns the minmax mipmap level clamps in \p pminMipmapLevelClamp and \p pmaxMipmapLevelClamp
  that's used when reading memory through the texture reference \p hTexRef.
 
  \param pminMipmapLevelClamp - Returned mipmap min level clamp
  \param pmaxMipmapLevelClamp - Returned mipmap max level clamp
  \param hTexRef              - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pminmipmaplevelclamp (:pointer :float))
  (pmaxmipmaplevelclamp (:pointer :float))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetmaxanisotropy" CUresult
  "\brief Gets the maximum anisotropy for a texture reference
 
  \deprecated
 
  Returns the maximum anisotropy in \p pmaxAniso that's used when reading memory through
  the texture reference \p hTexRef.
 
  \param pmaxAniso - Returned maximum anisotropy
  \param hTexRef   - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFlags, ::cuTexRefGetFormat"
  (pmaxaniso (:pointer :int))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetbordercolor" CUresult
  "\brief Gets the border color used by a texture reference
 
  \deprecated
 
  Returns in \p pBorderColor, values of the RGBA color used by
  the texture reference \p hTexRef.
  The color value is of type float and holds color components in
  the following sequence:
  pBorderColor[0] holds 'R' component
  pBorderColor[1] holds 'G' component
  pBorderColor[2] holds 'B' component
  pBorderColor[3] holds 'A' component
 
  \param hTexRef  - Texture reference
  \param pBorderColor   - Returned Type and Value of RGBA color
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddressMode,
  ::cuTexRefSetAddressMode, ::cuTexRefSetBorderColor"
  (pbordercolor (:pointer :float))
  (htexref CUtexref))

(cffi:defcfun "cutexrefgetflags" CUresult
  "\brief Gets the flags used by a texture reference
 
  \deprecated
 
  Returns in \p pFlags the flags of the texture reference \p hTexRef.
 
  \param pFlags  - Returned flags
  \param hTexRef - Texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefSetAddress,
  ::cuTexRefSetAddress2D, ::cuTexRefSetAddressMode, ::cuTexRefSetArray,
  ::cuTexRefSetFilterMode, ::cuTexRefSetFlags, ::cuTexRefSetFormat,
  ::cuTexRefGetAddress, ::cuTexRefGetAddressMode, ::cuTexRefGetArray,
  ::cuTexRefGetFilterMode, ::cuTexRefGetFormat"
  (pflags (:pointer :unsigned-int))
  (htexref CUtexref))

(cffi:defcfun "cutexrefcreate" CUresult
  "\brief Creates a texture reference
 
  \deprecated
 
  Creates a texture reference and returns its handle in \p pTexRef. Once
  created, the application must call ::cuTexRefSetArray() or
  ::cuTexRefSetAddress() to associate the reference with allocated memory.
  Other texture reference functions are used to specify the format and
  interpretation (addressing, filtering, etc.) to be used when the memory is
  read through this texture reference.
 
  \param pTexRef - Returned texture reference
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefDestroy"
  (ptexref (:pointer CUtexref)))

(cffi:defcfun "cutexrefdestroy" CUresult
  "\brief Destroys a texture reference
 
  \deprecated
 
  Destroys the texture reference specified by \p hTexRef.
 
  \param hTexRef - Texture reference to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuTexRefCreate"
  (htexref CUtexref))

(cffi:defcfun "cusurfrefsetarray" CUresult
  "\brief Sets the CUDA array for a surface reference.
 
  \deprecated
 
  Sets the CUDA array \p hArray to be read and written by the surface reference
  \p hSurfRef.  Any previous CUDA array state associated with the surface
  reference is superseded by this function.  \p Flags must be set to 0.
  The ::CUDA_ARRAY3D_SURFACE_LDST flag must have been set for the CUDA array.
  Any CUDA array previously bound to \p hSurfRef is unbound.

  \param hSurfRef - Surface reference handle
  \param hArray - CUDA array handle
  \param Flags - set to 0
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuModuleGetSurfRef,
  ::cuSurfRefGetArray,
  ::cudaBindSurfaceToArray"
  (hsurfref CUsurfref)
  (harray CUarray)
  (flags :unsigned-int))

(cffi:defcfun "cusurfrefgetarray" CUresult
  "\brief Passes back the CUDA array bound to a surface reference.
 
  \deprecated
 
  Returns in \p phArray the CUDA array bound to the surface reference
  \p hSurfRef, or returns ::CUDA_ERROR_INVALID_VALUE if the surface reference
  is not bound to any CUDA array.

  \param phArray - Surface reference handle
  \param hSurfRef - Surface reference handle
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa ::cuModuleGetSurfRef, ::cuSurfRefSetArray"
  (pharray (:pointer CUarray))
  (hsurfref CUsurfref))

(cffi:defcfun "cutexobjectcreate" CUresult
  "\brief Creates a texture object
 
  Creates a texture object and returns it in \p pTexObject. \p pResDesc describes
  the data to texture from. \p pTexDesc describes how the data should be sampled.
  \p pResViewDesc is an optional argument that specifies an alternate format for
  the data described by \p pResDesc, and also describes the subresource region
  to restrict access to when texturing. \p pResViewDesc can only be specified if
  the type of resource is a CUDA array or a CUDA mipmapped array.
 
  Texture objects are only supported on devices of compute capability 3.0 or higher.
  Additionally, a texture object is an opaque value, and, as such, should only be
  accessed through CUDA API calls.
 
  The ::CUDA_RESOURCE_DESC structure is defined as:
  \code
        typedef struct CUDA_RESOURCE_DESC_st
        {
            CUresourcetype resType;

            union {
                struct {
                    CUarray hArray;
                } array;
                struct {
                    CUmipmappedArray hMipmappedArray;
                } mipmap;
                struct {
                    CUdeviceptr devPtr;
                    CUarray_format format;
                    unsigned int numChannels;
                    size_t sizeInBytes;
                } linear;
                struct {
                    CUdeviceptr devPtr;
                    CUarray_format format;
                    unsigned int numChannels;
                    size_t width;
                    size_t height;
                    size_t pitchInBytes;
                } pitch2D;
            } res;

            unsigned int flags;
        } CUDA_RESOURCE_DESC;

  \endcode
  where:
  - ::CUDA_RESOURCE_DESC::resType specifies the type of resource to texture from.
  CUresourceType is defined as:
  \code
        typedef enum CUresourcetype_enum {
            CU_RESOURCE_TYPE_ARRAY           = 0x00,
            CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,
            CU_RESOURCE_TYPE_LINEAR          = 0x02,
            CU_RESOURCE_TYPE_PITCH2D         = 0x03
        } CUresourcetype;
  \endcode
 
  \par
  If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_ARRAY, ::CUDA_RESOURCE_DESC::res::array::hArray
  must be set to a valid CUDA array handle.
 
  \par
  If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY, ::CUDA_RESOURCE_DESC::res::mipmap::hMipmappedArray
  must be set to a valid CUDA mipmapped array handle.
 
  \par
  If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_LINEAR, ::CUDA_RESOURCE_DESC::res::linear::devPtr
  must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
  ::CUDA_RESOURCE_DESC::res::linear::format and ::CUDA_RESOURCE_DESC::res::linear::numChannels
  describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::linear::sizeInBytes
  specifies the size of the array in bytes. The total number of elements in the linear address range cannot exceed
  ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH. The number of elements is computed as (sizeInBytes  (sizeof(format)  numChannels)).
 
  \par
  If ::CUDA_RESOURCE_DESC::resType is set to ::CU_RESOURCE_TYPE_PITCH2D, ::CUDA_RESOURCE_DESC::res::pitch2D::devPtr
  must be set to a valid device pointer, that is aligned to ::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT.
  ::CUDA_RESOURCE_DESC::res::pitch2D::format and ::CUDA_RESOURCE_DESC::res::pitch2D::numChannels
  describe the format of each component and the number of components per array element. ::CUDA_RESOURCE_DESC::res::pitch2D::width
  and ::CUDA_RESOURCE_DESC::res::pitch2D::height specify the width and height of the array in elements, and cannot exceed
  ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH and ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT respectively.
  ::CUDA_RESOURCE_DESC::res::pitch2D::pitchInBytes specifies the pitch between two rows in bytes and has to be aligned to
  ::CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT. Pitch cannot exceed ::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH.
 
  - ::flags must be set to zero.
 
 
  The ::CUDA_TEXTURE_DESC struct is defined as
  \code
        typedef struct CUDA_TEXTURE_DESC_st {
            CUaddress_mode addressMode[3];
            CUfilter_mode filterMode;
            unsigned int flags;
            unsigned int maxAnisotropy;
            CUfilter_mode mipmapFilterMode;
            float mipmapLevelBias;
            float minMipmapLevelClamp;
            float maxMipmapLevelClamp;
        } CUDA_TEXTURE_DESC;
  \endcode
  where
  - ::CUDA_TEXTURE_DESC::addressMode specifies the addressing mode for each dimension of the texture data. ::CUaddress_mode is defined as:
    \code
        typedef enum CUaddress_mode_enum {
            CU_TR_ADDRESS_MODE_WRAP = 0,
            CU_TR_ADDRESS_MODE_CLAMP = 1,
            CU_TR_ADDRESS_MODE_MIRROR = 2,
            CU_TR_ADDRESS_MODE_BORDER = 3
        } CUaddress_mode;
    \endcode
    This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR. Also, if the flag, ::CU_TRSF_NORMALIZED_COORDINATES
    is not set, the only supported address mode is ::CU_TR_ADDRESS_MODE_CLAMP.
 
  - ::CUDA_TEXTURE_DESC::filterMode specifies the filtering mode to be used when fetching from the texture. CUfilter_mode is defined as:
    \code
        typedef enum CUfilter_mode_enum {
            CU_TR_FILTER_MODE_POINT = 0,
            CU_TR_FILTER_MODE_LINEAR = 1
        } CUfilter_mode;
    \endcode
    This is ignored if ::CUDA_RESOURCE_DESC::resType is ::CU_RESOURCE_TYPE_LINEAR.
 
  - ::CUDA_TEXTURE_DESC::flags can be any combination of the following:
    - ::CU_TRSF_READ_AS_INTEGER, which suppresses the default behavior of having the texture promote integer data to floating point data in the
      range [0, 1]. Note that texture with 32-bit integer format would not be promoted, regardless of whether or not this flag is specified.
    - ::CU_TRSF_NORMALIZED_COORDINATES, which suppresses the default behavior of having the texture coordinates range from [0, Dim) where Dim is
      the width or height of the CUDA array. Instead, the texture coordinates [0, 1.0) reference the entire breadth of the array dimension; Note
      that for CUDA mipmapped arrays, this flag has to be set.
 
  - ::CUDA_TEXTURE_DESC::maxAnisotropy specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be
    clamped to the range [1,16].
 
  - ::CUDA_TEXTURE_DESC::mipmapFilterMode specifies the filter mode when the calculated mipmap level lies between two defined mipmap levels.
 
  - ::CUDA_TEXTURE_DESC::mipmapLevelBias specifies the offset to be applied to the calculated mipmap level.
 
  - ::CUDA_TEXTURE_DESC::minMipmapLevelClamp specifies the lower end of the mipmap level range to clamp access to.
 
  - ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp specifies the upper end of the mipmap level range to clamp access to.
 
 
  The ::CUDA_RESOURCE_VIEW_DESC struct is defined as
  \code
        typedef struct CUDA_RESOURCE_VIEW_DESC_st
        {
            CUresourceViewFormat format;
            size_t width;
            size_t height;
            size_t depth;
            unsigned int firstMipmapLevel;
            unsigned int lastMipmapLevel;
            unsigned int firstLayer;
            unsigned int lastLayer;
        } CUDA_RESOURCE_VIEW_DESC;
  \endcode
  where:
  - ::CUDA_RESOURCE_VIEW_DESC::format specifies how the data contained in the CUDA array or CUDA mipmapped array should
    be interpreted. Note that this can incur a change in size of the texture data. If the resource view format is a block
    compressed format, then the underlying CUDA array or CUDA mipmapped array has to have a base of format ::CU_AD_FORMAT_UNSIGNED_INT32.
    with 2 or 4 channels, depending on the block compressed format. For ex., BC1 and BC4 require the underlying CUDA array to have
    a format of ::CU_AD_FORMAT_UNSIGNED_INT32 with 2 channels. The other BC formats require the underlying resource to have the same base
    format but with 4 channels.
 
  - ::CUDA_RESOURCE_VIEW_DESC::width specifies the new width of the texture data. If the resource view format is a block
    compressed format, this value has to be 4 times the original width of the resource. For non block compressed formats,
    this value has to be equal to that of the original resource.
 
  - ::CUDA_RESOURCE_VIEW_DESC::height specifies the new height of the texture data. If the resource view format is a block
    compressed format, this value has to be 4 times the original height of the resource. For non block compressed formats,
    this value has to be equal to that of the original resource.
 
  - ::CUDA_RESOURCE_VIEW_DESC::depth specifies the new depth of the texture data. This value has to be equal to that of the
    original resource.
 
  - ::CUDA_RESOURCE_VIEW_DESC::firstMipmapLevel specifies the most detailed mipmap level. This will be the new mipmap level zero.
    For non-mipmapped resources, this value has to be zero.::CUDA_TEXTURE_DESC::minMipmapLevelClamp and ::CUDA_TEXTURE_DESC::maxMipmapLevelClamp
    will be relative to this value. For ex., if the firstMipmapLevel is set to 2, and a minMipmapLevelClamp of 1.2 is specified,
    then the actual minimum mipmap level clamp will be 3.2.
 
  - ::CUDA_RESOURCE_VIEW_DESC::lastMipmapLevel specifies the least detailed mipmap level. For non-mipmapped resources, this value
    has to be zero.
 
  - ::CUDA_RESOURCE_VIEW_DESC::firstLayer specifies the first layer index for layered textures. This will be the new layer zero.
    For non-layered resources, this value has to be zero.
 
  - ::CUDA_RESOURCE_VIEW_DESC::lastLayer specifies the last layer index for layered textures. For non-layered resources,
    this value has to be zero.
 
 
  \param pTexObject   - Texture object to create
  \param pResDesc     - Resource descriptor
  \param pTexDesc     - Texture descriptor
  \param pResViewDesc - Resource view descriptor
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuTexObjectDestroy,
  ::cudaCreateTextureObject"
  (ptexobject (:pointer CUtexObject))
  (presdesc (:pointer))
  (ptexdesc (:pointer))
  (presviewdesc (:pointer)))

(cffi:defcfun "cutexobjectdestroy" CUresult
  "\brief Destroys a texture object
 
  Destroys the texture object specified by \p texObject.
 
  \param texObject - Texture object to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuTexObjectCreate,
  ::cudaDestroyTextureObject"
  (texobject CUtexObject))

(cffi:defcfun "cutexobjectgetresourcedesc" CUresult
  "\brief Returns a texture object's resource descriptor
 
  Returns the resource descriptor for the texture object specified by \p texObject.
 
  \param pResDesc  - Resource descriptor
  \param texObject - Texture object
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuTexObjectCreate,
  ::cudaGetTextureObjectResourceDesc,"
  (presdesc (:pointer))
  (texobject CUtexObject))

(cffi:defcfun "cutexobjectgettexturedesc" CUresult
  "\brief Returns a texture object's texture descriptor
 
  Returns the texture descriptor for the texture object specified by \p texObject.
 
  \param pTexDesc  - Texture descriptor
  \param texObject - Texture object
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuTexObjectCreate,
  ::cudaGetTextureObjectTextureDesc"
  (ptexdesc (:pointer CUDA-TEXTURE-DESC))
  (texobject CUtexObject))

(cffi:defcfun "cutexobjectgetresourceviewdesc" CUresult
  "\brief Returns a texture object's resource view descriptor
 
  Returns the resource view descriptor for the texture object specified by \p texObject.
  If no resource view was set for \p texObject, the ::CUDA_ERROR_INVALID_VALUE is returned.
 
  \param pResViewDesc - Resource view descriptor
  \param texObject    - Texture object
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuTexObjectCreate,
  ::cudaGetTextureObjectResourceViewDesc"
  (presviewdesc (:pointer))
  (texobject CUtexObject))

(cffi:defcfun "cusurfobjectcreate" CUresult
  "\brief Creates a surface object
 
  Creates a surface object and returns it in \p pSurfObject. \p pResDesc describes
  the data to perform surface loadstores on. ::CUDA_RESOURCE_DESC::resType must be
  ::CU_RESOURCE_TYPE_ARRAY and  ::CUDA_RESOURCE_DESC::res::array::hArray
  must be set to a valid CUDA array handle. ::CUDA_RESOURCE_DESC::flags must be set to zero.
 
  Surface objects are only supported on devices of compute capability 3.0 or higher.
  Additionally, a surface object is an opaque value, and, as such, should only be
  accessed through CUDA API calls.
 
  \param pSurfObject - Surface object to create
  \param pResDesc    - Resource descriptor
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuSurfObjectDestroy,
  ::cudaCreateSurfaceObject"
  (psurfobject (:pointer CUsurfObject))
  (presdesc (:pointer)))

(cffi:defcfun "cusurfobjectdestroy" CUresult
  "\brief Destroys a surface object
 
  Destroys the surface object specified by \p surfObject.
 
  \param surfObject - Surface object to destroy
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuSurfObjectCreate,
  ::cudaDestroySurfaceObject"
  (surfobject CUsurfObject))

(cffi:defcfun "cusurfobjectgetresourcedesc" CUresult
  "\brief Returns a surface object's resource descriptor
 
  Returns the resource descriptor for the surface object specified by \p surfObject.
 
  \param pResDesc   - Resource descriptor
  \param surfObject - Surface object
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE
 
  \sa
  ::cuSurfObjectCreate,
  ::cudaGetSurfaceObjectResourceDesc"
  (presdesc (:pointer))
  (surfobject CUsurfObject))

(cffi:defcfun "cudevicecanaccesspeer" CUresult
  "\brief Queries if a device may directly access a peer device's memory.
 
  Returns in \p canAccessPeer a value of 1 if contexts on \p dev are capable of
  directly accessing memory from contexts on \p peerDev and 0 otherwise.
  If direct access of \p peerDev from \p dev is possible, then access may be
  enabled on two specific contexts by calling ::cuCtxEnablePeerAccess().
 
  \param canAccessPeer - Returned access capability
  \param dev           - Device from which allocations on \p peerDev are to
                         be directly accessed.
  \param peerDev       - Device on which the allocations to be directly accessed
                         by \p dev reside.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE
  \notefnerr
 
  \sa
  ::cuCtxEnablePeerAccess,
  ::cuCtxDisablePeerAccess,
  ::cudaDeviceCanAccessPeer"
  (canaccesspeer (:pointer :int))
  (dev CUdevice)
  (peerdev CUdevice))

(cffi:defcfun "cuctxenablepeeraccess" CUresult
  "\brief Enables direct access to memory allocations in a peer context.
 
  If both the current context and \p peerContext are on devices which support unified
  addressing (as may be queried using ::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) and same
  major compute capability, then on success all allocations from \p peerContext will
  immediately be accessible by the current context.  See \ref CUDA_UNIFIED for additional
  details.
 
  Note that access granted by this call is unidirectional and that in order to access
  memory from the current context in \p peerContext, a separate symmetric call
  to ::cuCtxEnablePeerAccess() is required.
 
  There is a system-wide maximum of eight peer connections per device.
 
  Returns ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED if ::cuDeviceCanAccessPeer() indicates
  that the ::CUdevice of the current context cannot directly access memory
  from the ::CUdevice of \p peerContext.
 
  Returns ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED if direct access of
  \p peerContext from the current context has already been enabled.
 
  Returns ::CUDA_ERROR_TOO_MANY_PEERS if direct peer access is not possible
  because hardware resources required for peer access have been exhausted.
 
  Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, \p peerContext
  is not a valid context, or if the current context is \p peerContext.
 
  Returns ::CUDA_ERROR_INVALID_VALUE if \p Flags is not 0.
 
  \param peerContext - Peer context to enable direct access to from the current context
  \param Flags       - Reserved for future use and must be set to 0
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
  ::CUDA_ERROR_TOO_MANY_PEERS,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cuDeviceCanAccessPeer,
  ::cuCtxDisablePeerAccess,
  ::cudaDeviceEnablePeerAccess"
  (peercontext CUcontext)
  (flags :unsigned-int))

(cffi:defcfun "cuctxdisablepeeraccess" CUresult
  "\brief Disables direct access to memory allocations in a peer context and
  unregisters any registered allocations.
 
  Returns ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED if direct peer access has
  not yet been enabled from \p peerContext to the current context.
 
  Returns ::CUDA_ERROR_INVALID_CONTEXT if there is no current context, or if
  \p peerContext is not a valid context.
 
  \param peerContext - Peer context to disable direct access to
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  \notefnerr
 
  \sa
  ::cuDeviceCanAccessPeer,
  ::cuCtxEnablePeerAccess,
  ::cudaDeviceDisablePeerAccess"
  (peercontext CUcontext))

(cffi:defcfun "cudevicegetp2pattribute" CUresult
  "\brief Queries attributes of the link between two devices.
 
  Returns in \p value the value of the requested attribute \p attrib of the
  link between \p srcDevice and \p dstDevice. The supported attributes are:
  - ::CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK: A relative value indicating the
    performance of the link between two devices.
  - ::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED P2P: 1 if P2P Access is enable.
  - ::CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED: 1 if Atomic operations over
    the link are supported.
  - ::CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: 1 if cudaArray can
    be accessed over the link.
 
  Returns ::CUDA_ERROR_INVALID_DEVICE if \p srcDevice or \p dstDevice are not valid
  or if they represent the same device.
 
  Returns ::CUDA_ERROR_INVALID_VALUE if \p attrib is not valid or if \p value is
  a null pointer.
 
  \param value         - Returned value of the requested attribute
  \param attrib        - The requested attribute of the link between \p srcDevice and \p dstDevice.
  \param srcDevice     - The source device of the target link.
  \param dstDevice     - The destination device of the target link.
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_DEVICE,
  ::CUDA_ERROR_INVALID_VALUE
  \notefnerr
 
  \sa
  ::cuCtxEnablePeerAccess,
  ::cuCtxDisablePeerAccess,
  ::cuDeviceCanAccessPeer,
  ::cudaDeviceGetP2PAttribute"
  (value (:pointer :int))
  (attrib CUdevice-P2PAttribute)
  (srcdevice CUdevice)
  (dstdevice CUdevice))

(cffi:defcfun "cugraphicsunregisterresource" CUresult
  "\brief Unregisters a graphics resource for access by CUDA
 
  Unregisters the graphics resource \p resource so it is not accessible by
  CUDA unless registered again.
 
  If \p resource is invalid then ::CUDA_ERROR_INVALID_HANDLE is
  returned.
 
  \param resource - Resource to unregister
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_UNKNOWN
  \notefnerr
 
  \sa
  ::cuGraphicsD3D9RegisterResource,
  ::cuGraphicsD3D10RegisterResource,
  ::cuGraphicsD3D11RegisterResource,
  ::cuGraphicsGLRegisterBuffer,
  ::cuGraphicsGLRegisterImage,
  ::cudaGraphicsUnregisterResource"
  (resource CUgraphicsResource))

(cffi:defcfun "cugraphicssubresourcegetmappedarray" CUresult
  "\brief Get an array through which to access a subresource of a mapped graphics resource.
 
  Returns in \p pArray an array through which the subresource of the mapped
  graphics resource \p resource which corresponds to array index \p arrayIndex
  and mipmap level \p mipLevel may be accessed.  The value set in \p pArray may
  change every time that \p resource is mapped.
 
  If \p resource is not a texture then it cannot be accessed via an array and
  ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
  If \p arrayIndex is not a valid array index for \p resource then
  ::CUDA_ERROR_INVALID_VALUE is returned.
  If \p mipLevel is not a valid mipmap level for \p resource then
  ::CUDA_ERROR_INVALID_VALUE is returned.
  If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
 
  \param pArray      - Returned array through which a subresource of \p resource may be accessed
  \param resource    - Mapped resource to access
  \param arrayIndex  - Array index for array textures or cubemap face
                       index as defined by ::CUarray_cubemap_face for
                       cubemap textures for the subresource to access
  \param mipLevel    - Mipmap level for the subresource to access
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_MAPPED,
  ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  \notefnerr
 
  \sa
  ::cuGraphicsResourceGetMappedPointer,
  ::cudaGraphicsSubResourceGetMappedArray"
  (parray (:pointer CUarray))
  (resource CUgraphicsResource)
  (arrayindex :unsigned-int)
  (miplevel :unsigned-int))

(cffi:defcfun "cugraphicsresourcegetmappedmipmappedarray" CUresult
  "\brief Get a mipmapped array through which to access a mapped graphics resource.
 
  Returns in \p pMipmappedArray a mipmapped array through which the mapped graphics
  resource \p resource. The value set in \p pMipmappedArray may change every time
  that \p resource is mapped.
 
  If \p resource is not a texture then it cannot be accessed via a mipmapped array and
  ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY is returned.
  If \p resource is not mapped then ::CUDA_ERROR_NOT_MAPPED is returned.
 
  \param pMipmappedArray - Returned mipmapped array through which \p resource may be accessed
  \param resource        - Mapped resource to access
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_VALUE,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_MAPPED,
  ::CUDA_ERROR_NOT_MAPPED_AS_ARRAY
  \notefnerr
 
  \sa
  ::cuGraphicsResourceGetMappedPointer,
  ::cudaGraphicsResourceGetMappedMipmappedArray"
  (pmipmappedarray (:pointer CUmipmappedArray))
  (resource CUgraphicsResource))

(cffi:defcfun ("cugraphicsresourcegetmappedpointer_v2" cugraphicsresourcegetmappedpointer-v2) CUresult
  (pdevptr (:pointer CUdeviceptr))
  (psize (:pointer size-t))
  (resource CUgraphicsResource))

(cffi:defcfun ("cugraphicsresourcesetmapflags_v2" cugraphicsresourcesetmapflags-v2) CUresult
  (resource CUgraphicsResource)
  (flags :unsigned-int))

(cffi:defcfun "cugraphicsmapresources" CUresult
  "\brief Map graphics resources for access by CUDA
 
  Maps the \p count graphics resources in \p resources for access by CUDA.
 
  The resources in \p resources may be accessed by CUDA until they
  are unmapped. The graphics API from which \p resources were registered
  should not access any resources while they are mapped by CUDA. If an
  application does so, the results are undefined.
 
  This function provides the synchronization guarantee that any graphics calls
  issued before ::cuGraphicsMapResources() will complete before any subsequent CUDA
  work issued in \p stream begins.
 
  If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
  If any of \p resources are presently mapped for access by CUDA then ::CUDA_ERROR_ALREADY_MAPPED is returned.
 
  \param count      - Number of resources to map
  \param resources  - Resources to map for CUDA usage
  \param hStream    - Stream with which to synchronize
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_ALREADY_MAPPED,
  ::CUDA_ERROR_UNKNOWN
  \note_null_stream
  \notefnerr
 
  \sa
  ::cuGraphicsResourceGetMappedPointer,
  ::cuGraphicsSubResourceGetMappedArray,
  ::cuGraphicsUnmapResources,
  ::cudaGraphicsMapResources"
  (count :unsigned-int)
  (resources (:pointer CUgraphicsResource))
  (hstream CUstream))

(cffi:defcfun "cugraphicsunmapresources" CUresult
  "\brief Unmap graphics resources.
 
  Unmaps the \p count graphics resources in \p resources.
 
  Once unmapped, the resources in \p resources may not be accessed by CUDA
  until they are mapped again.
 
  This function provides the synchronization guarantee that any CUDA work issued
  in \p stream before ::cuGraphicsUnmapResources() will complete before any
  subsequently issued graphics work begins.
 
 
  If \p resources includes any duplicate entries then ::CUDA_ERROR_INVALID_HANDLE is returned.
  If any of \p resources are not presently mapped for access by CUDA then ::CUDA_ERROR_NOT_MAPPED is returned.
 
  \param count      - Number of resources to unmap
  \param resources  - Resources to unmap
  \param hStream    - Stream with which to synchronize
 
  \return
  ::CUDA_SUCCESS,
  ::CUDA_ERROR_DEINITIALIZED,
  ::CUDA_ERROR_NOT_INITIALIZED,
  ::CUDA_ERROR_INVALID_CONTEXT,
  ::CUDA_ERROR_INVALID_HANDLE,
  ::CUDA_ERROR_NOT_MAPPED,
  ::CUDA_ERROR_UNKNOWN
  \note_null_stream
  \notefnerr
 
  \sa
  ::cuGraphicsMapResources,
  ::cudaGraphicsUnmapResources"
  (count :unsigned-int)
  (resources (:pointer CUgraphicsResource))
  (hstream CUstream))

(cffi:defcfun "cugetexporttable" CUresult
  "@}"
  (ppexporttable (:pointer (:pointer :void)))
  (pexporttableid (:pointer CUuuid)))

