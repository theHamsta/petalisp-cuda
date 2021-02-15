(defpackage petalisp-cuda.custom-op
  (:use :cl
        :petalisp
        :petalisp.ir
        :petalisp.core)
  (:import-from :petalisp.ir
                :dendrite-cons
                :ensure-cluster
                :cluster-dendrites
                :dendrite-depth
                :dendrite
                :copy-dendrite
                :grow-dendrite
                :dendrite-shape
                :make-cluster
                :make-dendrite
                :dendrite-transformation
                :buffer-readers
                :kernel-sources
                :dendrite-kernel
                :dendrite-stem
                :make-load-instruction
                :ir-converter-pqueue
                :ir-converter-cluster-table
                :ir-converter-scalar-table
                :stem-kernel
                :make-store-instruction
                :finalize-kernel
                :kernel
                :*ir-converter*)
  (:export :lazy-custom-op
           :lazy-custom-op-execute
           :custom-op-kernel
           :custom-op-kernel-execute
           :custom-op-kernel-p))
(in-package petalisp-cuda.custom-op)
