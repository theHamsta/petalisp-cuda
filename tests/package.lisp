(defpackage petalisp-cuda/tests
  (:use :cl
        :petalisp
        :petalisp.core
        :petalisp-cuda
        :cl-cuda
        :rove
        :petalisp-cuda.cudnn-ops)
  (:import-from :petalisp.examples.iterative-methods
                :jacobi
                :rbgs
                :v-cycle)
  (:import-from :petalisp-cuda.memory.cuda-array
                :make-cuda-array
                :free-cuda-array
                :cuda-array-device
                :cuda-array-strides
                :cuda-array-size
                :cuda-array-shape)
  (:import-from :petalisp.examples.linear-algebra
                :norm
                :matmul
                :transpose
                :LU
                :eye
                :pivot-and-value)
  (:import-from :petalisp.test-suite
                :ndarray
                :approximately-equal
                :generate-matrix)
  (:import-from :petalisp-cuda.memory.memory-pool
                :array-table
                :allocated-cuda-arrays)
  (:import-from :petalisp-cuda.cudnn-ops
                :lazy-convolution
                :lazy-reduction)
  (:import-from :petalisp-cuda.cuda-immediate
                :cuda-immediate-p)
  (:export :run-tests))
(in-package :petalisp-cuda/tests)
