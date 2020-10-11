(defpackage petalisp-cuda/tests
  (:use :cl
        :petalisp
        :petalisp.core
        :petalisp-cuda
        :cl-cuda
        :rove)
  (:import-from :petalisp.examples.iterative-methods
                :jacobi
                :rbgs
                :v-cycle)
  (:import-from :petalisp-cuda.memory.cuda-array :make-cuda-array)
  (:import-from :petalisp.test-suite :ndarray)
  (:export :run-tests))
(in-package :petalisp-cuda/tests)