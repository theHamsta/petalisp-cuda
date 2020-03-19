(defpackage petalisp-cuda/tests
  (:use :cl
        :petalisp-cuda
        :petalisp-cuda.cuda-array
        :cl-cuda
        :rove))
(in-package :petalisp-cuda/tests)

;; NOTE: To run this test file, execute `(asdf:test-system :petalisp-cuda)' in your Lisp.

(deftest test-make-cuda-backend
  (make-instance 'petalisp-cuda.backend:cuda-backend))

(deftest test-make-cuda-backend2
  (with-cuda (0)
    (make-instance 'petalisp-cuda.backend:cuda-backend)))

(deftest test-make-cuda-array
  (with-cuda (0)
    (make-cuda-array '(10 20) 'float)))


