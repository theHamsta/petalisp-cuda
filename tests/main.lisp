(defpackage petalisp-cuda/tests
  (:use :cl
        :petalisp-cuda
        :petalisp-cuda.memory.cuda-array
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


(deftest test-descriptor
  (with-cuda (0)
    (petalisp-cuda.cudalibs::cudnn-create-tensor-descriptor
      (make-cuda-array '(10 20) 'float))))

(deftest test-descriptor
  (with-cuda (0)
    (progn
      (petalisp-cuda:use-cuda-backend)
      (let ((a (make-cuda-array '(10 20) 'float))
            (b make-cuda-array '(10 1) 'float))
        (petalisp-cuda.cudalibs:cudnn-array a b #'+)))))

