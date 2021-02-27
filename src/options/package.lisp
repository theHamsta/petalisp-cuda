(defpackage petalisp-cuda.options
  (:use :cl)
  (:export :*silence-cl-cuda*
           :*transfer-back-to-lisp*
           :*single-threaded*
           :*single-stream*
           :*nvcc-extra-options*
           :*shape-independent-code*
           :*generic-offsets*
           :*with-hash-table-memoization*
           :*page-locked-host-memory*
           :*strict-cast-mode*
           :*preferred-block-shape*
           :*slow-coordinate-load-strategy*
           :*warp-time-slicing*
           :*slow-coordinate-transposed-trick*
           :*cudnn-autotune*
           :*transfer-back-to-lisp*
           :*cudnn-default-math-type*))
(in-package petalisp-cuda.options)

(defparameter *silence-cl-cuda* t)
(defparameter *transfer-back-to-lisp* t)
(defparameter *single-threaded* t)
(defparameter *single-stream* t)
(defparameter *nvcc-extra-options* '("-use_fast_math" "--std=c++14" "-Xptxas" "-O3" "--expt-relaxed-constexpr" "--extra-device-vectorization" "-Wno-deprecated-gpu-targets"))
(defparameter *shape-independent-code* t)
(defparameter *generic-offsets* t)
(defparameter *with-hash-table-memoization* t)
(defparameter *page-locked-host-memory* t)
(defparameter *slow-coordinate-transposed-trick* nil)

(defparameter *cudnn-autotune* nil
  "Whether to do exhaustive searches for best convolution algorithms (alternative: use heuristics)")

(defparameter *strict-cast-mode* nil
  "Forbids casts from T to single-float to perform calculation on GPU")

(defparameter *max-array-printing-length* 5
  "Number of numbers to display at end and start when printing cuda array")

(defparameter *preferred-block-shape* '(16 16 1))

(defparameter *cudnn-default-math-type* nil
  "Determines which type conversions are allowed to accelerate CUDNN kernels with tensor cores")
