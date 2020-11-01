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
           :*page-locked-host-memory*))
(in-package petalisp-cuda.options)

(defparameter *silence-cl-cuda* t)
(defparameter *transfer-back-to-lisp* nil)
(defparameter *single-threaded* t)
(defparameter *single-stream* t)
(defparameter *nvcc-extra-options* '("-use_fast_math" "--std" "c++14" "-Xptxas" "-O3" "--expt-relaxed-constexpr" "--extra-device-vectorization" "-Wno-deprecated-gpu-targets"))
(defparameter *shape-independent-code* t)
(defparameter *generic-offsets* t)
(defparameter *with-hash-table-memoization* t)
(defparameter *page-locked-host-memory* t)

;; Number of numbers to display at end and start when printing cuda array
(defparameter *max-array-printing-length* 5)
