(in-package petalisp-cuda.iteration-scheme)

;; same as block-iteration-scheme, but uses a trick to cache slow coordinate accesses
;; https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
(defclass slow-coordinate-transposed-scheme (block-iteration-scheme)
  ())

