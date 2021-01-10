(in-package :petalisp-cuda/tests)

(deftest test-descriptor
  (unless petalisp-cuda.cudalibs::*cudnn-found*
    (skip "No cudnn!"))
  (with-cuda-backend
    (petalisp-cuda.cudnn-handler::cudnn-create-tensor-descriptor
      (make-cuda-array '(10 20) 'float)
      (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))))

(deftest test-reduction
  (unless petalisp-cuda.cudalibs::*cudnn-found*
    (skip "No cudnn!"))
  (approximately-equal
    (compute (let ((petalisp-cuda.options:*transfer-back-to-lisp* t))
              (with-cuda-backend
    (let ((a (make-cuda-array '(10 20) 'float))
            (b (make-cuda-array '(10 1) 'float)))
        (petalisp-cuda.cudnn-handler::cudnn-reduce-array a b #'+ (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*)))
    (let ((a (make-cuda-array #2A ((2 3 4)
                                   (1 2 3))
                              'float))
            (b (make-cuda-array '(2 1) 'float)))
        (petalisp-cuda.cudnn-handler::cudnn-reduce-array a b #'+ (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
      b))))
    #2a ((9) (6))))

(deftest test-convolution
  (unless petalisp-cuda.cudalibs::*cudnn-found*
    (skip "No cudnn!"))
  (with-cuda-backend
    (let ((x (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
          (y (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
          (w (make-cuda-array #4A((((1 2 2) (2 4 2) (2 3 2)))) 'float)))
      (petalisp-cuda.cudnn-handler::cudnn-convolution x w y (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
      y))
  (with-cuda-backend
    (let ((x (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
          (y (make-cuda-array (aops:rand* 'float '(1 1 12 13)) 'float))
          (w (make-cuda-array #4A((((1 2) (3 4)))) 'float))
          (petalisp-cuda.options:*cudnn-autotune* t))
      (petalisp-cuda.cudnn-handler::cudnn-convolution x
                                                      w
                                                      y
                                                      (petalisp-cuda.backend::cudnn-handler petalisp-cuda.backend::*cuda-backend*))
      y)))
