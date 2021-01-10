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
