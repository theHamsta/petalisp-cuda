(in-package :petalisp-cuda/tests)

(deftest test-lazy-convolution-compute
  (let ((a (α #'+ #4A((((2 2) (2 2)))) #4A((((2 2) (2 2))))))
        (b #4A((((2))))))
    (with-cuda-backend
      (compute
      (α #'+ a
           (β #'max
              (lazy-convolution
                a
                b)))))))

(deftest test-lazy-reduction
  (with-cuda-backend
    (compute
      (α #'+ 2 (lazy-reduction (α #'+ 2 #2A((2 3) (4 5))) (~) #'+)))))

(deftest test-lazy-convolution
  (let ((a (α #'+ #4A((((2 2) (2 2)))) #4A((((2 2) (2 2))))))
        (b #4A((((2))))))
    (with-cuda-backend
      (compute
           (α #'+ a
             (β #'max
                (lazy-convolution
                  a
                  b))))))
          ;(α #'+ 2 (lazy-convolution #4A((((2 2)
                                            ;(2 2))))
                                     ;#4A((((2))))))
        )

(deftest unnormalizing-transformation
  (ok
    (equalp
      (~ 1 ~ 7 ~ 4)
      (let ((normalized (transform
                          (~ 1 ~ 2 9 ~ 3 7)
                          (normalizing-transformation (~ 1 ~ 2 9 ~ 3 7)))))
        (transform
          normalized
          (petalisp-cuda.cudnn-ops::unnormalizing-transformation
            normalized
            (~ 1 ~ 2 9 ~ 3 7))))))
  (ok
    (equalp
      (~ 1 ~ 7 ~ 4)
      (let ((collapsed (transform
                          (~ 1 ~ 2 9 ~ 3 7)
                          (collapsing-transformation (~ 1 ~ 2 9 ~ 3 7)))))
        (transform
          collapsed
          (petalisp-cuda.cudnn-ops::unnormalizing-transformation
            collapsed
            (~ 1 ~ 2 9 ~ 3 7)))))))
