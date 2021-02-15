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
  (let ((a (α #'+ #2A((2 2) (2 2)) #2A((2 2) (2 2))))
        (b #2A((2))))
    (compute
      (petalisp.graphviz:view
        (petalisp.ir:ir-from-lazy-arrays
          (list
            (α #'+ a
             (β #'max
               (lazy-convolution
                 a
                 b))))))))
  (compute 
    (petalisp.graphviz:view
      (petalisp.ir:ir-from-lazy-arrays
        (list
          (α #'+ 2 (lazy-convolution #2A((2 2)
                                          (2 2))
                                     #2A((2)))))))))
