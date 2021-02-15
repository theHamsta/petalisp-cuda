(in-package :petalisp-cuda/tests)

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
                 b)))
            )))))
  (compute 
    (petalisp.graphviz:view
      (petalisp.ir:ir-from-lazy-arrays
        (list
          (α #'+ 2 (lazy-convolution #2A((2 2)
                                          (2 2))
                                     #2A((2)))))))))
