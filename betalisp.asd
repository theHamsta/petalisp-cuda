(defsystem "betalisp"
  :version "0.1.0"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("petalisp"
               "cl-cuda"
               "iterate")
  :components ((:module "src"
                :components
                ((:file "main") (:file "cuda-array"))))
  :description ""
  :long-description
  #.(read-file-string
     (subpathname *load-pathname* "README.org"))
  :in-order-to ((test-op (test-op "betalisp/tests"))))

;(defsystem "betalisp/tests"
  ;:author "Stephan Seitz"
  ;:license "GPLv3"
  ;:depends-on ("betalisp"
               ;"rove")
  ;:components ((:module "tests"
                ;:components
                ;((:file "main"))))
  ;:description "Test system for betalisp"

  ;:perform (test-op (op c) (symbol-call :rove :run c)))
