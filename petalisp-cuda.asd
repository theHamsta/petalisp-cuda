(defsystem "petalisp-cuda"
  :version "0.1.0"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("petalisp"
               "cl-cuda"
               "iterate")
  :components ((:module "src"
                :components
                ((:file "main")
                 (:file "cuda-array"))))
  :description ""
  :long-description
  #.(read-file-string
     (subpathname *load-pathname* "README.org"))
  :in-order-to ((test-op (test-op "petalisp-cuda/tests"))))

(defsystem "petalisp-cuda/tests"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("petalisp-cuda"
               "petalisp"
               "cl-cuda"
               "iterate"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for petalisp-cuda"

  :perform (test-op (op c) (symbol-call :rove :run c)))
