(defsystem "petalisp-cuda"
  :version "0.1.0"
  :author "Stephan Seitz <stephan.seitz@fau.de>"
  :license "GPLv3"
  :depends-on ("petalisp"
               "cl-cuda"
               "iterate"
               "array-operations"
               "cffi"
               "cffi-libffi")
  :components ((:module "src"
                :components
                ((:file "main")
                 (:file "cuda-array")
                 (:file "backend")
                 ))
                (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 ))))


(defsystem "petalisp-cuda/tests"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("petalisp-cuda"
               "petalisp"
               "petalisp.core"
               "cl-cuda"
               "array-operations"
               "iterate"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for petalisp-cuda"

  :perform (test-op (op c) (symbol-call :rove :run c)))
