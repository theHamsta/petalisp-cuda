(defsystem "petalisp-cuda"
  :version "0.1.0"
  :author "Stephan Seitz <stephan.seitz@fau.de>"
  :license "GPLv3"
  :depends-on ("petalisp"
               "petalisp.core"
               "petalisp.ir"
               "cl-cuda"
               "iterate"
               "array-operations"
               "cffi"
               "cffi-libffi"
               "trivia"
               "alexandria"
               "cl-pattern")
  :components ((:module "src/cuda-array"
                :components
                ((:file "cuda-array")))
               (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 (:file "cudnn")
                 (:file "package")))
               (:module "src"
                :components
                ((:file "backend")
                 (:file "package")))))


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
