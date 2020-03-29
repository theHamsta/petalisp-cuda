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
               "cl-itertools")
  :components ((:module "src/memory"
                :components
                ((:file "memory-pool")
                 (:file "cuda-array")))
               (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 (:file "cudnn")
                 (:file "cudnn-handler")))
               (:module "src"
                :components
                ((:file "device")
                 (:file "backend")
                 (:file "indexing")
                 (:file "code-generation")
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

  :perform (test-op (op c) (with-standard-io-syntax (symbol-call :rove :run c))))
