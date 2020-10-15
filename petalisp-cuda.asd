(defsystem "petalisp-cuda"
  :version "0.1.0"
  :author "Stephan Seitz <stephan.seitz@fau.de>"
  :license "GPLv3"
  :depends-on ("petalisp"
               "petalisp.core"
               "bordeaux-threads"
               "petalisp.ir"
               "cl-cuda"
               "iterate"
               "array-operations"
               "cffi"
               "cffi-libffi"
               "trivia"
               "alexandria"
               "trivial-garbage"
               "cl-itertools"
               "let-plus")
  :components ((:module "src/utils"
                :components
                ((:file "cl-cuda")
                 (:file "petalisp")))
               (:module "src/memory"
                :components
                ((:file "cuda-array")
                 (:file "memory-pool")))
               (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 (:file "cudnn")
                 (:file "cudnn-handler")))
               (:module "src/iteration-scheme"
                :components
                ((:file "package")
                 (:file "helpers")
                 (:file "block-iteration-scheme")
                 (:file "slow-coordinate-transposed-scheme")
                 (:file "selection")))
               (:module "src"
                :components
                ((:file "device")
                 (:file "type-conversion")
                 (:file "backend")
                 (:file "jit-execution")
                 (:file "map-call-operator")
                 (:file "package")))))


(defsystem "petalisp-cuda/tests"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("petalisp-cuda"
               "petalisp"
               "petalisp.test-suite"
               "petalisp.core"
               "cl-cuda"
               "array-operations"
               "iterate"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "package")
                 (:file "testing-backend")
                 (:file "main"))))
  :description "Test system for petalisp-cuda"

  :perform (test-op (op c) (symbol-call :rove :run c)))
