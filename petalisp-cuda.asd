(defsystem "petalisp-cuda"
  :version "0.1.0"
  :author "Stephan Seitz <stephan.seitz@fau.de>"
  :license "GPLv3"
  :serial t
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
               "lparallel"
               "alexandria"
               ;"trivial-garbage"
               "hash-set"
               "cl-itertools"
               "let-plus")
  :components ((:module "src/options"
                :components
                ((:file "package")))
               (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 (:file "cudnn")))
               (:module "src/utils"
                :components
                ((:file "cl-cuda")
                 (:file "petalisp")))
               (:module "src/memory"
                :components
                ((:file "cuda-array")
                 (:file "memory-pool")))
               (:module "src/cudnn-handler"
                :components
                ((:file "cudnn-handler")))
               (:module "src/type-conversion"
                :components
                 ((:file "type-conversion")))
               (:module "src/iteration-scheme"
                :components
                ((:file "package")
                 (:file "helpers")
                 (:file "block-iteration-scheme")
                 (:file "symbolic-block-iteration-scheme")
                 (:file "slow-coordinate-transposed-scheme")
                 (:file "selection")))
               (:module "src"
                :components
                ((:file "device")
                 (:file "cl-cuda-functions")
                 (:file "cuda-immediate")
                 (:file "backend")
                 (:file "jit-execution")
                 (:file "map-call-operator")
                 (:file "package")))))


(defsystem "petalisp-cuda/tests"
  :author "Stephan Seitz"
  :license "GPLv3"
  :serial t
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
                 (:file "main")
                 (:file "test-cuda-array")
                 (:file "test-cudnn"))))
  :description "Test system for petalisp-cuda"
  :perform (test-op (op c) (symbol-call :rove :run c)))
