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
                ((:file "cl-cuda")))
               (:module "src/memory"
                :components
                ((:file "cuda-array")
                 (:file "memory-pool")))
               (:module "src/cudalibs"
                :components
                ((:file "cuda")
                 (:file "cudnn")
                 (:file "cudnn-handler")))
               (:module "src"
                :components
                ((:file "device")
                 (:file "type-conversion")
                 (:file "backend")
                 (:file "iteration-scheme")
                 (:file "jit-execution")
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
