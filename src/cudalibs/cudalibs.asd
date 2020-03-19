(defsystem "petalisp-cuda.cudalibs"
  :author "Stephan Seitz"
  :license "GPLv3"
  :depends-on ("cffi")
  :serial t
  :components ((:components
                ((:file "cuda")
                 (:file "cudnn")))))
