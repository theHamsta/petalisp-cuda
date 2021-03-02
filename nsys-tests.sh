#! /bin/sh
#
# nsys-tests.sh
# Copyright (C) 2021 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
#


nsys profile -t nvtx,cuda,cudnn,cublas --force-overwrite=true --stats true --output=myapp sbcl --eval '(setf *debugger-hook* (lambda (c h) (declare (ignore c h)) (uiop:quit -1)))'  --eval "(progn (ql:quickload :petalisp-cuda)(ql:quickload :petalisp-cuda/tests) (asdf:test-system :petalisp-cuda/tests) (exit))"
