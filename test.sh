#! /bin/sh
#
# test.sh
# Copyright (C) 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.
#


sbcl --eval "(progn (ql:quickload :petalisp-cuda)(ql:quickload :petalisp-cuda/tests)(asdf:test-system :petalisp-cuda/tests))"
