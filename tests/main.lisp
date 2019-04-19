(defpackage betalisp/tests/main
  (:use :cl
        :betalisp
        :rove))
(in-package :betalisp/tests/main)

;; NOTE: To run this test file, execute `(asdf:test-system :betalisp)' in your Lisp.

(deftest test-target-1
  (testing "should (= 1 1) to be true"
    (ok (= 1 1))))
