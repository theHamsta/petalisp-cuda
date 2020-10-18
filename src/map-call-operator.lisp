(in-package petalisp-cuda.jitexecution)

(defun map-call-operator (operator arguments)
  ;; LHS: Petalisp/code/type-inference/package.lisp
  ;; RHS: cl-cuda/src/lang/built-in.lisp
  (alexandria:switch (operator :test #'equal)
    ('+ '+)
    (#'+ '+)
    ('petalisp.type-inference:the-number '+)
    ('petalisp.type-inference:double-float+ '+)
    ('petalisp.type-inference:single-float+ '+)
    ('petalisp.type-inference:short-float+ '+)
    ('petalisp.type-inference:long-float+ '+)

    ('- '-)
    (#'- '-)
    ('petalisp.type-inference:double-float- '-)
    ('petalisp.type-inference:single-float- '-)
    ('petalisp.type-inference:short-float- '-)
    ('petalisp.type-inference:long-float- '-)

    ('* '*)
    (#'* '*)
    ('petalisp.type-inference:double-float* '*)
    ('petalisp.type-inference:single-float* '*)
    ('petalisp.type-inference:short-float* '*)
    ('petalisp.type-inference:long-float* '*)

    ('/ '/)
    (#'/ '/)
    ('petalisp.type-inference:double-float/ '/)
    ('petalisp.type-inference:single-float/ '/)
    ('petalisp.type-inference:short-float/ '/)
    ('petalisp.type-inference:long-float/ '/)

    ('= '==)
    (#'= '==)
    ;(CMPEQ '==)
    ('petalisp.type-inference:double-float= '==)
    ('petalisp.type-inference:single-float= '==)
    ('petalisp.type-inference:short-float= '==)
    ('petalisp.type-inference:long-float= '==)

    ('> '>)
    (#'> '>)
    ('petalisp.type-inference:double-float> '>)
    ('petalisp.type-inference:single-float> '>)
    ('petalisp.type-inference:short-float> '>)
    ('petalisp.type-inference:long-float> '>)

    ('< '<)
    (#'< '<)
    ('petalisp.type-inference:double-float< '<)
    ('petalisp.type-inference:single-float< '<)
    ('petalisp.type-inference:short-float< '<)
    ('petalisp.type-inference:long-float< '<)

    ('<= '<=)
    (#'<= '<=)
    ('petalisp.type-inference:double-float<= '<=)
    ('petalisp.type-inference:single-float<= '<=)
    ('petalisp.type-inference:short-float<= '<=)
    ('petalisp.type-inference:long-float<= '<=)

    ('>= '>=)
    (#'>= '>=)
    ('petalisp.type-inference:double-float>= '>=)
    ('petalisp.type-inference:single-float>= '>=)
    ('petalisp.type-inference:short-float>= '>=)
    ('petalisp.type-inference:long-float>= '>=)

    ('min 'min)
    (#'min 'min)
    ('petalisp.type-inference:double-float-min 'min)
    ('petalisp.type-inference:single-float-min 'min)
    ('petalisp.type-inference:short-float-min 'min)
    ('petalisp.type-inference:long-float-min 'min)

    ('max 'max)
    (#'max 'max)
    ('petalisp.type-inference:double-float-max 'max)
    ('petalisp.type-inference:single-float-max 'max)
    ('petalisp.type-inference:short-float-max 'max)
    ('petalisp.type-inference:long-float-max 'max)

    ('abs 'abs)
    (#'abs 'abs)
    ('petalisp.type-inference:double-float-abs 'abs)
    ('petalisp.type-inference:single-float-abs 'abs)
    ('petalisp.type-inference:short-float-abs 'abs)
    ('petalisp.type-inference:long-float-abs 'abs)

    ('cos 'cos)
    (#'cos 'cos)
    ('petalisp.type-inference:double-float-cos 'cos)
    ('petalisp.type-inference:single-float-cos 'cos)
    ('petalisp.type-inference:short-float-cos 'cos)
    ('petalisp.type-inference:long-float-cos 'cos)

    ('sin 'sin)
    (#'sin 'sin)
    ('petalisp.type-inference:double-float-sin 'sin)
    ('petalisp.type-inference:single-float-sin 'sin)
    ('petalisp.type-inference:short-float-sin 'sin)
    ('petalisp.type-inference:long-float-sin 'sin)

    ('tan 'tan)
    (#'tan 'tan)
    ('petalisp.type-inference:double-float-tan 'tan)
    ('petalisp.type-inference:single-float-tan 'tan)
    ('petalisp.type-inference:short-float-tan 'tan)
    ('petalisp.type-inference:long-float-tan 'tan)

    ('exp 'exp)
    (#'exp 'exp)
    ('petalisp.type-inference::double-float-exp 'exp)
    ('petalisp.type-inference::single-float-exp 'exp)
    ('petalisp.type-inference::short-float-exp 'exp)
    ('petalisp.type-inference::long-float-exp 'exp)

    ('log 'log)
    (#'log 'log)
    ('petalisp.type-inference::double-float-ln 'log)
    ('petalisp.type-inference::single-float-ln 'logf)
    ('petalisp.type-inference::short-float-ln 'logf)
    ('petalisp.type-inference::long-float-ln 'log)

    ('floor 'floor)
    (#'floor 'floor)
    ('ceiling 'ceiling)
    (#'ceiling 'ceiling)
    ('sqrt 'sqrt)
    (#'sqrt 'sqrt)
    (#'rem 'rem)

    (t (let ((source-form (when (functionp operator) (function-lambda-expression operator))))
         (if source-form
             (let* ((lambda-arguments (nth 1 source-form))
                    (lambda-body? (last source-form))
                    (lambda-body (if (equal 'BLOCK (caar lambda-body?)) (car (last (first lambda-body?))) (car lambda-body?))))
               (when cl-cuda:*show-messages*
                 (format t "Creating kernel lambda: ~A~%" lambda-body))
               (loop for ir-arg in arguments
                     for arg in lambda-arguments do
                     (setf lambda-body (subst ir-arg arg lambda-body)))
               (values nil (analyze-multiple-value-lambda lambda-body)))
         (error "Cannot convert Petalisp instruction ~A to cl-cuda instruction.
More copy paste required here!~%
You may also try to compile a pure function with (debug 3) so that petalisp-cuda can retrieve its source from." operator))))))

(defun num-values (tree)
  (trivia:match tree
                ((cons 'values rest) (length rest))
                ((cons a b) (max (num-values a) (num-values b)))
                (_ 1)))

(defun nth-value-lambda (n tree)
  (trivia:match tree
                ((cons 'values rest) (nth n rest))
                ((cons a b) (cons (nth-value-lambda n a) (nth-value-lambda n b)))
                (a a)))

(defun analyze-multiple-value-lambda (lambda-body)
  (loop for i from 0 to (1- (num-values lambda-body))
        collect (nth-value-lambda i lambda-body)))

