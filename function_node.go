package gosymbol

import "math"

// ============================================================
// Func — named function applications
// ============================================================

type Func struct {
	name string
	arg  Expr
}

func funcOf(name string, arg Expr) *Func { return &Func{name: name, arg: arg} }

func SinOf(arg Expr) Expr   { return funcOf("sin", arg).Simplify() }
func CosOf(arg Expr) Expr   { return funcOf("cos", arg).Simplify() }
func TanOf(arg Expr) Expr   { return funcOf("tan", arg).Simplify() }
func ExpOf(arg Expr) Expr   { return funcOf("exp", arg).Simplify() }
func LnOf(arg Expr) Expr    { return funcOf("ln", arg).Simplify() }
func SqrtOf(arg Expr) Expr  { return PowOf(arg, F(1, 2)) }
func AbsOf(arg Expr) Expr   { return funcOf("abs", arg).Simplify() }
func AsinOf(arg Expr) Expr  { return funcOf("asin", arg).Simplify() }
func AcosOf(arg Expr) Expr  { return funcOf("acos", arg).Simplify() }
func AtanOf(arg Expr) Expr  { return funcOf("atan", arg).Simplify() }
func SinhOf(arg Expr) Expr  { return funcOf("sinh", arg).Simplify() }
func CoshOf(arg Expr) Expr  { return funcOf("cosh", arg).Simplify() }
func TanhOf(arg Expr) Expr  { return funcOf("tanh", arg).Simplify() }
func AsinhOf(arg Expr) Expr { return funcOf("asinh", arg).Simplify() }
func AcoshOf(arg Expr) Expr { return funcOf("acosh", arg).Simplify() }
func AtanhOf(arg Expr) Expr { return funcOf("atanh", arg).Simplify() }
func FloorOf(arg Expr) Expr { return funcOf("floor", arg).Simplify() }
func CeilOf(arg Expr) Expr  { return funcOf("ceil", arg).Simplify() }
func SignOf(arg Expr) Expr  { return funcOf("sign", arg).Simplify() }

func (f *Func) Simplify() Expr {
	if simplificationDepthExceeded(f) {
		return f
	}
	arg := f.arg.Simplify()
	if n, ok := arg.(*Num); ok {
		v, _ := n.val.Float64()
		switch f.name {
		case "sin":
			return NFloat(math.Sin(v))
		case "cos":
			return NFloat(math.Cos(v))
		case "tan":
			return NFloat(math.Tan(v))
		case "exp":
			return NFloat(math.Exp(v))
		case "ln":
			if v > 0 {
				return NFloat(math.Log(v))
			}
		case "abs":
			return NFloat(math.Abs(v))
		case "asin":
			return NFloat(math.Asin(v))
		case "acos":
			return NFloat(math.Acos(v))
		case "atan":
			return NFloat(math.Atan(v))
		case "sinh":
			return NFloat(math.Sinh(v))
		case "cosh":
			return NFloat(math.Cosh(v))
		case "tanh":
			return NFloat(math.Tanh(v))
		case "asinh":
			return NFloat(math.Asinh(v))
		case "acosh":
			return NFloat(math.Acosh(v))
		case "atanh":
			return NFloat(math.Atanh(v))
		case "floor":
			return NFloat(math.Floor(v))
		case "ceil":
			return NFloat(math.Ceil(v))
		case "sign":
			switch {
			case v > 0:
				return N(1)
			case v < 0:
				return N(-1)
			default:
				return N(0)
			}
		}
	}
	switch f.name {
	case "sin":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), SinOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "asin" {
			return inner.arg
		}
	case "cos":
		if isNumEqual(arg, 0) {
			return N(1)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return CosOf(negatedArgument)
		}
		if inner, ok := arg.(*Func); ok && inner.name == "acos" {
			return inner.arg
		}
	case "tan":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), TanOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "atan" {
			return inner.arg
		}
	case "ln":
		if n2, ok := arg.(*Num); ok && n2.IsOne() {
			return N(0)
		}
		if inner, ok := arg.(*Func); ok && inner.name == "exp" {
			return inner.arg
		}
	case "exp":
		if n2, ok := arg.(*Num); ok && n2.IsZero() {
			return N(1)
		}
		if inner, ok := arg.(*Func); ok && inner.name == "ln" {
			return inner.arg
		}
	case "sinh":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), SinhOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "asinh" {
			return inner.arg
		}
	case "cosh":
		if isNumEqual(arg, 0) {
			return N(1)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return CoshOf(negatedArgument)
		}
		if inner, ok := arg.(*Func); ok && inner.name == "acosh" {
			return inner.arg
		}
	case "tanh":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), TanhOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "atanh" {
			return inner.arg
		}
	case "asin":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), AsinOf(negatedArgument)).Simplify()
		}
	case "atan":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), AtanOf(negatedArgument)).Simplify()
		}
	case "asinh":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), AsinhOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "sinh" {
			return inner.arg
		}
	case "atanh":
		if isNumEqual(arg, 0) {
			return N(0)
		}
		if negatedArgument, isNegated := peelLeadingNegative(arg); isNegated {
			return MulOf(N(-1), AtanhOf(negatedArgument)).Simplify()
		}
		if inner, ok := arg.(*Func); ok && inner.name == "tanh" {
			return inner.arg
		}
	case "abs":
		if n2, ok := arg.(*Num); ok && n2.IsPositive() {
			return n2
		}
		if sym, ok := arg.(*Sym); ok && sym.assumptions.Positive {
			return sym
		}
		if m, ok := arg.(*Mul); ok && len(m.factors) >= 1 {
			if coeff, ok2 := m.factors[0].(*Num); ok2 && coeff.IsNegOne() {
				inner := m.factors[1:]
				if len(inner) == 1 {
					return AbsOf(inner[0])
				}
				return AbsOf(MulOf(inner...))
			}
		}
	}
	return &Func{name: f.name, arg: arg}
}

func (f *Func) Canonicalize() Expr { return Canonicalize(f) }

func (f *Func) String() string { return f.name + "(" + f.arg.String() + ")" }

func (f *Func) LaTeX() string {
	switch f.name {
	case "abs":
		return "\\left|" + f.arg.LaTeX() + "\\right|"
	case "floor":
		return "\\lfloor " + f.arg.LaTeX() + " \\rfloor"
	case "ceil":
		return "\\lceil " + f.arg.LaTeX() + " \\rceil"
	case "sign":
		return "\\operatorname{sign}\\left(" + f.arg.LaTeX() + "\\right)"
	}
	return latexFuncName(f.name) + "\\left(" + f.arg.LaTeX() + "\\right)"
}

func (f *Func) Sub(varName string, value Expr) Expr {
	return funcOf(f.name, f.arg.Sub(varName, value)).Simplify()
}

func (f *Func) Diff(varName string) Expr {
	du := f.arg.Diff(varName)
	var outer Expr
	switch f.name {
	case "sin":
		outer = CosOf(f.arg)
	case "cos":
		outer = MulOf(N(-1), SinOf(f.arg))
	case "tan":
		outer = AddOf(N(1), PowOf(TanOf(f.arg), N(2)))
	case "exp":
		outer = ExpOf(f.arg)
	case "ln":
		outer = PowOf(f.arg, N(-1))
	case "asin":
		outer = PowOf(AddOf(N(1), MulOf(N(-1), PowOf(f.arg, N(2)))), F(-1, 2))
	case "acos":
		outer = MulOf(N(-1), PowOf(AddOf(N(1), MulOf(N(-1), PowOf(f.arg, N(2)))), F(-1, 2)))
	case "atan":
		outer = PowOf(AddOf(N(1), PowOf(f.arg, N(2))), N(-1))
	case "sinh":
		outer = CoshOf(f.arg)
	case "cosh":
		outer = SinhOf(f.arg)
	case "tanh":
		outer = AddOf(N(1), MulOf(N(-1), PowOf(TanhOf(f.arg), N(2))))
	case "asinh":
		outer = PowOf(AddOf(PowOf(f.arg, N(2)), N(1)), F(-1, 2))
	case "acosh":
		outer = MulOf(
			PowOf(AddOf(f.arg, N(-1)), F(-1, 2)),
			PowOf(AddOf(f.arg, N(1)), F(-1, 2)),
		)
	case "atanh":
		outer = PowOf(AddOf(N(1), MulOf(N(-1), PowOf(f.arg, N(2)))), N(-1))
	default:
		return MulOf(funcOf("D["+f.name+"]", f.arg), du)
	}
	return MulOf(outer, du).Simplify()
}

func (f *Func) Eval() (*Num, bool) {
	n, ok := f.arg.Eval()
	if !ok {
		return nil, false
	}
	v, _ := n.val.Float64()
	switch f.name {
	case "sin":
		return NFloat(math.Sin(v)), true
	case "cos":
		return NFloat(math.Cos(v)), true
	case "tan":
		return NFloat(math.Tan(v)), true
	case "exp":
		return NFloat(math.Exp(v)), true
	case "ln":
		return NFloat(math.Log(v)), true
	case "abs":
		return NFloat(math.Abs(v)), true
	case "asin":
		return NFloat(math.Asin(v)), true
	case "acos":
		return NFloat(math.Acos(v)), true
	case "atan":
		return NFloat(math.Atan(v)), true
	case "sinh":
		return NFloat(math.Sinh(v)), true
	case "cosh":
		return NFloat(math.Cosh(v)), true
	case "tanh":
		return NFloat(math.Tanh(v)), true
	case "asinh":
		return NFloat(math.Asinh(v)), true
	case "acosh":
		return NFloat(math.Acosh(v)), true
	case "atanh":
		return NFloat(math.Atanh(v)), true
	case "floor":
		return NFloat(math.Floor(v)), true
	case "ceil":
		return NFloat(math.Ceil(v)), true
	}
	return nil, false
}

func (f *Func) Equal(other Expr) bool {
	o, ok := other.(*Func)
	return ok && f.name == o.name && f.arg.Equal(o.arg)
}

func (f *Func) exprType() string { return "func" }
func (f *Func) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "func", "name": f.name, "arg": f.arg.toJSON()}
}
func (f *Func) FuncName() string { return f.name }
func (f *Func) Arg() Expr        { return f.arg }

func isNumEqual(e Expr, v int64) bool {
	n, ok := e.(*Num)
	return ok && n.Equal(N(v))
}

func peelLeadingNegative(argument Expr) (Expr, bool) {
	multiplicationNode, isMultiplication := argument.(*Mul)
	if !isMultiplication || len(multiplicationNode.factors) < 2 {
		return nil, false
	}
	leadingFactor, isNumber := multiplicationNode.factors[0].(*Num)
	if !isNumber || !leadingFactor.IsNegOne() {
		return nil, false
	}
	if len(multiplicationNode.factors) == 2 {
		return multiplicationNode.factors[1], true
	}
	return MulOf(multiplicationNode.factors[1:]...).Simplify(), true
}
