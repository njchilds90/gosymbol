package gosymbol

// ============================================================
// Integration (rule-based symbolic + numerical)
// ============================================================

func Integrate(expr Expr, varName string) (Expr, bool) {
	expr = expr.Simplify()
	switch v := expr.(type) {
	case *Num:
		return MulOf(v, S(varName)), true
	case *Sym:
		if v.name == varName {
			return MulOf(F(1, 2), PowOf(S(varName), N(2))), true
		}
		return MulOf(v, S(varName)), true
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 {
				if n.IsNegOne() {
					return LnOf(AbsOf(S(varName))), true
				}
				newExp := numAdd(n, N(1))
				return MulOf(numRecip(newExp), PowOf(S(varName), newExp)), true
			}
		}
		if sym, ok := v.exp.(*Sym); ok && sym.name == varName {
			if _, ok2 := v.base.(*Num); ok2 {
				return MulOf(PowOf(v.base, S(varName)), PowOf(LnOf(v.base), N(-1))), true
			}
		}
		return nil, false
	case *Mul:
		coeff := N(1)
		rest := []Expr{}
		for _, f := range v.factors {
			if n, ok := f.(*Num); ok {
				coeff = numMul(coeff, n)
			} else {
				rest = append(rest, f)
			}
		}
		var inner Expr
		if len(rest) == 1 {
			inner = rest[0]
		} else {
			inner = &Mul{factors: rest}
		}
		intInner, ok := Integrate(inner, varName)
		if !ok {
			return nil, false
		}
		return MulOf(coeff, intInner).Simplify(), true
	case *Add:
		terms := make([]Expr, len(v.terms))
		for i, t := range v.terms {
			intT, ok := Integrate(t, varName)
			if !ok {
				return nil, false
			}
			terms[i] = intT
		}
		return AddOf(terms...).Simplify(), true
	case *Func:
		switch v.name {
		case "sin":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return MulOf(N(-1), CosOf(S(varName))), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(N(-1), numRecip(coeff), CosOf(v.arg)), true
					}
				}
			}
		case "cos":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return SinOf(S(varName)), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(numRecip(coeff), SinOf(v.arg)), true
					}
				}
			}
		case "exp":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return ExpOf(S(varName)), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(numRecip(coeff), ExpOf(v.arg)), true
					}
				}
			}
		case "ln":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(MulOf(S(varName), LnOf(S(varName))), MulOf(N(-1), S(varName))), true
			}
		case "asin":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AsinOf(S(varName))),
					SqrtOf(AddOf(N(1), MulOf(N(-1), PowOf(S(varName), N(2))))),
				), true
			}
		case "atan":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AtanOf(S(varName))),
					MulOf(N(-1), F(1, 2), LnOf(AddOf(N(1), PowOf(S(varName), N(2))))),
				), true
			}
		}
		return nil, false
	}
	return nil, false
}

// IntegrateWithConstant returns an antiderivative with an explicit +C term.
func IntegrateWithConstant(expr Expr, varName string) (Expr, bool) {
	result, ok := Integrate(expr, varName)
	if !ok {
		return nil, false
	}
	return AddOf(result, S("C")), true
}

func DefiniteIntegrate(expr Expr, varName string, a, b float64) float64 {
	nodes := []float64{
		-0.9739065285, -0.8650633667, -0.6794095683,
		-0.4333953941, -0.1488743390, 0.1488743390,
		0.4333953941, 0.6794095683, 0.8650633667, 0.9739065285,
	}
	weights := []float64{
		0.0666713443, 0.1494513492, 0.2190863625,
		0.2692667193, 0.2955242247, 0.2955242247,
		0.2692667193, 0.2190863625, 0.1494513492, 0.0666713443,
	}
	sum := 0.0
	mid := (a + b) / 2
	half := (b - a) / 2
	for i, t := range nodes {
		xi := mid + half*t
		subbed := expr.Sub(varName, NFloat(xi))
		if v, ok := subbed.Eval(); ok {
			f, _ := v.val.Float64()
			sum += weights[i] * f
		}
	}
	return half * sum
}

// ============================================================
// Taylor / Maclaurin series
// ============================================================

func TaylorSeries(expr Expr, varName string, a Expr, order int) Expr {
	terms := []Expr{}
	current := expr
	factorial := N(1)
	for k := 0; k <= order; k++ {
		if k > 0 {
			factorial = numMul(factorial, N(int64(k)))
		}
		coeff := MulOf(current.Sub(varName, a), PowOf(factorial, N(-1))).Simplify()
		if n, ok := coeff.(*Num); ok && n.IsZero() {
			current = Diff(current, varName)
			continue
		}
		var xTerm Expr
		if k == 0 {
			xTerm = coeff
		} else if k == 1 {
			xTerm = MulOf(coeff, AddOf(S(varName), MulOf(N(-1), a)))
		} else {
			xTerm = MulOf(coeff, PowOf(AddOf(S(varName), MulOf(N(-1), a)), N(int64(k))))
		}
		terms = append(terms, xTerm)
		current = Diff(current, varName)
	}
	return AddOf(terms...).Simplify()
}

func TaylorSeriesWithRemainder(expr Expr, varName string, a Expr, order int) Expr {
	series := TaylorSeries(expr, varName, a, order)
	remainder := OTerm(varName, order+1)
	if add, ok := series.(*Add); ok {
		return &Add{terms: append(add.terms, remainder)}
	}
	return &Add{terms: []Expr{series, remainder}}
}

func MaclaurinSeries(expr Expr, varName string, order int) Expr {
	return TaylorSeries(expr, varName, N(0), order)
}

func MaclaurinSeriesWithRemainder(expr Expr, varName string, order int) Expr {
	return TaylorSeriesWithRemainder(expr, varName, N(0), order)
}
