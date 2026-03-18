package gosymbol

import (
	"math"
	"sort"
)

// ============================================================
// Canonicalization, Collect, Cancel
// ============================================================

// Canonicalize expands and fully simplifies an expression.
func Canonicalize(e Expr) Expr { return Expand(e).Simplify() }

// Collect groups terms by powers of varName.
func Collect(expr Expr, varName string) Expr {
	coeffs := PolyCoeffs(expr, varName)
	if len(coeffs) == 0 {
		return N(0)
	}
	degrees := make([]int, 0, len(coeffs))
	for d := range coeffs {
		degrees = append(degrees, d)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(degrees)))
	terms := make([]Expr, 0, len(degrees))
	for _, d := range degrees {
		c := coeffs[d]
		if cn, ok := c.(*Num); ok && cn.IsZero() {
			continue
		}
		switch d {
		case 0:
			terms = append(terms, c)
		case 1:
			terms = append(terms, MulOf(c, S(varName)))
		default:
			terms = append(terms, MulOf(c, PowOf(S(varName), N(int64(d)))))
		}
	}
	if len(terms) == 0 {
		return N(0)
	}
	return AddOf(terms...).Simplify()
}

// Cancel simplifies a rational expression num/denom.
func Cancel(num, denom Expr) Expr {
	num = num.Simplify()
	denom = denom.Simplify()
	if nn, ok := num.Eval(); ok {
		if dn, ok2 := denom.Eval(); ok2 {
			if dn.IsZero() {
				panic("gosymbol: Cancel: zero denominator")
			}
			return numDiv(nn, dn)
		}
	}
	if dn, ok := denom.(*Num); ok && dn.IsOne() {
		return num
	}
	if dn, ok := denom.(*Num); ok && dn.IsNegOne() {
		return MulOf(N(-1), num).Simplify()
	}
	numCoeff, numRest := extractCoefficient(num)
	denCoeff, denRest := extractCoefficient(denom)
	if numRest.Equal(denRest) {
		return numDiv(numCoeff, denCoeff)
	}
	return MulOf(num, PowOf(denom, N(-1))).Simplify()
}

// ============================================================
// Symbolic Factoring
// ============================================================

// FactorResult holds the result of a factoring attempt.
type FactorResult struct {
	Factors []Expr
	Success bool
}

// Factor attempts to factor a polynomial in varName.
// Handles: common GCD factor, difference of squares, perfect square trinomials,
// monic quadratics with integer roots, sum/difference of cubes.
func Factor(expr Expr, varName string) FactorResult {
	expr = Collect(expr, varName).Simplify()
	coeffs := PolyCoeffs(expr, varName)
	deg := Degree(expr, varName)

	commonFactor := N(0)
	for _, c := range coeffs {
		if cn, ok := c.(*Num); ok && cn.IsInteger() {
			if commonFactor.IsZero() {
				commonFactor = numAbs(cn)
			} else {
				a := numAbs(commonFactor).val.Num().Int64()
				b := numAbs(cn).val.Num().Int64()
				commonFactor = N(gcdInt(a, b))
			}
		} else {
			commonFactor = N(1)
			break
		}
	}
	if commonFactor.IsZero() {
		commonFactor = N(1)
	}

	scaledCoeffs := map[int]Expr{}
	if !commonFactor.IsOne() {
		for d, c := range coeffs {
			scaledCoeffs[d] = MulOf(c, PowOf(commonFactor, N(-1))).Simplify()
		}
	} else {
		scaledCoeffs = coeffs
	}

	x := S(varName)

	if deg == 2 {
		a2, hasA := scaledCoeffs[2]
		b1, hasB := scaledCoeffs[1]
		c0, hasC := scaledCoeffs[0]
		if !hasB {
			b1 = N(0)
		}
		if !hasC {
			c0 = N(0)
		}
		if !hasA {
			goto fallthrough2
		}

		// Difference of squares: x² - c (c > 0, perfect square)
		if isNumEqual(b1, 0) {
			an, aok := a2.Eval()
			cn, cok := c0.Eval()
			if aok && cok && an.IsOne() {
				cf, _ := cn.val.Float64()
				if cf < 0 {
					sqrtC := math.Sqrt(-cf)
					if math.Abs(sqrtC-math.Round(sqrtC)) < 1e-10 {
						sq := N(int64(math.Round(sqrtC)))
						result := []Expr{AddOf(x, MulOf(N(-1), sq)), AddOf(x, sq)}
						if !commonFactor.IsOne() {
							result = append([]Expr{commonFactor}, result...)
						}
						return FactorResult{Factors: result, Success: true}
					}
				}
			}
		}

		// Perfect square trinomial
		{
			an, aok := a2.Eval()
			cn, cok := c0.Eval()
			bn, bok := b1.Eval()
			if aok && cok && bok {
				af, _ := an.val.Float64()
				cf, _ := cn.val.Float64()
				bf, _ := bn.val.Float64()
				sqA := math.Sqrt(math.Abs(af))
				sqC := math.Sqrt(math.Abs(cf))
				if math.Abs(sqA-math.Round(sqA)) < 1e-10 && math.Abs(sqC-math.Round(sqC)) < 1e-10 {
					if math.Abs(2*sqA*sqC-math.Abs(bf)) < 1e-10 {
						sA, sC := int64(math.Round(sqA)), int64(math.Round(sqC))
						sign := N(1)
						if bf < 0 {
							sign = N(-1)
						}
						inner := AddOf(MulOf(N(sA), x), MulOf(sign, N(sC)))
						result := []Expr{PowOf(inner, N(2))}
						if !commonFactor.IsOne() {
							result = append([]Expr{commonFactor}, result...)
						}
						return FactorResult{Factors: result, Success: true}
					}
				}
			}
		}

		// Integer root factoring for monic quadratics
		{
			an, aok := a2.Eval()
			cn, cok := c0.Eval()
			bn, bok := b1.Eval()
			if aok && cok && bok && an.IsOne() && an.IsInteger() && cn.IsInteger() && bn.IsInteger() {
				cv := cn.val.Num().Int64()
				bv := bn.val.Num().Int64()
				absCv := cv
				if absCv < 0 {
					absCv = -absCv
				}
				found := false
				var r1, r2 int64
				for d := int64(1); d <= absCv && d <= 1000; d++ {
					if absCv%d == 0 {
						for _, candidate := range []int64{d, -d, absCv / d, -(absCv / d)} {
							if candidate*candidate+bv*candidate+cv == 0 {
								r1 = candidate
								r2 = -bv - r1
								found = true
								break
							}
						}
						if found {
							break
						}
					}
				}
				if found {
					result := []Expr{AddOf(x, N(-r1)), AddOf(x, N(-r2))}
					if !commonFactor.IsOne() {
						result = append([]Expr{commonFactor}, result...)
					}
					return FactorResult{Factors: result, Success: true}
				}
			}
		}
	}
fallthrough2:

	// Degree 3: sum/difference of cubes
	if deg == 3 {
		c3, has3 := scaledCoeffs[3]
		c0, has0 := scaledCoeffs[0]
		_, has2 := scaledCoeffs[2]
		_, has1 := scaledCoeffs[1]
		if has3 && has0 && !has2 && !has1 {
			an, aok := c3.Eval()
			cn, cok := c0.Eval()
			if aok && cok && an.IsOne() {
				cf, _ := cn.val.Float64()
				cbrtC := math.Cbrt(math.Abs(cf))
				if math.Abs(cbrtC-math.Round(cbrtC)) < 1e-10 {
					b := int64(math.Round(cbrtC))
					if cf < 0 {
						result := []Expr{
							AddOf(x, N(-b)),
							AddOf(PowOf(x, N(2)), MulOf(N(b), x), N(b*b)),
						}
						if !commonFactor.IsOne() {
							result = append([]Expr{commonFactor}, result...)
						}
						return FactorResult{Factors: result, Success: true}
					} else if cf > 0 {
						result := []Expr{
							AddOf(x, N(b)),
							AddOf(PowOf(x, N(2)), MulOf(N(-b), x), N(b*b)),
						}
						if !commonFactor.IsOne() {
							result = append([]Expr{commonFactor}, result...)
						}
						return FactorResult{Factors: result, Success: true}
					}
				}
			}
		}
	}

	if !commonFactor.IsOne() {
		return FactorResult{
			Factors: []Expr{commonFactor, MulOf(PowOf(commonFactor, N(-1)), expr).Simplify()},
			Success: true,
		}
	}
	return FactorResult{Factors: []Expr{expr}, Success: false}
}

// ============================================================
// Limits
// ============================================================

// LimitResult holds the result of a limit computation.
type LimitResult struct {
	Value   Expr
	Success bool
	Error   string
}

// Limit computes lim_{varName -> point} expr.
// Tries direct substitution, L'Hôpital (0/0), then Taylor expansion.
func Limit(expr Expr, varName string, point Expr) LimitResult {
	return limitRecursive(expr, varName, point, 5)
}

func limitRecursive(expr Expr, varName string, point Expr, maxLhopital int) LimitResult {
	expr = expr.Simplify()
	subbed := expr.Sub(varName, point).Simplify()
	if v, ok := subbed.Eval(); ok {
		f, _ := v.val.Float64()
		if !math.IsNaN(f) && !math.IsInf(f, 0) {
			return LimitResult{Value: subbed, Success: true}
		}
	}
	syms := FreeSymbols(subbed)
	if _, hasVar := syms[varName]; !hasVar {
		return LimitResult{Value: subbed, Success: true}
	}
	if maxLhopital > 0 {
		if num, denom, ok := extractQuotient(expr); ok {
			numAtPoint := num.Sub(varName, point).Simplify()
			denAtPoint := denom.Sub(varName, point).Simplify()
			nv, nok := numAtPoint.Eval()
			dv, dok := denAtPoint.Eval()
			if nok && dv != nil && nv.IsZero() && dok && dv.IsZero() {
				dNum := Diff(num, varName)
				dDen := Diff(denom, varName)
				return limitRecursive(MulOf(dNum, PowOf(dDen, N(-1))), varName, point, maxLhopital-1)
			}
		}
	}
	if pt, ok := point.Eval(); ok {
		_, ptOk := pt.val.Float64()
		if ptOk {
			series := TaylorSeries(expr, varName, point, 4)
			subSeries := series.Sub(varName, point).Simplify()
			if v, ok2 := subSeries.Eval(); ok2 {
				f, _ := v.val.Float64()
				if !math.IsNaN(f) && !math.IsInf(f, 0) {
					return LimitResult{Value: subSeries, Success: true}
				}
			}
		}
	}
	return LimitResult{
		Error:   "limit could not be determined: " + expr.String() + " as " + varName + " -> " + point.String(),
		Success: false,
	}
}

func extractQuotient(e Expr) (num, denom Expr, ok bool) {
	m, isMul := e.(*Mul)
	if !isMul {
		return nil, nil, false
	}
	var numFactors, denomFactors []Expr
	for _, f := range m.factors {
		if p, isPow := f.(*Pow); isPow {
			if en, isNum := p.exp.(*Num); isNum && en.IsNegOne() {
				denomFactors = append(denomFactors, p.base)
				continue
			}
		}
		numFactors = append(numFactors, f)
	}
	if len(denomFactors) == 0 {
		return nil, nil, false
	}
	var n, d Expr
	switch len(numFactors) {
	case 0:
		n = N(1)
	case 1:
		n = numFactors[0]
	default:
		n = &Mul{factors: numFactors}
	}
	if len(denomFactors) == 1 {
		d = denomFactors[0]
	} else {
		d = &Mul{factors: denomFactors}
	}
	return n, d, true
}

// ============================================================
// Partial Derivatives and Vector Calculus
// ============================================================

// PDiff computes the partial derivative ∂/∂varName of expr.
func PDiff(expr Expr, varName string) Expr { return Diff(expr, varName) }

// Gradient returns the gradient ∇f as a slice of partial derivatives.
func Gradient(expr Expr, varNames []string) []Expr {
	result := make([]Expr, len(varNames))
	for i, v := range varNames {
		result[i] = PDiff(expr, v)
	}
	return result
}

// Jacobian returns the m×n Jacobian matrix.
func Jacobian(exprs []Expr, varNames []string) *Matrix {
	mat := NewMatrix(len(exprs), len(varNames))
	for i, e := range exprs {
		for j, v := range varNames {
			mat.Set(i, j, PDiff(e, v))
		}
	}
	return mat
}

// Hessian returns the n×n matrix of second partial derivatives.
func Hessian(expr Expr, varNames []string) *Matrix {
	n := len(varNames)
	mat := NewMatrix(n, n)
	for i, vi := range varNames {
		for j, vj := range varNames {
			mat.Set(i, j, PDiff(PDiff(expr, vi), vj))
		}
	}
	return mat
}

// Laplacian returns ∇²f = sum of second partial derivatives.
func Laplacian(expr Expr, varNames []string) Expr {
	terms := make([]Expr, len(varNames))
	for i, v := range varNames {
		terms[i] = PDiff(PDiff(expr, v), v)
	}
	return AddOf(terms...).Simplify()
}

// Divergence returns ∇·F for a vector field.
func Divergence(exprs []Expr, varNames []string) Expr {
	if len(exprs) != len(varNames) {
		panic("gosymbol: Divergence requires len(exprs) == len(varNames)")
	}
	terms := make([]Expr, len(exprs))
	for i := range exprs {
		terms[i] = PDiff(exprs[i], varNames[i])
	}
	return AddOf(terms...).Simplify()
}

// Curl returns ∇×F for a 3D vector field.
func Curl(field [3]Expr, vars [3]string) [3]Expr {
	return [3]Expr{
		AddOf(PDiff(field[2], vars[1]), MulOf(N(-1), PDiff(field[1], vars[2]))).Simplify(),
		AddOf(PDiff(field[0], vars[2]), MulOf(N(-1), PDiff(field[2], vars[0]))).Simplify(),
		AddOf(PDiff(field[1], vars[0]), MulOf(N(-1), PDiff(field[0], vars[1]))).Simplify(),
	}
}

// ============================================================
// Top-level convenience functions (original API)
// ============================================================

func Simplify(e Expr) Expr { return e.Simplify() }
func String(e Expr) string { return e.String() }
func LaTeX(e Expr) string  { return e.LaTeX() }

func Sub(expr Expr, varName string, value Expr) Expr {
	return expr.Sub(varName, value).Simplify()
}

func Diff(expr Expr, varName string) Expr {
	return expr.Diff(varName).Simplify()
}

func Diff2(expr Expr, varName string) Expr {
	return Diff(Diff(expr, varName), varName)
}

func DiffN(expr Expr, varName string, n int) Expr {
	if n < 0 {
		panic("gosymbol: DiffN requires n >= 0")
	}
	result := expr
	for i := 0; i < n; i++ {
		result = Diff(result, varName)
	}
	return result
}

func Expand(e Expr) Expr { return expandExpr(e).Simplify() }

func expandExpr(e Expr) Expr {
	switch v := e.(type) {
	case *Mul:
		expanded := make([]Expr, len(v.factors))
		for i, f := range v.factors {
			expanded[i] = expandExpr(f)
		}
		for i, f := range expanded {
			if a, ok := f.(*Add); ok {
				rest := make([]Expr, 0, len(expanded)-1)
				for j, ef := range expanded {
					if j != i {
						rest = append(rest, ef)
					}
				}
				terms := make([]Expr, len(a.terms))
				for k, t := range a.terms {
					terms[k] = expandExpr(MulOf(append([]Expr{t}, rest...)...))
				}
				return expandExpr(AddOf(terms...))
			}
		}
		return MulOf(expanded...)
	case *Add:
		newTerms := make([]Expr, len(v.terms))
		for i, t := range v.terms {
			newTerms[i] = expandExpr(t)
		}
		return AddOf(newTerms...)
	case *Pow:
		if n, ok := v.exp.(*Num); ok && n.IsInteger() {
			exp := n.val.Num().Int64()
			if exp >= 0 && exp <= 10 {
				result := Expr(N(1))
				base := expandExpr(v.base)
				for i := int64(0); i < exp; i++ {
					result = expandExpr(&Mul{factors: []Expr{result, base}})
				}
				return result
			}
		}
		return &Pow{base: expandExpr(v.base), exp: expandExpr(v.exp)}
	}
	return e
}

// ============================================================
// Free Symbols
// ============================================================

func FreeSymbols(e Expr) map[string]struct{} {
	result := map[string]struct{}{}
	collectSymbols(e, result)
	return result
}

func collectSymbols(e Expr, out map[string]struct{}) {
	switch v := e.(type) {
	case *Sym:
		out[v.name] = struct{}{}
	case *Add:
		for _, t := range v.terms {
			collectSymbols(t, out)
		}
	case *Mul:
		for _, f := range v.factors {
			collectSymbols(f, out)
		}
	case *Pow:
		collectSymbols(v.base, out)
		collectSymbols(v.exp, out)
	case *Func:
		collectSymbols(v.arg, out)
	}
}

// ============================================================
// Polynomial utilities
// ============================================================

func Degree(expr Expr, varName string) int {
	expr = expr.Simplify()
	switch v := expr.(type) {
	case *Num:
		return 0
	case *Sym:
		if v.name == varName {
			return 1
		}
		return 0
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 && n.IsInteger() {
				return int(n.val.Num().Int64())
			}
		}
		return 0
	case *Add:
		maxDeg := 0
		for _, t := range v.terms {
			if d := Degree(t, varName); d > maxDeg {
				maxDeg = d
			}
		}
		return maxDeg
	case *Mul:
		totalDeg := 0
		for _, f := range v.factors {
			totalDeg += Degree(f, varName)
		}
		return totalDeg
	}
	return 0
}

type PolyCoeffsResult map[int]Expr

func PolyCoeffs(expr Expr, varName string) PolyCoeffsResult {
	result := PolyCoeffsResult{}
	extractCoeffs(expr.Simplify(), varName, result)
	return result
}

func extractCoeffs(e Expr, varName string, out PolyCoeffsResult) {
	switch v := e.(type) {
	case *Num:
		addCoeff(out, 0, v)
	case *Sym:
		if v.name == varName {
			addCoeff(out, 1, N(1))
		} else {
			addCoeff(out, 0, v)
		}
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 && n.IsInteger() {
				addCoeff(out, int(n.val.Num().Int64()), N(1))
				return
			}
		}
		addCoeff(out, 0, e)
	case *Mul:
		deg := 0
		coeffFactors := []Expr{}
		for _, f := range v.factors {
			if d := Degree(f, varName); d > 0 {
				deg += d
			} else {
				coeffFactors = append(coeffFactors, f)
			}
		}
		var coeff Expr
		switch len(coeffFactors) {
		case 0:
			coeff = N(1)
		case 1:
			coeff = coeffFactors[0]
		default:
			coeff = MulOf(coeffFactors...)
		}
		addCoeff(out, deg, coeff)
	case *Add:
		for _, t := range v.terms {
			extractCoeffs(t, varName, out)
		}
	}
}

func addCoeff(out PolyCoeffsResult, deg int, val Expr) {
	if existing, ok := out[deg]; ok {
		out[deg] = AddOf(existing, val).Simplify()
	} else {
		out[deg] = val.Simplify()
	}
}

type ApartResult struct {
	Terms []Expr
	Error string
}

func Apart(num, denom Expr, varName string) ApartResult {
	num = num.Simplify()
	denom = denom.Simplify()
	fr := Factor(denom, varName)
	if !fr.Success || len(fr.Factors) < 2 {
		return ApartResult{
			Terms: []Expr{MulOf(num, PowOf(denom, N(-1))).Simplify()},
			Error: "denominator could not be factored",
		}
	}
	dDen := Diff(denom, varName)
	var linearFactors, roots []Expr
	for _, factor := range fr.Factors {
		if Degree(factor, varName) != 1 {
			continue
		}
		coeffs := PolyCoeffs(factor, varName)
		a1 := N(1)
		b0 := N(0)
		if c, ok := coeffs[1]; ok {
			if cn, ok2 := c.(*Num); ok2 {
				a1 = cn
			}
		}
		if c, ok := coeffs[0]; ok {
			if cn, ok2 := c.(*Num); ok2 {
				b0 = cn
			}
		}
		if !a1.IsZero() {
			root := numDiv(numNeg(b0), a1)
			linearFactors = append(linearFactors, factor)
			roots = append(roots, root)
		}
	}
	if len(linearFactors) == 0 {
		return ApartResult{
			Terms: []Expr{MulOf(num, PowOf(denom, N(-1))).Simplify()},
			Error: "no linear factors found in denominator",
		}
	}
	terms := make([]Expr, len(roots))
	for i, root := range roots {
		residue := MulOf(Sub(num, varName, root), PowOf(Sub(dDen, varName, root), N(-1))).Simplify()
		terms[i] = MulOf(residue, PowOf(linearFactors[i], N(-1))).Simplify()
	}
	return ApartResult{Terms: terms}
}
