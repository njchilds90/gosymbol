package gosymbol

import (
	"fmt"
	"math"
	"sort"
)

// ============================================================
// Solvers
// ============================================================

type SolveResult struct {
	Solutions []Expr
	ExactForm bool
	Error     string
}

// SolveEquation solves a basic symbolic equation in one variable.
func SolveEquation(eq *Equation, varName string) SolveResult {
	if eq == nil {
		return SolveResult{Error: "nil equation"}
	}
	residual := Expand(eq.Residual()).Simplify()
	switch Degree(residual, varName) {
	case 0:
		if n, ok := residual.Eval(); ok && n.IsZero() {
			return SolveResult{Error: "identity (0 = 0): infinite solutions"}
		}
		return SolveResult{Error: "no solution (inconsistent)"}
	case 1:
		coeffs := PolyCoeffs(residual, varName)
		return SolveLinear(coeffs[1], coeffs[0])
	case 2:
		coeffs := PolyCoeffs(residual, varName)
		return SolveQuadraticExact(coeffs[2], coeffs[1], coeffs[0])
	default:
		return SolveResult{Error: "symbolic equation solving currently supports polynomial degree <= 2"}
	}
}

func SolveLinear(a, b Expr) SolveResult {
	an, aok := a.Eval()
	bn, bok := b.Eval()
	if aok && bok {
		if an.IsZero() {
			if bn.IsZero() {
				return SolveResult{Error: "identity (0 = 0): infinite solutions"}
			}
			return SolveResult{Error: "no solution (inconsistent)"}
		}
		return SolveResult{Solutions: []Expr{numMul(numNeg(bn), numRecip(an))}, ExactForm: true}
	}
	return SolveResult{Solutions: []Expr{MulOf(N(-1), b, PowOf(a, N(-1))).Simplify()}, ExactForm: false}
}

func SolveQuadratic(a, b, c Expr) SolveResult {
	an, aok := a.Eval()
	bn, bok := b.Eval()
	cn, cok := c.Eval()
	if !aok || !bok || !cok {
		return SolveResult{Error: "SolveQuadratic requires numeric coefficients"}
	}
	af, _ := an.val.Float64()
	bf, _ := bn.val.Float64()
	cf, _ := cn.val.Float64()
	if af == 0 {
		return SolveLinear(b, c)
	}
	disc := bf*bf - 4*af*cf
	if disc < 0 {
		return SolveResult{Error: fmt.Sprintf("complex roots: %g ± %gi", -bf/(2*af), math.Sqrt(-disc)/(2*af))}
	}
	sq := math.Sqrt(disc)
	return SolveResult{Solutions: []Expr{NFloat((-bf + sq) / (2 * af)), NFloat((-bf - sq) / (2 * af))}, ExactForm: false}
}

func SolveQuadraticExact(a, b, c Expr) SolveResult {
	an, aok := a.Eval()
	bn, bok := b.Eval()
	cn, cok := c.Eval()
	if !aok || !bok || !cok {
		disc := AddOf(PowOf(b, N(2)), MulOf(N(-4), a, c))
		denom := MulOf(N(2), a)
		x1 := MulOf(AddOf(MulOf(N(-1), b), SqrtOf(disc)), PowOf(denom, N(-1)))
		x2 := MulOf(AddOf(MulOf(N(-1), b), MulOf(N(-1), SqrtOf(disc))), PowOf(denom, N(-1)))
		return SolveResult{Solutions: []Expr{x1.Simplify(), x2.Simplify()}, ExactForm: true}
	}
	af, _ := an.val.Float64()
	bf, _ := bn.val.Float64()
	cf, _ := cn.val.Float64()
	if af == 0 {
		return SolveLinear(b, c)
	}
	disc := bf*bf - 4*af*cf
	if disc < 0 {
		return SolveResult{Error: fmt.Sprintf("complex roots: %g ± %gi", -bf/(2*af), math.Sqrt(-disc)/(2*af))}
	}
	sq := math.Sqrt(disc)
	sqInt := int64(math.Round(sq))
	twoA := numMul(N(2), an)
	if float64(sqInt)*float64(sqInt) == disc {
		x1 := numDiv(numAdd(numNeg(bn), N(sqInt)), twoA)
		x2 := numDiv(numSub(numNeg(bn), N(sqInt)), twoA)
		return SolveResult{Solutions: []Expr{x1, x2}, ExactForm: true}
	}
	return SolveResult{Solutions: []Expr{NFloat((-bf + sq) / (2 * af)), NFloat((-bf - sq) / (2 * af))}, ExactForm: false}
}

func SolveCubic(a, b, c, d Expr) SolveResult {
	an, aok := a.Eval()
	bn, bok := b.Eval()
	cn, cok := c.Eval()
	dn, dok := d.Eval()
	if !aok || !bok || !cok || !dok {
		return SolveResult{Error: "SolveCubic requires numeric coefficients"}
	}
	af, _ := an.val.Float64()
	bf, _ := bn.val.Float64()
	cf, _ := cn.val.Float64()
	df, _ := dn.val.Float64()
	if af == 0 {
		return SolveQuadratic(b, c, d)
	}
	p := (3*af*cf - bf*bf) / (3 * af * af)
	q := (2*bf*bf*bf - 9*af*bf*cf + 27*af*af*df) / (27 * af * af * af)
	offset := bf / (3 * af)
	disc := -(4*p*p*p + 27*q*q)

	var roots []Expr
	if disc > 0 {
		m := 2 * math.Sqrt(-p/3)
		theta := math.Acos(3*q/(p*m)) / 3
		for k := 0; k < 3; k++ {
			roots = append(roots, NFloat(m*math.Cos(theta-2*math.Pi*float64(k)/3)-offset))
		}
	} else if disc == 0 {
		if q == 0 {
			roots = []Expr{NFloat(-offset), NFloat(-offset), NFloat(-offset)}
		} else {
			roots = []Expr{NFloat(3*q/p - offset), NFloat(-3*q/(2*p) - offset)}
		}
	} else {
		A := math.Cbrt(-q/2 + math.Sqrt(q*q/4+p*p*p/27))
		B := float64(0)
		if A != 0 {
			B = -p / (3 * A)
		}
		realRoot := A + B - offset
		realImag := math.Sqrt(3) / 2 * math.Abs(A-B)
		return SolveResult{
			Solutions: []Expr{NFloat(realRoot)},
			Error:     fmt.Sprintf("1 real root (%.6g); complex pair: real=%.6g, imag=±%.6g", realRoot, -A/2-B/2-offset, realImag),
		}
	}
	return SolveResult{Solutions: roots, ExactForm: false}
}

func SolvePolynomialNewton(expr Expr, varName string, searchRange, tol float64, maxIter int) SolveResult {
	if searchRange <= 0 {
		searchRange = 100
	}
	if tol <= 0 {
		tol = 1e-10
	}
	if maxIter <= 0 {
		maxIter = 100
	}
	f := func(x float64) float64 {
		v := expr.Sub(varName, NFloat(x)).Simplify()
		if n, ok := v.Eval(); ok {
			f64, _ := n.val.Float64()
			return f64
		}
		return math.NaN()
	}
	df := func(x float64) float64 {
		v := Diff(expr, varName).Sub(varName, NFloat(x)).Simplify()
		if n, ok := v.Eval(); ok {
			f64, _ := n.val.Float64()
			return f64
		}
		return math.NaN()
	}
	var roots []float64
	for i := 0; i <= 200; i++ {
		x := -searchRange + 2*searchRange*float64(i)/200
		for iter := 0; iter < maxIter; iter++ {
			fx := f(x)
			if math.IsNaN(fx) {
				break
			}
			if math.Abs(fx) < tol {
				dup := false
				for _, r := range roots {
					if math.Abs(r-x) < tol*100 {
						dup = true
						break
					}
				}
				if !dup {
					roots = append(roots, x)
				}
				break
			}
			dfx := df(x)
			if math.IsNaN(dfx) || math.Abs(dfx) < 1e-15 {
				break
			}
			x -= fx / dfx
			if math.Abs(x) > searchRange*10 {
				break
			}
		}
	}
	sort.Float64s(roots)
	solutions := make([]Expr, len(roots))
	for i, r := range roots {
		solutions[i] = NFloat(r)
	}
	return SolveResult{Solutions: solutions, ExactForm: false}
}

func SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2 Expr) (xSol, ySol Expr, err error) {
	det := AddOf(MulOf(a1, b2), MulOf(N(-1), MulOf(a2, b1))).Simplify()
	dn, ok := det.Eval()
	if !ok || dn.IsZero() {
		return nil, nil, fmt.Errorf("system is singular or has no unique solution")
	}
	dx := AddOf(MulOf(c1, b2), MulOf(N(-1), MulOf(c2, b1))).Simplify()
	dy := AddOf(MulOf(a1, c2), MulOf(N(-1), MulOf(a2, c1))).Simplify()
	return MulOf(dx, PowOf(det, N(-1))).Simplify(), MulOf(dy, PowOf(det, N(-1))).Simplify(), nil
}
