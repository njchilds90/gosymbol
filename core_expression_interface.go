// Package gosymbol provides a deterministic symbolic math kernel for Go.
//
// Design goals:
//   - Zero external dependencies
//   - Exact rational arithmetic (math/big.Rat)
//   - Deterministic simplification and stable output
//   - AI/LLM friendly: JSON, LaTeX, infix parsing, and MCP-ready APIs
//   - Embeddable in Go services, CLI tools, and agent backends
package gosymbol

import "fmt"

// Expr is the common interface implemented by every symbolic expression node.
type Expr interface {
	Simplify() Expr
	String() string
	LaTeX() string
	Sub(varName string, value Expr) Expr
	Diff(varName string) Expr
	Eval() (*Num, bool)
	Equal(other Expr) bool
	exprType() string
	toJSON() map[string]interface{}
}

// ============================================================
// Equation
// ============================================================

type Equation struct{ LHS, RHS Expr }

func Eq(lhs, rhs Expr) *Equation { return &Equation{LHS: lhs, RHS: rhs} }
func (e *Equation) String() string {
	return e.LHS.String() + " = " + e.RHS.String()
}
func (e *Equation) LaTeX() string { return e.LHS.LaTeX() + " = " + e.RHS.LaTeX() }
func (e *Equation) Residual() Expr {
	return AddOf(e.LHS, MulOf(N(-1), e.RHS)).Simplify()
}

// ============================================================
// BigO — remainder term for series
// ============================================================

type BigO struct {
	varName string
	order   int
}

func OTerm(varName string, order int) *BigO { return &BigO{varName: varName, order: order} }

func (o *BigO) Simplify() Expr        { return o }
func (o *BigO) String() string        { return fmt.Sprintf("O(%s^%d)", o.varName, o.order) }
func (o *BigO) LaTeX() string         { return fmt.Sprintf("\\mathcal{O}(%s^{%d})", o.varName, o.order) }
func (o *BigO) Sub(string, Expr) Expr { return o }
func (o *BigO) Diff(string) Expr      { return N(0) }
func (o *BigO) Eval() (*Num, bool)    { return nil, false }
func (o *BigO) Equal(other Expr) bool {
	ob, ok := other.(*BigO)
	return ok && ob.varName == o.varName && ob.order == o.order
}
func (o *BigO) exprType() string { return "bigo" }
func (o *BigO) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "bigo", "var": o.varName, "order": o.order}
}
func (o *BigO) Order() int { return o.order }

// ============================================================
// Deep Simplification and Trig Identities
// ============================================================

// TrigSimplify applies trig identities: sin²+cos²=1, exp(ln(x))=x, ln(exp(x))=x.
func TrigSimplify(e Expr) Expr {
	return trigSimplifyExpr(e.Simplify()).Simplify()
}

func trigSimplifyExpr(e Expr) Expr {
	switch v := e.(type) {
	case *Add:
		newTerms := make([]Expr, len(v.terms))
		for i, t := range v.terms {
			newTerms[i] = trigSimplifyExpr(t)
		}
		return trigFindPythagorean(AddOf(newTerms...))
	case *Mul:
		newFactors := make([]Expr, len(v.factors))
		for i, f := range v.factors {
			newFactors[i] = trigSimplifyExpr(f)
		}
		if len(newFactors) == 3 {
			if leadingCoefficient, ok := newFactors[0].(*Num); ok && leadingCoefficient.Equal(N(2)) {
				if sineFunction, ok := newFactors[1].(*Func); ok && sineFunction.name == "sin" {
					if cosineFunction, ok := newFactors[2].(*Func); ok && cosineFunction.name == "cos" && sineFunction.arg.Equal(cosineFunction.arg) {
						return SinOf(MulOf(N(2), sineFunction.arg))
					}
				}
				if cosineFunction, ok := newFactors[1].(*Func); ok && cosineFunction.name == "cos" {
					if sineFunction, ok := newFactors[2].(*Func); ok && sineFunction.name == "sin" && sineFunction.arg.Equal(cosineFunction.arg) {
						return SinOf(MulOf(N(2), sineFunction.arg))
					}
				}
				if hyperbolicSineFunction, ok := newFactors[1].(*Func); ok && hyperbolicSineFunction.name == "sinh" {
					if hyperbolicCosineFunction, ok := newFactors[2].(*Func); ok && hyperbolicCosineFunction.name == "cosh" && hyperbolicSineFunction.arg.Equal(hyperbolicCosineFunction.arg) {
						return SinhOf(MulOf(N(2), hyperbolicSineFunction.arg))
					}
				}
				if hyperbolicCosineFunction, ok := newFactors[1].(*Func); ok && hyperbolicCosineFunction.name == "cosh" {
					if hyperbolicSineFunction, ok := newFactors[2].(*Func); ok && hyperbolicSineFunction.name == "sinh" && hyperbolicSineFunction.arg.Equal(hyperbolicCosineFunction.arg) {
						return SinhOf(MulOf(N(2), hyperbolicSineFunction.arg))
					}
				}
			}
		}
		return MulOf(newFactors...)
	case *Pow:
		return PowOf(trigSimplifyExpr(v.base), v.exp)
	case *Func:
		return funcOf(v.name, trigSimplifyExpr(v.arg)).Simplify()
	}
	return e
}

func trigFindPythagorean(e Expr) Expr {
	add, ok := e.(*Add)
	if !ok {
		return e
	}
	type trigTerm struct {
		funcName string
		argStr   string
		argExpr  Expr
		coeff    *Num
		idx      int
	}
	var trigTerms []trigTerm
	for idx, t := range add.terms {
		coeff, inner := extractCoefficient(t)
		if p, ok2 := inner.(*Pow); ok2 {
			if fn, ok3 := p.base.(*Func); ok3 {
				if en, ok4 := p.exp.(*Num); ok4 && en.IsInteger() && en.val.Num().Int64() == 2 {
					if fn.name == "sin" || fn.name == "cos" || fn.name == "sinh" || fn.name == "cosh" {
						trigTerms = append(trigTerms, trigTerm{funcName: fn.name, argStr: fn.arg.String(), argExpr: fn.arg, coeff: coeff, idx: idx})
					}
				}
			}
		}
	}
	for i := 0; i < len(trigTerms); i++ {
		for j := i + 1; j < len(trigTerms); j++ {
			ti, tj := trigTerms[i], trigTerms[j]
			if ti.argStr == tj.argStr && ti.funcName != tj.funcName && numCmp(ti.coeff, tj.coeff) == 0 {
				if (ti.funcName == "sinh" && tj.funcName == "cosh") || (ti.funcName == "cosh" && tj.funcName == "sinh") {
					continue
				}
				newTerms := []Expr{}
				for idx, t := range add.terms {
					if idx != ti.idx && idx != tj.idx {
						newTerms = append(newTerms, t)
					}
				}
				newTerms = append(newTerms, ti.coeff)
				return AddOf(newTerms...).Simplify()
			}
			if ti.argStr == tj.argStr && numCmp(ti.coeff, numNeg(tj.coeff)) == 0 {
				if (ti.funcName == "cos" && tj.funcName == "sin") || (ti.funcName == "sin" && tj.funcName == "cos") {
					newTerms := []Expr{}
					for idx, t := range add.terms {
						if idx != ti.idx && idx != tj.idx {
							newTerms = append(newTerms, t)
						}
					}
					angle := ti.argExpr
					newTerms = append(newTerms, MulOf(ti.coeff, CosOf(MulOf(N(2), angle))).Simplify())
					return AddOf(newTerms...).Simplify()
				}
			}
			if ti.argStr == tj.argStr && numCmp(ti.coeff, tj.coeff) == 0 {
				if (ti.funcName == "cosh" && tj.funcName == "sinh") || (ti.funcName == "sinh" && tj.funcName == "cosh") {
					newTerms := []Expr{}
					for idx, t := range add.terms {
						if idx != ti.idx && idx != tj.idx {
							newTerms = append(newTerms, t)
						}
					}
					if ti.funcName == "cosh" {
						newTerms = append(newTerms, ti.coeff)
					} else {
						newTerms = append(newTerms, numNeg(ti.coeff))
					}
					return AddOf(newTerms...).Simplify()
				}
			}
		}
	}
	return e
}

func extractCoefficient(e Expr) (*Num, Expr) {
	if m, ok := e.(*Mul); ok && len(m.factors) >= 2 {
		if coeff, ok2 := m.factors[0].(*Num); ok2 {
			rest := m.factors[1:]
			if len(rest) == 1 {
				return coeff, rest[0]
			}
			return coeff, &Mul{factors: rest}
		}
	}
	return N(1), e
}

// DeepSimplify applies repeated simplification+trig passes until stable.
func DeepSimplify(e Expr) Expr {
	prev := ""
	curr := e.Simplify()
	for i := 0; i < 10; i++ {
		str := curr.String()
		if str == prev {
			break
		}
		prev = str
		curr = TrigSimplify(curr).Simplify()
	}
	return curr
}
