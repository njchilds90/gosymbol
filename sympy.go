// Package gosympy provides a minimal, deterministic symbolic math kernel for Go.
//
// Design goals:
//   - Single file, zero external dependencies
//   - Exact rational arithmetic (math/big.Rat)
//   - Deterministic simplification and stable output
//   - AI/LLM friendly: JSON, LaTeX, and MCP-ready APIs
//   - Embeddable in Go services, CLI tools, and agent backends
//
// Quick start:
//
//	x := gosympy.S("x")
//	expr := gosympy.AddOf(gosympy.MulOf(gosympy.N(2), x), gosympy.N(3))
//	fmt.Println(gosympy.String(expr))   // 2*x + 3
//	fmt.Println(gosympy.LaTeX(expr))    // 2 x + 3
//	d := gosympy.Diff(expr, "x")        // 2
//	fmt.Println(gosympy.String(d))
package gosympy

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
)

// ============================================================
// Core Interface
// ============================================================

// Expr is the central interface for all symbolic expressions.
// Every expression can simplify itself, render to string or LaTeX,
// substitute a variable, differentiate, and serialize to JSON.
type Expr interface {
	// Simplify returns a simplified form of the expression.
	Simplify() Expr
	// String returns a human-readable infix representation.
	String() string
	// LaTeX returns a LaTeX representation.
	LaTeX() string
	// Sub substitutes varName with value throughout the expression.
	Sub(varName string, value Expr) Expr
	// Diff returns the derivative with respect to varName.
	Diff(varName string) Expr
	// Eval attempts to reduce the expression to a *Num if all symbols are resolved.
	Eval() (*Num, bool)
	// Equal returns true if two expressions are structurally identical after simplification.
	Equal(other Expr) bool
	// exprType returns the node type tag used for JSON serialization.
	exprType() string
	// toJSON returns the JSON-serializable representation.
	toJSON() map[string]interface{}
}

// ============================================================
// Num — exact rational number
// ============================================================

// Num represents an exact rational number backed by math/big.Rat.
type Num struct {
	val *big.Rat
}

// N creates an integer Num.
func N(n int64) *Num { return &Num{val: new(big.Rat).SetInt64(n)} }

// F creates a rational Num p/q.
func F(p, q int64) *Num {
	if q == 0 {
		panic("gosympy: denominator is zero")
	}
	return &Num{val: new(big.Rat).SetFrac(big.NewInt(p), big.NewInt(q))}
}

// NFloat creates a Num from a float64 approximation (use sparingly).
func NFloat(f float64) *Num {
	r := new(big.Rat).SetFloat64(f)
	return &Num{val: r}
}

func (n *Num) Simplify() Expr            { return n }
func (n *Num) Sub(string, Expr) Expr     { return n }
func (n *Num) Diff(string) Expr          { return N(0) }
func (n *Num) Eval() (*Num, bool)        { return n, true }
func (n *Num) Equal(other Expr) bool     { o, ok := other.(*Num); return ok && n.val.Cmp(o.val) == 0 }
func (n *Num) exprType() string          { return "num" }
func (n *Num) Float64() float64          { f, _ := n.val.Float64(); return f }
func (n *Num) IsZero() bool              { return n.val.Sign() == 0 }
func (n *Num) IsOne() bool               { return n.val.Cmp(new(big.Rat).SetInt64(1)) == 0 }
func (n *Num) IsNegOne() bool            { return n.val.Cmp(new(big.Rat).SetInt64(-1)) == 0 }
func (n *Num) IsInteger() bool           { return n.val.IsInt() }
func (n *Num) Rat() *big.Rat             { return new(big.Rat).Set(n.val) }

func (n *Num) String() string {
	if n.val.IsInt() {
		return n.val.Num().String()
	}
	return n.val.RatString()
}

func (n *Num) LaTeX() string {
	if n.val.IsInt() {
		return n.val.Num().String()
	}
	sign := ""
	v := new(big.Rat).Set(n.val)
	if v.Sign() < 0 {
		sign = "-"
		v.Neg(v)
	}
	return fmt.Sprintf("%s\\frac{%s}{%s}", sign, v.Num().String(), v.Denom().String())
}

func (n *Num) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "num", "value": n.String()}
}

func numAdd(a, b *Num) *Num {
	return &Num{val: new(big.Rat).Add(a.val, b.val)}
}
func numMul(a, b *Num) *Num {
	return &Num{val: new(big.Rat).Mul(a.val, b.val)}
}
func numNeg(a *Num) *Num {
	return &Num{val: new(big.Rat).Neg(a.val)}
}
func numRecip(a *Num) *Num {
	if a.IsZero() {
		panic("gosympy: division by zero")
	}
	return &Num{val: new(big.Rat).Inv(a.val)}
}

// ============================================================
// Sym — symbolic variable
// ============================================================

// Sym represents a named symbolic variable.
type Sym struct{ name string }

// S creates a symbolic variable with the given name.
func S(name string) *Sym { return &Sym{name: name} }

func (s *Sym) Simplify() Expr        { return s }
func (s *Sym) String() string        { return s.name }
func (s *Sym) LaTeX() string         { return s.name }
func (s *Sym) Eval() (*Num, bool)    { return nil, false }
func (s *Sym) Equal(other Expr) bool { o, ok := other.(*Sym); return ok && s.name == o.name }
func (s *Sym) exprType() string      { return "sym" }
func (s *Sym) Name() string          { return s.name }

func (s *Sym) Sub(varName string, value Expr) Expr {
	if s.name == varName {
		return value
	}
	return s
}

func (s *Sym) Diff(varName string) Expr {
	if s.name == varName {
		return N(1)
	}
	return N(0)
}

func (s *Sym) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "sym", "name": s.name}
}

// ============================================================
// Add — sum of terms
// ============================================================

// Add represents a sum of zero or more terms.
type Add struct{ terms []Expr }

// AddOf creates an Add node from the given terms, then simplifies.
func AddOf(terms ...Expr) Expr {
	return (&Add{terms: terms}).Simplify()
}

func (a *Add) Simplify() Expr {
	// 1. Recursively simplify children.
	flat := make([]Expr, 0, len(a.terms))
	for _, t := range a.terms {
		s := t.Simplify()
		if inner, ok := s.(*Add); ok {
			flat = append(flat, inner.terms...)
		} else {
			flat = append(flat, s)
		}
	}

	// 2. Collect numeric constant and group like terms.
	numAccum := N(0)
	symCoeffs := map[string]*Num{} // symbol name -> coefficient
	symOrder := []string{}
	others := []Expr{}

	for _, t := range flat {
		switch v := t.(type) {
		case *Num:
			numAccum = numAdd(numAccum, v)
		case *Sym:
			if _, seen := symCoeffs[v.name]; !seen {
				symOrder = append(symOrder, v.name)
				symCoeffs[v.name] = N(0)
			}
			symCoeffs[v.name] = numAdd(symCoeffs[v.name], N(1))
		default:
			others = append(others, t)
		}
	}

	result := []Expr{}
	sort.Strings(symOrder)
	for _, name := range symOrder {
		coeff := symCoeffs[name]
		if coeff.IsZero() {
			continue
		}
		if coeff.IsOne() {
			result = append(result, S(name))
		} else {
			result = append(result, MulOf(coeff, S(name)))
		}
	}
	result = append(result, others...)
	if !numAccum.IsZero() {
		result = append(result, numAccum)
	}

	if len(result) == 0 {
		return N(0)
	}
	if len(result) == 1 {
		return result[0]
	}
	return &Add{terms: result}
}

func (a *Add) String() string {
	if len(a.terms) == 0 {
		return "0"
	}
	parts := make([]string, len(a.terms))
	for i, t := range a.terms {
		parts[i] = t.String()
	}
	return strings.Join(parts, " + ")
}

func (a *Add) LaTeX() string {
	parts := make([]string, len(a.terms))
	for i, t := range a.terms {
		parts[i] = t.LaTeX()
	}
	return strings.Join(parts, " + ")
}

func (a *Add) Sub(varName string, value Expr) Expr {
	newTerms := make([]Expr, len(a.terms))
	for i, t := range a.terms {
		newTerms[i] = t.Sub(varName, value)
	}
	return AddOf(newTerms...)
}

func (a *Add) Diff(varName string) Expr {
	dTerms := make([]Expr, len(a.terms))
	for i, t := range a.terms {
		dTerms[i] = t.Diff(varName)
	}
	return AddOf(dTerms...)
}

func (a *Add) Eval() (*Num, bool) {
	acc := N(0)
	for _, t := range a.terms {
		v, ok := t.Eval()
		if !ok {
			return nil, false
		}
		acc = numAdd(acc, v)
	}
	return acc, true
}

func (a *Add) Equal(other Expr) bool {
	o, ok := other.(*Add)
	if !ok || len(a.terms) != len(o.terms) {
		return false
	}
	for i := range a.terms {
		if !a.terms[i].Equal(o.terms[i]) {
			return false
		}
	}
	return true
}

func (a *Add) exprType() string { return "add" }
func (a *Add) toJSON() map[string]interface{} {
	ts := make([]map[string]interface{}, len(a.terms))
	for i, t := range a.terms {
		ts[i] = t.toJSON()
	}
	return map[string]interface{}{"type": "add", "terms": ts}
}

// ============================================================
// Mul — product of factors
// ============================================================

// Mul represents a product of zero or more factors.
type Mul struct{ factors []Expr }

// MulOf creates a Mul node from the given factors, then simplifies.
func MulOf(factors ...Expr) Expr {
	return (&Mul{factors: factors}).Simplify()
}

func (m *Mul) Simplify() Expr {
	flat := make([]Expr, 0, len(m.factors))
	for _, f := range m.factors {
		s := f.Simplify()
		if inner, ok := s.(*Mul); ok {
			flat = append(flat, inner.factors...)
		} else {
			flat = append(flat, s)
		}
	}

	// Collect numeric coefficient.
	coeff := N(1)
	others := []Expr{}
	for _, f := range flat {
		if v, ok := f.(*Num); ok {
			coeff = numMul(coeff, v)
		} else {
			others = append(others, f)
		}
	}

	if coeff.IsZero() {
		return N(0)
	}

	if len(others) == 0 {
		return coeff
	}

	// Sort non-numeric factors for determinism.
	sort.Slice(others, func(i, j int) bool {
		return others[i].String() < others[j].String()
	})

	if coeff.IsOne() {
		if len(others) == 1 {
			return others[0]
		}
		return &Mul{factors: others}
	}
	if coeff.IsNegOne() && len(others) == 1 {
		return &Mul{factors: append([]Expr{N(-1)}, others...)}
	}
	return &Mul{factors: append([]Expr{coeff}, others...)}
}

func (m *Mul) String() string {
	if len(m.factors) == 0 {
		return "1"
	}
	parts := make([]string, len(m.factors))
	for i, f := range m.factors {
		_, isAdd := f.(*Add)
		if isAdd {
			parts[i] = "(" + f.String() + ")"
		} else {
			parts[i] = f.String()
		}
	}
	return strings.Join(parts, "*")
}

func (m *Mul) LaTeX() string {
	parts := make([]string, len(m.factors))
	for i, f := range m.factors {
		_, isAdd := f.(*Add)
		if isAdd {
			parts[i] = "\\left(" + f.LaTeX() + "\\right)"
		} else {
			parts[i] = f.LaTeX()
		}
	}
	return strings.Join(parts, " ")
}

func (m *Mul) Sub(varName string, value Expr) Expr {
	newFactors := make([]Expr, len(m.factors))
	for i, f := range m.factors {
		newFactors[i] = f.Sub(varName, value)
	}
	return MulOf(newFactors...)
}

// Product rule: d/dx(u*v*w...) = sum over i of (d/dx f_i) * prod(others)
func (m *Mul) Diff(varName string) Expr {
	terms := make([]Expr, len(m.factors))
	for i, fi := range m.factors {
		dfi := fi.Diff(varName)
		others := make([]Expr, 0, len(m.factors)-1)
		for j, fj := range m.factors {
			if j != i {
				others = append(others, fj)
			}
		}
		if len(others) == 0 {
			terms[i] = dfi
		} else {
			terms[i] = MulOf(append([]Expr{dfi}, others...)...)
		}
	}
	return AddOf(terms...)
}

func (m *Mul) Eval() (*Num, bool) {
	acc := N(1)
	for _, f := range m.factors {
		v, ok := f.Eval()
		if !ok {
			return nil, false
		}
		acc = numMul(acc, v)
	}
	return acc, true
}

func (m *Mul) Equal(other Expr) bool {
	o, ok := other.(*Mul)
	if !ok || len(m.factors) != len(o.factors) {
		return false
	}
	for i := range m.factors {
		if !m.factors[i].Equal(o.factors[i]) {
			return false
		}
	}
	return true
}

func (m *Mul) exprType() string { return "mul" }
func (m *Mul) toJSON() map[string]interface{} {
	fs := make([]map[string]interface{}, len(m.factors))
	for i, f := range m.factors {
		fs[i] = f.toJSON()
	}
	return map[string]interface{}{"type": "mul", "factors": fs}
}

// ============================================================
// Pow — base^exponent
// ============================================================

// Pow represents base raised to an exponent.
type Pow struct {
	base, exp Expr
}

// PowOf creates a Pow node, then simplifies.
func PowOf(base, exp Expr) Expr {
	return (&Pow{base: base, exp: exp}).Simplify()
}

func (p *Pow) Simplify() Expr {
	base := p.base.Simplify()
	exp := p.exp.Simplify()

	// x^0 = 1
	if en, ok := exp.(*Num); ok && en.IsZero() {
		return N(1)
	}
	// x^1 = x
	if en, ok := exp.(*Num); ok && en.IsOne() {
		return base
	}
	// 0^n = 0 (n > 0)
	if bn, ok := base.(*Num); ok && bn.IsZero() {
		return N(0)
	}
	// 1^n = 1
	if bn, ok := base.(*Num); ok && bn.IsOne() {
		return N(1)
	}
	// num^int => exact rational
	if bn, ok := base.(*Num); ok {
		if en, ok2 := exp.(*Num); ok2 && en.IsInteger() {
			e := en.val.Num().Int64()
			if e >= 0 && e <= 20 {
				result := N(1)
				for i := int64(0); i < e; i++ {
					result = numMul(result, bn)
				}
				return result
			}
		}
	}
	return &Pow{base: base, exp: exp}
}

func (p *Pow) String() string {
	baseStr := p.base.String()
	expStr := p.exp.String()
	_, baseIsAdd := p.base.(*Add)
	_, baseIsMul := p.base.(*Mul)
	if baseIsAdd || baseIsMul {
		baseStr = "(" + baseStr + ")"
	}
	return baseStr + "^" + expStr
}

func (p *Pow) LaTeX() string {
	baseStr := p.base.LaTeX()
	expStr := p.exp.LaTeX()
	_, baseIsAdd := p.base.(*Add)
	_, baseIsMul := p.base.(*Mul)
	if baseIsAdd || baseIsMul {
		baseStr = "\\left(" + baseStr + "\\right)"
	}
	return baseStr + "^{" + expStr + "}"
}

func (p *Pow) Sub(varName string, value Expr) Expr {
	return PowOf(p.base.Sub(varName, value), p.exp.Sub(varName, value))
}

// Chain rule + power rule: d/dx(u^v) = v*u^(v-1)*du/dx  (when v is constant)
// General form: d/dx(u^v) = u^v * (v'*ln(u) + v*u'/u) — approximated symbolically.
func (p *Pow) Diff(varName string) Expr {
	du := p.base.Diff(varName)
	dv := p.exp.Diff(varName)

	_, expIsNum := p.exp.(*Num)
	if expIsNum {
		// Power rule: v * u^(v-1) * du/dx
		newExp := AddOf(p.exp, N(-1))
		return MulOf(p.exp, PowOf(p.base, newExp), du)
	}
	_, baseIsNum := p.base.(*Num)
	if baseIsNum {
		// a^u: d/dx = a^u * ln(a) * du/dx
		return MulOf(PowOf(p.base, p.exp), LnOf(p.base), dv)
	}
	// General: u^v * (dv*ln(u) + v*du/u)
	logTerm := MulOf(dv, LnOf(p.base))
	divTerm := MulOf(p.exp, du, PowOf(p.base, N(-1)))
	return MulOf(PowOf(p.base, p.exp), AddOf(logTerm, divTerm))
}

func (p *Pow) Eval() (*Num, bool) {
	b, ok1 := p.base.Eval()
	e, ok2 := p.exp.Eval()
	if ok1 && ok2 {
		bf, _ := b.val.Float64()
		ef, _ := e.val.Float64()
		r := math.Pow(bf, ef)
		return NFloat(r), true
	}
	return nil, false
}

func (p *Pow) Equal(other Expr) bool {
	o, ok := other.(*Pow)
	return ok && p.base.Equal(o.base) && p.exp.Equal(o.exp)
}

func (p *Pow) exprType() string { return "pow" }
func (p *Pow) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "pow", "base": p.base.toJSON(), "exp": p.exp.toJSON()}
}

// ============================================================
// Func — named function applications (sin, cos, exp, ln, etc.)
// ============================================================

// Func represents a named unary function applied to an argument.
type Func struct {
	name string
	arg  Expr
}

func funcOf(name string, arg Expr) *Func {
	return &Func{name: name, arg: arg}
}

// SinOf returns sin(arg).
func SinOf(arg Expr) Expr { return funcOf("sin", arg).Simplify() }

// CosOf returns cos(arg).
func CosOf(arg Expr) Expr { return funcOf("cos", arg).Simplify() }

// TanOf returns tan(arg).
func TanOf(arg Expr) Expr { return funcOf("tan", arg).Simplify() }

// ExpOf returns e^arg.
func ExpOf(arg Expr) Expr { return funcOf("exp", arg).Simplify() }

// LnOf returns ln(arg).
func LnOf(arg Expr) Expr { return funcOf("ln", arg).Simplify() }

// SqrtOf returns sqrt(arg) = arg^(1/2).
func SqrtOf(arg Expr) Expr { return PowOf(arg, F(1, 2)) }

// AbsOf returns |arg|.
func AbsOf(arg Expr) Expr { return funcOf("abs", arg).Simplify() }

func (f *Func) Simplify() Expr {
	arg := f.arg.Simplify()
	// Evaluate numerically if possible.
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
		}
	}
	// ln(1) = 0
	if f.name == "ln" {
		if n, ok := arg.(*Num); ok && n.IsOne() {
			return N(0)
		}
	}
	// exp(0) = 1
	if f.name == "exp" {
		if n, ok := arg.(*Num); ok && n.IsZero() {
			return N(1)
		}
	}
	return &Func{name: f.name, arg: arg}
}

func (f *Func) String() string { return f.name + "(" + f.arg.String() + ")" }
func (f *Func) LaTeX() string {
	switch f.name {
	case "sin", "cos", "tan", "exp", "ln", "abs":
		if f.name == "abs" {
			return "\\left|" + f.arg.LaTeX() + "\\right|"
		}
		return "\\" + f.name + "\\left(" + f.arg.LaTeX() + "\\right)"
	}
	return "\\operatorname{" + f.name + "}\\left(" + f.arg.LaTeX() + "\\right)"
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
		outer = AddOf(N(1), PowOf(CosOf(f.arg), N(-2)))
	case "exp":
		outer = ExpOf(f.arg)
	case "ln":
		outer = PowOf(f.arg, N(-1))
	default:
		// Unknown: return d/dx f(u) symbolically
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
	case "exp":
		return NFloat(math.Exp(v)), true
	case "ln":
		return NFloat(math.Log(v)), true
	case "abs":
		return NFloat(math.Abs(v)), true
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

// ============================================================
// Equation
// ============================================================

// Equation represents lhs = rhs.
type Equation struct {
	LHS, RHS Expr
}

// Eq creates a symbolic equation.
func Eq(lhs, rhs Expr) *Equation { return &Equation{LHS: lhs, RHS: rhs} }

func (e *Equation) String() string {
	return e.LHS.String() + " = " + e.RHS.String()
}

func (e *Equation) LaTeX() string {
	return e.LHS.LaTeX() + " = " + e.RHS.LaTeX()
}

// Residual returns lhs - rhs (the expression that equals zero when the equation is satisfied).
func (e *Equation) Residual() Expr {
	return AddOf(e.LHS, MulOf(N(-1), e.RHS)).Simplify()
}

// ============================================================
// Top-level convenience functions (match Python SymPy's API feel)
// ============================================================

// Simplify simplifies an expression.
func Simplify(e Expr) Expr { return e.Simplify() }

// String returns the string representation.
func String(e Expr) string { return e.String() }

// LaTeX returns the LaTeX representation.
func LaTeX(e Expr) string { return e.LaTeX() }

// Sub substitutes varName with value in expr.
func Sub(expr Expr, varName string, value Expr) Expr {
	return expr.Sub(varName, value).Simplify()
}

// Diff differentiates expr with respect to varName.
func Diff(expr Expr, varName string) Expr {
	return expr.Diff(varName).Simplify()
}

// Diff2 computes the second derivative.
func Diff2(expr Expr, varName string) Expr {
	return Diff(Diff(expr, varName), varName)
}

// DiffN computes the n-th derivative.
func DiffN(expr Expr, varName string, n int) Expr {
	result := expr
	for i := 0; i < n; i++ {
		result = Diff(result, varName)
	}
	return result
}

// Expand algebraically expands (distributes multiplication over addition).
func Expand(e Expr) Expr {
	return expandExpr(e).Simplify()
}

func expandExpr(e Expr) Expr {
	switch v := e.(type) {
	case *Mul:
		// Collect all factors after expanding each.
		expanded := make([]Expr, len(v.factors))
		for i, f := range v.factors {
			expanded[i] = expandExpr(f)
		}
		// Distribute: find the first Add factor and distribute.
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
					newFactors := append([]Expr{t}, rest...)
					terms[k] = expandExpr(MulOf(newFactors...))
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
		// (a+b)^n expansion for small integer n
		if n, ok := v.exp.(*Num); ok && n.IsInteger() {
			exp := n.val.Num().Int64()
			if exp >= 0 && exp <= 10 {
				result := Expr(N(1))
				base := expandExpr(v.base)
				for i := int64(0); i < exp; i++ {
					result = expandExpr(MulOf(result, base))
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

// FreeSymbols returns the set of symbol names present in expr.
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

// Degree returns the degree of expr as a polynomial in varName.
// Returns -1 if expr does not depend on varName.
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
			d := Degree(t, varName)
			if d > maxDeg {
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

// PolyCoeffsResult holds polynomial coefficients indexed by degree.
type PolyCoeffsResult map[int]Expr

// PolyCoeffs extracts coefficients of expr as a polynomial in varName.
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
				deg := int(n.val.Num().Int64())
				addCoeff(out, deg, N(1))
				return
			}
		}
		addCoeff(out, 0, e)
	case *Mul:
		// Separate variable-power factor from coefficient.
		deg := 0
		coeffFactors := []Expr{}
		for _, f := range v.factors {
			d := Degree(f, varName)
			if d > 0 {
				deg += d
			} else {
				coeffFactors = append(coeffFactors, f)
			}
		}
		var coeff Expr
		if len(coeffFactors) == 0 {
			coeff = N(1)
		} else if len(coeffFactors) == 1 {
			coeff = coeffFactors[0]
		} else {
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

// ============================================================
// Solvers
// ============================================================

// SolveResult carries the result of a solver operation.
type SolveResult struct {
	Solutions []Expr
	ExactForm bool
	Error     string
}

// SolveLinear solves a*x + b = 0 exactly.
// Returns x = -b/a.
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
		sol := numMul(numNeg(bn), numRecip(an))
		return SolveResult{Solutions: []Expr{sol}, ExactForm: true}
	}
	// Symbolic: x = -b/a
	sol := MulOf(N(-1), b, PowOf(a, N(-1))).Simplify()
	return SolveResult{Solutions: []Expr{sol}, ExactForm: false}
}

// SolveQuadratic solves a*x^2 + b*x + c = 0.
// Returns float64 roots; if discriminant < 0, returns complex roots as strings via Error.
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
		realPart := -bf / (2 * af)
		imagPart := math.Sqrt(-disc) / (2 * af)
		return SolveResult{
			Error: fmt.Sprintf("complex roots: %g ± %gi", realPart, imagPart),
		}
	}
	sq := math.Sqrt(disc)
	x1 := (-bf + sq) / (2 * af)
	x2 := (-bf - sq) / (2 * af)
	return SolveResult{
		Solutions: []Expr{NFloat(x1), NFloat(x2)},
		ExactForm: false,
	}
}

// SolveLinearSystem solves a 2x2 linear system using Cramer's rule.
// Equations: a1*x + b1*y = c1, a2*x + b2*y = c2.
func SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2 Expr) (xSol, ySol Expr, err error) {
	det := AddOf(MulOf(a1, b2), MulOf(N(-1), MulOf(a2, b1))).Simplify()
	dn, ok := det.Eval()
	if !ok || dn.IsZero() {
		return nil, nil, fmt.Errorf("system is singular or has no unique solution")
	}
	dx := AddOf(MulOf(c1, b2), MulOf(N(-1), MulOf(c2, b1))).Simplify()
	dy := AddOf(MulOf(a1, c2), MulOf(N(-1), MulOf(a2, c1))).Simplify()
	xSol = MulOf(dx, PowOf(det, N(-1))).Simplify()
	ySol = MulOf(dy, PowOf(det, N(-1))).Simplify()
	return xSol, ySol, nil
}

// ============================================================
// Integration (rule-based)
// ============================================================

// Integrate attempts symbolic integration of expr with respect to varName.
// Supported: constants, powers of x, sums, constant multiples, basic trig, exp, 1/x.
func Integrate(expr Expr, varName string) (Expr, bool) {
	expr = expr.Simplify()
	switch v := expr.(type) {
	case *Num:
		// ∫ c dx = c*x
		return MulOf(v, S(varName)), true

	case *Sym:
		if v.name == varName {
			// ∫ x dx = x^2/2
			return MulOf(F(1, 2), PowOf(S(varName), N(2))), true
		}
		// ∫ c dx = c*x
		return MulOf(v, S(varName)), true

	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			// ∫ x^n dx
			if n, ok2 := v.exp.(*Num); ok2 {
				if n.IsNegOne() {
					// ∫ x^(-1) dx = ln|x|
					return LnOf(AbsOf(S(varName))), true
				}
				newExp := numAdd(n, N(1))
				coeff := numRecip(newExp)
				return MulOf(coeff, PowOf(S(varName), newExp)), true
			}
		}
		return nil, false

	case *Mul:
		// Pull out constant coefficient.
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
		case "cos":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return SinOf(S(varName)), true
			}
		case "exp":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return ExpOf(S(varName)), true
			}
		}
		return nil, false
	}
	return nil, false
}

// DefiniteIntegrate computes ∫_a^b expr dx numerically using Gaussian quadrature (n=100).
func DefiniteIntegrate(expr Expr, varName string, a, b float64) float64 {
	// Gauss-Legendre 10-point quadrature on [a,b]
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
// Taylor series
// ============================================================

// TaylorSeries computes the Taylor series of expr around x=a up to order n.
func TaylorSeries(expr Expr, varName string, a Expr, order int) Expr {
	terms := []Expr{}
	current := expr
	factorial := N(1)
	aVal := a

	for k := 0; k <= order; k++ {
		if k > 0 {
			factorial = numMul(factorial, N(int64(k)))
		}
		coeff := MulOf(current.Sub(varName, aVal), PowOf(factorial, N(-1))).Simplify()
		if n, ok := coeff.(*Num); ok && n.IsZero() {
			current = Diff(current, varName)
			continue
		}
		var xTerm Expr
		if k == 0 {
			xTerm = coeff
		} else if k == 1 {
			xTerm = MulOf(coeff, AddOf(S(varName), MulOf(N(-1), aVal)))
		} else {
			xTerm = MulOf(coeff, PowOf(AddOf(S(varName), MulOf(N(-1), aVal)), N(int64(k))))
		}
		terms = append(terms, xTerm)
		current = Diff(current, varName)
	}
	return AddOf(terms...).Simplify()
}

// ============================================================
// JSON Serialization / Deserialization
// ============================================================

// ToJSON serializes an expression to a JSON string.
func ToJSON(e Expr) (string, error) {
	b, err := json.Marshal(e.toJSON())
	return string(b), err
}

// FromJSON deserializes an expression from a JSON map.
// Useful for AI agents sending structured expression trees.
func FromJSON(data map[string]interface{}) (Expr, error) {
	typ, ok := data["type"].(string)
	if !ok {
		return nil, fmt.Errorf("missing 'type' field")
	}
	switch typ {
	case "num":
		val, _ := data["value"].(string)
		r := new(big.Rat)
		if _, ok := r.SetString(val); !ok {
			return nil, fmt.Errorf("invalid num value: %s", val)
		}
		return &Num{val: r}, nil
	case "sym":
		name, _ := data["name"].(string)
		return S(name), nil
	case "add":
		rawTerms, _ := data["terms"].([]interface{})
		terms := make([]Expr, len(rawTerms))
		for i, t := range rawTerms {
			m, _ := t.(map[string]interface{})
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			terms[i] = e
		}
		return AddOf(terms...), nil
	case "mul":
		rawFactors, _ := data["factors"].([]interface{})
		factors := make([]Expr, len(rawFactors))
		for i, f := range rawFactors {
			m, _ := f.(map[string]interface{})
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			factors[i] = e
		}
		return MulOf(factors...), nil
	case "pow":
		baseM, _ := data["base"].(map[string]interface{})
		expM, _ := data["exp"].(map[string]interface{})
		base, err := FromJSON(baseM)
		if err != nil {
			return nil, err
		}
		exp, err := FromJSON(expM)
		if err != nil {
			return nil, err
		}
		return PowOf(base, exp), nil
	case "func":
		name, _ := data["name"].(string)
		argM, _ := data["arg"].(map[string]interface{})
		arg, err := FromJSON(argM)
		if err != nil {
			return nil, err
		}
		return funcOf(name, arg).Simplify(), nil
	}
	return nil, fmt.Errorf("unknown expression type: %s", typ)
}

// ============================================================
// MCP-style tool call interface
// ============================================================

// ToolRequest represents an incoming AI tool call for symbolic math.
type ToolRequest struct {
	Tool   string                 `json:"tool"`
	Params map[string]interface{} `json:"params"`
}

// ToolResponse represents the result of a tool call.
type ToolResponse struct {
	Result  interface{} `json:"result,omitempty"`
	LaTeX   string      `json:"latex,omitempty"`
	String  string      `json:"string,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// HandleToolCall processes a ToolRequest and returns a ToolResponse.
// Supported tools: simplify, diff, integrate, expand, solve_linear, solve_quadratic,
//                  free_symbols, degree, to_latex, taylor, substitute.
func HandleToolCall(req ToolRequest) ToolResponse {
	getExpr := func(key string) (Expr, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		switch val := v.(type) {
		case map[string]interface{}:
			return FromJSON(val)
		case string:
			return nil, fmt.Errorf("string expressions not yet supported; use JSON tree for '%s'", key)
		}
		return nil, fmt.Errorf("invalid type for param %s", key)
	}
	getString := func(key string) (string, error) {
		v, ok := req.Params[key]
		if !ok {
			return "", fmt.Errorf("missing param: %s", key)
		}
		s, ok := v.(string)
		if !ok {
			return "", fmt.Errorf("param %s must be a string", key)
		}
		return s, nil
	}
	respond := func(e Expr) ToolResponse {
		j, _ := ToJSON(e)
		var m map[string]interface{}
		json.Unmarshal([]byte(j), &m)
		return ToolResponse{Result: m, LaTeX: LaTeX(e), String: String(e)}
	}

	switch req.Tool {
	case "simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Simplify(e))

	case "diff":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Diff(e, v))

	case "integrate":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		result, ok := Integrate(e, v)
		if !ok {
			return ToolResponse{Error: "integration failed: unsupported form"}
		}
		return respond(result)

	case "expand":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Expand(e))

	case "substitute":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		val, err := getExpr("value")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Sub(e, v, val))

	case "to_latex":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{LaTeX: LaTeX(e), String: String(e)}

	case "free_symbols":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		syms := FreeSymbols(e)
		names := make([]string, 0, len(syms))
		for n := range syms {
			names = append(names, n)
		}
		sort.Strings(names)
		return ToolResponse{Result: names}

	case "degree":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{Result: Degree(e, v)}

	case "solve_linear":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		res := SolveLinear(a, b)
		if res.Error != "" {
			return ToolResponse{Error: res.Error}
		}
		sols := make([]map[string]interface{}, len(res.Solutions))
		latexSols := make([]string, len(res.Solutions))
		strSols := make([]string, len(res.Solutions))
		for i, s := range res.Solutions {
			j, _ := ToJSON(s)
			json.Unmarshal([]byte(j), &sols[i])
			latexSols[i] = LaTeX(s)
			strSols[i] = String(s)
		}
		return ToolResponse{
			Result: sols,
			LaTeX:  strings.Join(latexSols, ", "),
			String: strings.Join(strSols, ", "),
		}

	case "solve_quadratic":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c, err := getExpr("c")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		res := SolveQuadratic(a, b, c)
		if res.Error != "" {
			return ToolResponse{Error: res.Error}
		}
		strSols := make([]string, len(res.Solutions))
		for i, s := range res.Solutions {
			strSols[i] = String(s)
		}
		return ToolResponse{
			Result: strSols,
			String: strings.Join(strSols, ", "),
		}

	case "taylor":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		aExpr, err := getExpr("around")
		if err != nil {
			aExpr = N(0)
		}
		orderFloat, _ := req.Params["order"].(float64)
		order := int(orderFloat)
		if order <= 0 {
			order = 5
		}
		return respond(TaylorSeries(e, v, aExpr, order))
	}

	return ToolResponse{Error: fmt.Sprintf("unknown tool: %s", req.Tool)}
}

// ============================================================
// Pretty-print helpers
// ============================================================

// PrettyPrint returns a multi-line string suitable for terminal display.
func PrettyPrint(e Expr) string {
	return "  " + e.String() + "\n"
}

// MCPToolSpec returns a JSON schema describing available tools for AI agent use.
func MCPToolSpec() string {
	spec := map[string]interface{}{
		"tools": []map[string]interface{}{
			{
				"name":        "simplify",
				"description": "Simplify a symbolic expression",
				"inputSchema": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"expr": map[string]interface{}{"type": "object", "description": "JSON expression tree"}},
					"required":   []string{"expr"},
				},
			},
			{
				"name":        "diff",
				"description": "Differentiate an expression with respect to a variable",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expr": map[string]interface{}{"type": "object"},
						"var":  map[string]interface{}{"type": "string"},
					},
					"required": []string{"expr", "var"},
				},
			},
			{
				"name":        "integrate",
				"description": "Integrate an expression with respect to a variable (rule-based)",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expr": map[string]interface{}{"type": "object"},
						"var":  map[string]interface{}{"type": "string"},
					},
					"required": []string{"expr", "var"},
				},
			},
			{
				"name":        "expand",
				"description": "Algebraically expand an expression",
				"inputSchema": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"expr": map[string]interface{}{"type": "object"}},
					"required":   []string{"expr"},
				},
			},
			{
				"name":        "substitute",
				"description": "Substitute a variable with a value in an expression",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expr":  map[string]interface{}{"type": "object"},
						"var":   map[string]interface{}{"type": "string"},
						"value": map[string]interface{}{"type": "object"},
					},
					"required": []string{"expr", "var", "value"},
				},
			},
			{
				"name":        "to_latex",
				"description": "Convert an expression to LaTeX",
				"inputSchema": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"expr": map[string]interface{}{"type": "object"}},
					"required":   []string{"expr"},
				},
			},
			{
				"name":        "free_symbols",
				"description": "Return the set of free symbol names in an expression",
				"inputSchema": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"expr": map[string]interface{}{"type": "object"}},
					"required":   []string{"expr"},
				},
			},
			{
				"name":        "degree",
				"description": "Return the degree of an expression as a polynomial in a variable",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expr": map[string]interface{}{"type": "object"},
						"var":  map[string]interface{}{"type": "string"},
					},
					"required": []string{"expr", "var"},
				},
			},
			{
				"name":        "solve_linear",
				"description": "Solve a*x + b = 0 for x exactly",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"a": map[string]interface{}{"type": "object"},
						"b": map[string]interface{}{"type": "object"},
					},
					"required": []string{"a", "b"},
				},
			},
			{
				"name":        "solve_quadratic",
				"description": "Solve a*x^2 + b*x + c = 0",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"a": map[string]interface{}{"type": "object"},
						"b": map[string]interface{}{"type": "object"},
						"c": map[string]interface{}{"type": "object"},
					},
					"required": []string{"a", "b", "c"},
				},
			},
			{
				"name":        "taylor",
				"description": "Compute Taylor series of expression around a point up to given order",
				"inputSchema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"expr":   map[string]interface{}{"type": "object"},
						"var":    map[string]interface{}{"type": "string"},
						"around": map[string]interface{}{"type": "object"},
						"order":  map[string]interface{}{"type": "integer"},
					},
					"required": []string{"expr", "var"},
				},
			},
		},
	}
	b, _ := json.MarshalIndent(spec, "", "  ")
	return string(b)
}
