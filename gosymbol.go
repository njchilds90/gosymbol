// Package gosymbol provides a deterministic symbolic math kernel for Go.
//
// Design goals:
//   - Single file, zero external dependencies
//   - Exact rational arithmetic (math/big.Rat)
//   - Deterministic simplification and stable output
//   - AI/LLM friendly: JSON, LaTeX, and MCP-ready APIs
//   - Embeddable in Go services, CLI tools, and agent backends
package gosymbol

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
// Num — exact rational number
// ============================================================

type Num struct{ val *big.Rat }

func N(n int64) *Num { return &Num{val: new(big.Rat).SetInt64(n)} }
func F(p, q int64) *Num {
	if q == 0 {
		panic("gosymbol: denominator is zero")
	}
	return &Num{val: new(big.Rat).SetFrac(big.NewInt(p), big.NewInt(q))}
}
func NFloat(f float64) *Num { return &Num{val: new(big.Rat).SetFloat64(f)} }

func (n *Num) Simplify() Expr        { return n }
func (n *Num) Sub(string, Expr) Expr { return n }
func (n *Num) Diff(string) Expr      { return N(0) }
func (n *Num) Eval() (*Num, bool)    { return n, true }
func (n *Num) Equal(other Expr) bool { o, ok := other.(*Num); return ok && n.val.Cmp(o.val) == 0 }
func (n *Num) exprType() string      { return "num" }
func (n *Num) Float64() float64      { f, _ := n.val.Float64(); return f }
func (n *Num) IsZero() bool          { return n.val.Sign() == 0 }
func (n *Num) IsOne() bool           { return n.val.Cmp(new(big.Rat).SetInt64(1)) == 0 }
func (n *Num) IsNegOne() bool        { return n.val.Cmp(new(big.Rat).SetInt64(-1)) == 0 }
func (n *Num) IsInteger() bool       { return n.val.IsInt() }
func (n *Num) Rat() *big.Rat         { return new(big.Rat).Set(n.val) }
func (n *Num) IsPositive() bool      { return n.val.Sign() > 0 }
func (n *Num) IsNegative() bool      { return n.val.Sign() < 0 }

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

func numAdd(a, b *Num) *Num { return &Num{val: new(big.Rat).Add(a.val, b.val)} }
func numSub(a, b *Num) *Num { return &Num{val: new(big.Rat).Sub(a.val, b.val)} }
func numMul(a, b *Num) *Num { return &Num{val: new(big.Rat).Mul(a.val, b.val)} }
func numNeg(a *Num) *Num    { return &Num{val: new(big.Rat).Neg(a.val)} }
func numRecip(a *Num) *Num {
	if a.IsZero() {
		panic("gosymbol: division by zero")
	}
	return &Num{val: new(big.Rat).Inv(a.val)}
}
func numDiv(a, b *Num) *Num { return numMul(a, numRecip(b)) }
func numAbs(a *Num) *Num {
	r := new(big.Rat).Set(a.val)
	if r.Sign() < 0 {
		r.Neg(r)
	}
	return &Num{val: r}
}
func numCmp(a, b *Num) int { return a.val.Cmp(b.val) }
func gcdInt(a, b int64) int64 {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// ============================================================
// Sym — symbolic variable
// ============================================================

type Sym struct{ name string }

func S(name string) *Sym      { return &Sym{name: name} }
func (s *Sym) Simplify() Expr { return s }
func (s *Sym) String() string { return s.name }
func (s *Sym) LaTeX() string  { return s.name }
func (s *Sym) Eval() (*Num, bool) {
	return nil, false
}
func (s *Sym) Equal(other Expr) bool { o, ok := other.(*Sym); return ok && s.name == o.name }
func (s *Sym) exprType() string      { return "sym" }
func (s *Sym) Name() string          { return s.name }
func (s *Sym) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "sym", "name": s.name}
}
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

// ============================================================
// Add — sum of terms
// ============================================================

type Add struct{ terms []Expr }

func AddOf(terms ...Expr) Expr { return (&Add{terms: terms}).Simplify() }

func (a *Add) Simplify() Expr {
	flat := make([]Expr, 0, len(a.terms))
	for _, t := range a.terms {
		s := t.Simplify()
		if inner, ok := s.(*Add); ok {
			flat = append(flat, inner.terms...)
		} else {
			flat = append(flat, s)
		}
	}
	numAccum := N(0)
	symCoeffs := map[string]*Num{}
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
func (a *Add) Terms() []Expr { return a.terms }

// ============================================================
// Mul — product of factors
// ============================================================

type Mul struct{ factors []Expr }

func MulOf(factors ...Expr) Expr { return (&Mul{factors: factors}).Simplify() }

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

	// Precompute sort keys to avoid repeated String() calls in comparator.
	type keyed struct {
		e   Expr
		key string
	}
	ks := make([]keyed, len(others))
	for i, e := range others {
		ks[i] = keyed{e: e, key: e.String()}
	}
	sort.Slice(ks, func(i, j int) bool { return ks[i].key < ks[j].key })
	sortedOthers := make([]Expr, len(ks))
	for i := range ks {
		sortedOthers[i] = ks[i].e
	}
	others = sortedOthers

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
func (m *Mul) Factors() []Expr { return m.factors }

// ============================================================
// Pow — base^exponent
// ============================================================

type Pow struct{ base, exp Expr }

func PowOf(base, exp Expr) Expr { return (&Pow{base: base, exp: exp}).Simplify() }

func (p *Pow) Simplify() Expr {
	base := p.base.Simplify()
	exp := p.exp.Simplify()

	if en, ok := exp.(*Num); ok && en.IsZero() {
		return N(1)
	}
	if en, ok := exp.(*Num); ok && en.IsOne() {
		return base
	}

	// Handle 0^exp carefully.
	if bn, ok := base.(*Num); ok && bn.IsZero() {
		if en, ok2 := exp.(*Num); ok2 {
			// 0^0 is indeterminate; 0^negative is division by zero.
			if en.IsZero() || en.IsNegative() {
				return &Pow{base: base, exp: exp}
			}
		}
		return N(0)
	}

	if bn, ok := base.(*Num); ok && bn.IsOne() {
		return N(1)
	}
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
			if e < 0 && e >= -20 {
				posE := -e
				result := N(1)
				for i := int64(0); i < posE; i++ {
					result = numMul(result, bn)
				}
				// Will panic if result == 0, but base==0 was handled above.
				return numRecip(result)
			}
		}
	}
	if inner, ok := base.(*Pow); ok {
		newExp := MulOf(inner.exp, exp).Simplify()
		return PowOf(inner.base, newExp)
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

func (p *Pow) Diff(varName string) Expr {
	du := p.base.Diff(varName)
	dv := p.exp.Diff(varName)
	_, expIsNum := p.exp.(*Num)
	if expIsNum {
		newExp := AddOf(p.exp, N(-1))
		return MulOf(p.exp, PowOf(p.base, newExp), du)
	}
	_, baseIsNum := p.base.(*Num)
	if baseIsNum {
		return MulOf(PowOf(p.base, p.exp), LnOf(p.base), dv)
	}
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
		pf := math.Pow(bf, ef)
		if math.IsNaN(pf) || math.IsInf(pf, 0) {
			return nil, false
		}
		return NFloat(pf), true
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
func (p *Pow) Base() Expr    { return p.base }
func (p *Pow) ExpExpr() Expr { return p.exp }

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
func FloorOf(arg Expr) Expr { return funcOf("floor", arg).Simplify() }
func CeilOf(arg Expr) Expr  { return funcOf("ceil", arg).Simplify() }
func SignOf(arg Expr) Expr  { return funcOf("sign", arg).Simplify() }

func (f *Func) Simplify() Expr {
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
	case "cos":
		if isNumEqual(arg, 0) {
			return N(1)
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
	case "abs":
		if n2, ok := arg.(*Num); ok && n2.IsPositive() {
			return n2
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

func (f *Func) String() string { return f.name + "(" + f.arg.String() + ")" }

func (f *Func) LaTeX() string {
	switch f.name {
	case "sin", "cos", "tan", "exp", "ln", "sinh", "cosh", "tanh":
		return "\\" + f.name + "\\left(" + f.arg.LaTeX() + "\\right)"
	case "asin":
		return "\\arcsin\\left(" + f.arg.LaTeX() + "\\right)"
	case "acos":
		return "\\arccos\\left(" + f.arg.LaTeX() + "\\right)"
	case "atan":
		return "\\arctan\\left(" + f.arg.LaTeX() + "\\right)"
	case "abs":
		return "\\left|" + f.arg.LaTeX() + "\\right|"
	case "floor":
		return "\\lfloor " + f.arg.LaTeX() + " \\rfloor"
	case "ceil":
		return "\\lceil " + f.arg.LaTeX() + " \\rceil"
	case "sign":
		return "\\operatorname{sign}\\left(" + f.arg.LaTeX() + "\\right)"
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
// Matrix — symbolic matrix
// ============================================================

type Matrix struct {
	rows, cols int
	data       [][]Expr
}

func NewMatrix(rows, cols int) *Matrix {
	data := make([][]Expr, rows)
	for i := range data {
		data[i] = make([]Expr, cols)
		for j := range data[i] {
			data[i][j] = N(0)
		}
	}
	return &Matrix{rows: rows, cols: cols, data: data}
}

func MatrixFromSlice(rows, cols int, entries []Expr) *Matrix {
	if len(entries) != rows*cols {
		panic(fmt.Sprintf("gosymbol: MatrixFromSlice needs %d entries, got %d", rows*cols, len(entries)))
	}
	m := NewMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.data[i][j] = entries[i*cols+j]
		}
	}
	return m
}

func (m *Matrix) checkBounds(row, col int) {
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		panic(fmt.Sprintf("gosymbol: matrix index out of range [%d,%d] for %dx%d", row, col, m.rows, m.cols))
	}
}

func (m *Matrix) Get(row, col int) Expr {
	m.checkBounds(row, col)
	return m.data[row][col]
}
func (m *Matrix) Set(row, col int, val Expr) {
	m.checkBounds(row, col)
	m.data[row][col] = val
}
func (m *Matrix) Rows() int { return m.rows }
func (m *Matrix) Cols() int { return m.cols }

func (m *Matrix) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	for i := 0; i < m.rows; i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString("[")
		for j := 0; j < m.cols; j++ {
			if j > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(m.data[i][j].String())
		}
		sb.WriteString("]")
	}
	sb.WriteString("]")
	return sb.String()
}

func (m *Matrix) LaTeX() string {
	var sb strings.Builder
	sb.WriteString("\\begin{pmatrix}")
	for i := 0; i < m.rows; i++ {
		if i > 0 {
			sb.WriteString(" \\\\ ")
		}
		for j := 0; j < m.cols; j++ {
			if j > 0 {
				sb.WriteString(" & ")
			}
			sb.WriteString(m.data[i][j].LaTeX())
		}
	}
	sb.WriteString("\\end{pmatrix}")
	return sb.String()
}

func (m *Matrix) MatAdd(other *Matrix) *Matrix {
	if m.rows != other.rows || m.cols != other.cols {
		panic("gosymbol: matrix dimension mismatch in MatAdd")
	}
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = AddOf(m.data[i][j], other.data[i][j]).Simplify()
		}
	}
	return result
}

func (m *Matrix) MatSub(other *Matrix) *Matrix {
	if m.rows != other.rows || m.cols != other.cols {
		panic("gosymbol: matrix dimension mismatch in MatSub")
	}
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = AddOf(m.data[i][j], MulOf(N(-1), other.data[i][j])).Simplify()
		}
	}
	return result
}

func (m *Matrix) MatMul(other *Matrix) *Matrix {
	if m.cols != other.rows {
		panic("gosymbol: matrix dimension mismatch in MatMul")
	}
	result := NewMatrix(m.rows, other.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < other.cols; j++ {
			terms := make([]Expr, m.cols)
			for k := 0; k < m.cols; k++ {
				terms[k] = MulOf(m.data[i][k], other.data[k][j])
			}
			result.data[i][j] = AddOf(terms...).Simplify()
		}
	}
	return result
}

func (m *Matrix) Scale(scalar Expr) *Matrix {
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = MulOf(scalar, m.data[i][j]).Simplify()
		}
	}
	return result
}

func (m *Matrix) Transpose() *Matrix {
	result := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[j][i] = m.data[i][j]
		}
	}
	return result
}

func (m *Matrix) Trace() Expr {
	if m.rows != m.cols {
		panic("gosymbol: Trace requires a square matrix")
	}
	terms := make([]Expr, m.rows)
	for i := 0; i < m.rows; i++ {
		terms[i] = m.data[i][i]
	}
	return AddOf(terms...).Simplify()
}

func (m *Matrix) Det() Expr {
	if m.rows != m.cols {
		panic("gosymbol: Det requires a square matrix")
	}
	return matDet(m.data, m.rows)
}

func matDet(data [][]Expr, n int) Expr {
	if n == 1 {
		return data[0][0].Simplify()
	}
	if n == 2 {
		return AddOf(
			MulOf(data[0][0], data[1][1]),
			MulOf(N(-1), MulOf(data[0][1], data[1][0])),
		).Simplify()
	}
	terms := make([]Expr, n)
	for j := 0; j < n; j++ {
		minor := makeMinor(data, n, 0, j)
		sign := N(1)
		if j%2 == 1 {
			sign = N(-1)
		}
		terms[j] = MulOf(sign, data[0][j], matDet(minor, n-1))
	}
	return AddOf(terms...).Simplify()
}

func makeMinor(data [][]Expr, n, skipRow, skipCol int) [][]Expr {
	minor := make([][]Expr, n-1)
	mi := 0
	for i := 0; i < n; i++ {
		if i == skipRow {
			continue
		}
		minor[mi] = make([]Expr, n-1)
		mj := 0
		for j := 0; j < n; j++ {
			if j == skipCol {
				continue
			}
			minor[mi][mj] = data[i][j]
			mj++
		}
		mi++
	}
	return minor
}

func (m *Matrix) Inverse() (*Matrix, error) {
	if m.rows != m.cols {
		return nil, fmt.Errorf("gosymbol: Inverse requires a square matrix")
	}
	det := m.Det()
	if dn, ok := det.Eval(); ok && dn.IsZero() {
		return nil, fmt.Errorf("gosymbol: matrix is singular")
	}
	n := m.rows
	cof := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			minor := makeMinor(m.data, n, i, j)
			sign := N(1)
			if (i+j)%2 == 1 {
				sign = N(-1)
			}
			cof.data[i][j] = MulOf(sign, matDet(minor, n-1)).Simplify()
		}
	}
	adj := cof.Transpose()
	return adj.Scale(PowOf(det, N(-1))), nil
}

func (m *Matrix) ApplySub(varName string, value Expr) *Matrix {
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = m.data[i][j].Sub(varName, value).Simplify()
		}
	}
	return result
}

func (m *Matrix) ApplyDiff(varName string) *Matrix {
	result := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			result.data[i][j] = m.data[i][j].Diff(varName).Simplify()
		}
	}
	return result
}

func Identity(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.data[i][i] = N(1)
	}
	return m
}

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
		coeff    *Num
		idx      int
	}
	var trigTerms []trigTerm
	for idx, t := range add.terms {
		coeff, inner := extractCoefficient(t)
		if p, ok2 := inner.(*Pow); ok2 {
			if fn, ok3 := p.base.(*Func); ok3 {
				if en, ok4 := p.exp.(*Num); ok4 && en.IsInteger() && en.val.Num().Int64() == 2 {
					if fn.name == "sin" || fn.name == "cos" {
						trigTerms = append(trigTerms, trigTerm{fn.name, fn.arg.String(), coeff, idx})
					}
				}
			}
		}
	}
	for i := 0; i < len(trigTerms); i++ {
		for j := i + 1; j < len(trigTerms); j++ {
			ti, tj := trigTerms[i], trigTerms[j]
			if ti.argStr == tj.argStr && ti.funcName != tj.funcName && numCmp(ti.coeff, tj.coeff) == 0 {
				newTerms := []Expr{}
				for idx, t := range add.terms {
					if idx != ti.idx && idx != tj.idx {
						newTerms = append(newTerms, t)
					}
				}
				newTerms = append(newTerms, ti.coeff)
				return AddOf(newTerms...).Simplify()
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

// ============================================================
// Solvers
// ============================================================

type SolveResult struct {
	Solutions []Expr
	ExactForm bool
	Error     string
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

// ============================================================
// Partial Fractions
// ============================================================

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

// ============================================================
// JSON Serialization
// ============================================================

func ToJSON(e Expr) (string, error) {
	b, err := json.Marshal(e.toJSON())
	return string(b), err
}

func FromJSON(data map[string]interface{}) (Expr, error) {
	if data == nil {
		return nil, fmt.Errorf("expression must be an object")
	}
	typAny, ok := data["type"]
	if !ok {
		return nil, fmt.Errorf("missing 'type' field")
	}
	typ, ok := typAny.(string)
	if !ok || typ == "" {
		return nil, fmt.Errorf("field 'type' must be a non-empty string")
	}

	subObj := func(field string) (map[string]interface{}, error) {
		v, ok := data[field]
		if !ok {
			return nil, fmt.Errorf("%s: missing %q", typ, field)
		}
		m, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: %q must be an object", typ, field)
		}
		return m, nil
	}

	subObjArray := func(field string) ([]map[string]interface{}, error) {
		v, ok := data[field]
		if !ok {
			return nil, fmt.Errorf("%s: missing %q", typ, field)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: %q must be an array", typ, field)
		}
		out := make([]map[string]interface{}, len(raw))
		for i, it := range raw {
			m, ok := it.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("%s: %q[%d] must be an object", typ, field, i)
			}
			out[i] = m
		}
		return out, nil
	}

	subString := func(field string) (string, error) {
		v, ok := data[field]
		if !ok {
			return "", fmt.Errorf("%s: missing %q", typ, field)
		}
		s, ok := v.(string)
		if !ok || s == "" {
			return "", fmt.Errorf("%s: %q must be a non-empty string", typ, field)
		}
		return s, nil
	}

	subNumberAsInt := func(field string) (int, error) {
		v, ok := data[field]
		if !ok {
			return 0, fmt.Errorf("%s: missing %q", typ, field)
		}
		n, ok := v.(float64)
		if !ok {
			return 0, fmt.Errorf("%s: %q must be a number", typ, field)
		}
		return int(n), nil
	}

	switch typ {
	case "num":
		valAny, ok := data["value"]
		if !ok {
			return nil, fmt.Errorf("num: missing 'value'")
		}
		val, ok := valAny.(string)
		if !ok || val == "" {
			return nil, fmt.Errorf("num: 'value' must be a non-empty string")
		}
		r := new(big.Rat)
		if _, ok := r.SetString(val); !ok {
			return nil, fmt.Errorf("invalid num value: %s", val)
		}
		return &Num{val: r}, nil

	case "sym":
		name, err := subString("name")
		if err != nil {
			return nil, err
		}
		return S(name), nil

	case "add":
		objs, err := subObjArray("terms")
		if err != nil {
			return nil, err
		}
		terms := make([]Expr, len(objs))
		for i, o := range objs {
			e, err := FromJSON(o)
			if err != nil {
				return nil, fmt.Errorf("add: terms[%d]: %w", i, err)
			}
			terms[i] = e
		}
		return AddOf(terms...), nil

	case "mul":
		objs, err := subObjArray("factors")
		if err != nil {
			return nil, err
		}
		factors := make([]Expr, len(objs))
		for i, o := range objs {
			e, err := FromJSON(o)
			if err != nil {
				return nil, fmt.Errorf("mul: factors[%d]: %w", i, err)
			}
			factors[i] = e
		}
		return MulOf(factors...), nil

	case "pow":
		baseM, err := subObj("base")
		if err != nil {
			return nil, err
		}
		expM, err := subObj("exp")
		if err != nil {
			return nil, err
		}
		base, err := FromJSON(baseM)
		if err != nil {
			return nil, fmt.Errorf("pow: base: %w", err)
		}
		exp, err := FromJSON(expM)
		if err != nil {
			return nil, fmt.Errorf("pow: exp: %w", err)
		}
		return PowOf(base, exp), nil

	case "func":
		name, err := subString("name")
		if err != nil {
			return nil, err
		}
		argM, err := subObj("arg")
		if err != nil {
			return nil, err
		}
		arg, err := FromJSON(argM)
		if err != nil {
			return nil, fmt.Errorf("func: arg: %w", err)
		}
		return funcOf(name, arg).Simplify(), nil

	case "bigo":
		v, err := subString("var")
		if err != nil {
			return nil, err
		}
		order, err := subNumberAsInt("order")
		if err != nil {
			return nil, err
		}
		return OTerm(v, order), nil
	}
	return nil, fmt.Errorf("unknown expression type: %s", typ)
}

// ============================================================
// MCP Tool Interface
// ============================================================

type ToolRequest struct {
	Tool   string                 `json:"tool"`
	Params map[string]interface{} `json:"params"`
}

type ToolResponse struct {
	Result interface{} `json:"result,omitempty"`
	LaTeX  string      `json:"latex,omitempty"`
	String string      `json:"string,omitempty"`
	Error  string      `json:"error,omitempty"`
}

func HandleToolCall(req ToolRequest) ToolResponse {
	getExpr := func(key string) (Expr, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		val, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid type for param %s", key)
		}
		return FromJSON(val)
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
	getStrings := func(key string) ([]string, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be array", key)
		}
		result := make([]string, len(raw))
		for i, r := range raw {
			s, ok := r.(string)
			if !ok {
				return nil, fmt.Errorf("param %s[%d] must be string", key, i)
			}
			result[i] = s
		}
		return result, nil
	}
	getExprList := func(key string) ([]Expr, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be array", key)
		}
		result := make([]Expr, len(raw))
		for i, r := range raw {
			m, ok := r.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("param %s[%d] must be expression object", key, i)
			}
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			result[i] = e
		}
		return result, nil
	}
	getMatrix := func(key string) (*Matrix, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be matrix object", key)
		}

		rowsF, ok := raw["rows"].(float64)
		if !ok {
			return nil, fmt.Errorf("matrix.rows must be a number")
		}
		colsF, ok := raw["cols"].(float64)
		if !ok {
			return nil, fmt.Errorf("matrix.cols must be a number")
		}
		rows, cols := int(rowsF), int(colsF)
		if rows <= 0 || cols <= 0 {
			return nil, fmt.Errorf("matrix dimensions must be positive")
		}

		entriesRawAny, ok := raw["entries"]
		if !ok {
			return nil, fmt.Errorf("matrix.entries missing")
		}
		entriesRaw, ok := entriesRawAny.([]interface{})
		if !ok {
			return nil, fmt.Errorf("matrix.entries must be an array")
		}
		if len(entriesRaw) != rows*cols {
			return nil, fmt.Errorf("matrix entries count mismatch")
		}
		entries := make([]Expr, rows*cols)
		for i, er := range entriesRaw {
			m, ok := er.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("matrix entry %d must be expression", i)
			}
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			entries[i] = e
		}
		return MatrixFromSlice(rows, cols, entries), nil
	}
	respond := func(e Expr) ToolResponse {
		return ToolResponse{Result: e.toJSON(), LaTeX: LaTeX(e), String: String(e)}
	}
	respondMatrix := func(mat *Matrix) ToolResponse {
		return ToolResponse{
			Result: map[string]interface{}{"rows": mat.rows, "cols": mat.cols, "string": mat.String()},
			LaTeX:  mat.LaTeX(),
			String: mat.String(),
		}
	}
	solvesTool := func(res SolveResult) ToolResponse {
		if res.Error != "" && len(res.Solutions) == 0 {
			return ToolResponse{Error: res.Error}
		}
		strs := make([]string, len(res.Solutions))
		for i, s := range res.Solutions {
			strs[i] = String(s)
		}
		resp := ToolResponse{Result: strs, String: strings.Join(strs, ", ")}
		if res.Error != "" {
			resp.Error = res.Error
		}
		return resp
	}

	switch req.Tool {
	case "simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Simplify(e))

	case "deep_simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(DeepSimplify(e))

	case "trig_simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(TrigSimplify(e))

	case "canonicalize":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Canonicalize(e))

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

	case "diff2":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Diff2(e, v))

	case "diffn":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		nAny, ok := req.Params["n"]
		if !ok {
			return ToolResponse{Error: "missing param: n"}
		}
		nF, ok := nAny.(float64)
		if !ok {
			return ToolResponse{Error: "param n must be a number"}
		}
		n := int(nF)
		if n < 0 {
			return ToolResponse{Error: "param n must be >= 0"}
		}
		return respond(DiffN(e, v, n))

	case "pdiff":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(PDiff(e, v))

	case "gradient":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		grad := Gradient(e, vars)
		strs := make([]string, len(grad))
		latexStrs := make([]string, len(grad))
		for i, g := range grad {
			strs[i] = String(g)
			latexStrs[i] = LaTeX(g)
		}
		return ToolResponse{
			Result: strs,
			String: "[" + strings.Join(strs, ", ") + "]",
			LaTeX:  "[" + strings.Join(latexStrs, ", ") + "]",
		}

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

	case "definite_integrate":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		aF, ok := req.Params["a"].(float64)
		if !ok {
			return ToolResponse{Error: "param a must be a number"}
		}
		bF, ok := req.Params["b"].(float64)
		if !ok {
			return ToolResponse{Error: "param b must be a number"}
		}
		result := DefiniteIntegrate(e, v, aF, bF)
		return ToolResponse{Result: result, String: fmt.Sprintf("%.10g", result)}

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

	case "poly_coeffs":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		coeffs := PolyCoeffs(e, v)
		result := map[string]string{}
		for deg, c := range coeffs {
			result[fmt.Sprintf("%d", deg)] = String(c)
		}
		return ToolResponse{Result: result}

	case "collect":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Collect(e, v))

	case "cancel":
		num, err := getExpr("num")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		denom, err := getExpr("denom")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Cancel(num, denom))

	case "apart":
		num, err := getExpr("num")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		denom, err := getExpr("denom")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		result := Apart(num, denom, v)
		strs := make([]string, len(result.Terms))
		for i, t := range result.Terms {
			strs[i] = String(t)
		}
		return ToolResponse{Result: strs, String: strings.Join(strs, " + "), Error: result.Error}

	case "factor":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		fr := Factor(e, v)
		strs := make([]string, len(fr.Factors))
		for i, f := range fr.Factors {
			strs[i] = String(f)
		}
		return ToolResponse{Result: strs, String: strings.Join(strs, " * ")}

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
			sols[i] = s.toJSON()
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
		return solvesTool(SolveQuadratic(a, b, c))

	case "solve_quadratic_exact":
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
		return solvesTool(SolveQuadraticExact(a, b, c))

	case "solve_cubic":
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
		d, err := getExpr("d")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return solvesTool(SolveCubic(a, b, c, d))

	case "solve_polynomial_newton":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		rangeF, _ := req.Params["range"].(float64)
		tolF, _ := req.Params["tol"].(float64)
		iterF, _ := req.Params["max_iter"].(float64)
		return solvesTool(SolvePolynomialNewton(e, v, rangeF, tolF, int(iterF)))

	case "solve_system_2x2":
		a1, err := getExpr("a1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b1, err := getExpr("b1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c1, err := getExpr("c1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		a2, err := getExpr("a2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b2, err := getExpr("b2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c2, err := getExpr("c2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		x, y, err := SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2)
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{
			Result: map[string]string{"x": String(x), "y": String(y)},
			String: "x=" + String(x) + ", y=" + String(y),
			LaTeX:  "x=" + LaTeX(x) + ",\\ y=" + LaTeX(y),
		}

	case "limit":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		pt, err := getExpr("point")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		res := Limit(e, v, pt)
		if !res.Success {
			return ToolResponse{Error: res.Error}
		}
		return respond(res.Value)

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

	case "taylor_remainder":
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
		return respond(TaylorSeriesWithRemainder(e, v, aExpr, order))

	case "maclaurin":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		orderFloat, _ := req.Params["order"].(float64)
		order := int(orderFloat)
		if order <= 0 {
			order = 5
		}
		return respond(MaclaurinSeries(e, v, order))

	case "jacobian":
		exprs, err := getExprList("exprs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(Jacobian(exprs, vars))

	case "hessian":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(Hessian(e, vars))

	case "laplacian":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Laplacian(e, vars))

	case "divergence":
		exprs, err := getExprList("exprs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Divergence(exprs, vars))

	case "matrix_det":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(m.Det())

	case "matrix_inv":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		inv, err := m.Inverse()
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(inv)

	case "matrix_trace":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(m.Trace())

	case "matrix_mul":
		m1, err := getMatrix("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		m2, err := getMatrix("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(m1.MatMul(m2))

	case "mcp_spec":
		return ToolResponse{Result: MCPToolSpec(), String: "MCP tool specification"}
	}

	return ToolResponse{Error: fmt.Sprintf("unknown tool: %s", req.Tool)}
}

// ============================================================
// Pretty-print and MCP spec
// ============================================================

func PrettyPrint(e Expr) string { return "  " + e.String() + "\n" }

func MCPToolSpec() string {
	tools := []map[string]interface{}{
		ts("simplify", "Simplify a symbolic expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("deep_simplify", "Apply multiple simplification passes including trig identities", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("trig_simplify", "Apply trig identities (sin²+cos²=1, exp(ln(x))=x, etc.)", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("canonicalize", "Expand and canonicalize expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("expand", "Algebraically expand expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("collect", "Collect terms by powers of variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("cancel", "Simplify rational num/denom", []string{"num", "denom"}, map[string]string{"num": "object", "denom": "object"}),
		ts("factor", "Factor polynomial in variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("apart", "Partial fraction decomposition", []string{"num", "denom", "var"}, map[string]string{"num": "object", "denom": "object", "var": "string"}),
		ts("substitute", "Substitute var with value", []string{"expr", "var", "value"}, map[string]string{"expr": "object", "var": "string", "value": "object"}),
		ts("to_latex", "Convert to LaTeX", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("free_symbols", "Return free symbol names", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("diff", "First derivative d/dx", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("diff2", "Second derivative d²/dx²", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("diffn", "nth derivative. Requires n (int)", []string{"expr", "var", "n"}, map[string]string{"expr": "object", "var": "string", "n": "integer"}),
		ts("pdiff", "Partial derivative ∂/∂var", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("gradient", "Gradient vector ∇f. Requires vars (string[])", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("jacobian", "Jacobian matrix. Requires exprs (array) and vars (array)", []string{"exprs", "vars"}, map[string]string{"exprs": "array", "vars": "array"}),
		ts("hessian", "Hessian matrix of second partials", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("laplacian", "Laplacian ∇²f", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("divergence", "Divergence ∇·F", []string{"exprs", "vars"}, map[string]string{"exprs": "array", "vars": "array"}),
		ts("integrate", "Symbolic integration (rule-based)", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("definite_integrate", "Numerical ∫_a^b. Requires a,b (numbers)", []string{"expr", "var", "a", "b"}, map[string]string{"expr": "object", "var": "string", "a": "number", "b": "number"}),
		ts("taylor", "Taylor series", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "around": "object", "order": "integer"}),
		ts("taylor_remainder", "Taylor series with BigO remainder", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "order": "integer"}),
		ts("maclaurin", "Maclaurin series (Taylor around 0)", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "order": "integer"}),
		ts("degree", "Polynomial degree in variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("poly_coeffs", "Extract polynomial coefficients by degree", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("limit", "lim_{var->point} expr", []string{"expr", "var", "point"}, map[string]string{"expr": "object", "var": "string", "point": "object"}),
		ts("solve_linear", "Solve a*x+b=0 exactly", []string{"a", "b"}, map[string]string{"a": "object", "b": "object"}),
		ts("solve_quadratic", "Solve a*x²+b*x+c=0 (float)", []string{"a", "b", "c"}, map[string]string{"a": "object", "b": "object", "c": "object"}),
		ts("solve_quadratic_exact", "Solve a*x²+b*x+c=0 with exact roots when possible", []string{"a", "b", "c"}, map[string]string{"a": "object", "b": "object", "c": "object"}),
		ts("solve_cubic", "Solve a*x³+b*x²+c*x+d=0 (Cardano)", []string{"a", "b", "c", "d"}, map[string]string{}),
		ts("solve_polynomial_newton", "Numerical root finding. Optional: range, tol, max_iter", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("solve_system_2x2", "2×2 linear system: a1*x+b1*y=c1, a2*x+b2*y=c2", []string{"a1", "b1", "c1", "a2", "b2", "c2"}, map[string]string{}),
		ts("matrix_det", "Matrix det. matrix={rows,cols,entries:[expr,...]}", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_inv", "Symbolic matrix inverse", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_trace", "Matrix trace", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_mul", "Matrix multiply a*b", []string{"a", "b"}, map[string]string{"a": "object", "b": "object"}),
		ts("mcp_spec", "Return this tool schema", []string{}, map[string]string{}),
	}
	spec := map[string]interface{}{"tools": tools}
	b, _ := json.MarshalIndent(spec, "", "  ")
	return string(b)
}

func ts(name, description string, required []string, props map[string]string) map[string]interface{} {
	properties := map[string]interface{}{}
	for k, typ := range props {
		properties[k] = map[string]interface{}{"type": typ}
	}
	return map[string]interface{}{
		"name":        name,
		"description": description,
		"inputSchema": map[string]interface{}{
			"type":       "object",
			"properties": properties,
			"required":   required,
		},
	}
}