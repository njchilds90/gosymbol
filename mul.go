package gosymbol

import (
	"sort"
	"strings"
)

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
	powerMap := map[string]Expr{}
	powerExpr := map[string]Expr{}
	order := []string{}
	for _, f := range flat {
		if v, ok := f.(*Num); ok {
			coeff = numMul(coeff, v)
			continue
		}
		base := f
		exp := Expr(N(1))
		if p, ok := f.(*Pow); ok {
			base = p.base
			exp = p.exp
		}
		key := base.String()
		if _, seen := powerMap[key]; !seen {
			order = append(order, key)
			powerMap[key] = exp
			powerExpr[key] = base
			continue
		}
		powerMap[key] = AddOf(powerMap[key], exp)
	}
	if coeff.IsZero() {
		return N(0)
	}
	others := make([]Expr, 0, len(order))
	for _, key := range order {
		exp := powerMap[key].Simplify()
		if n, ok := exp.(*Num); ok && n.IsZero() {
			continue
		}
		base := powerExpr[key]
		if n, ok := exp.(*Num); ok && n.IsOne() {
			others = append(others, base)
		} else {
			others = append(others, PowOf(base, exp))
		}
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
