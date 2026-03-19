package gosymbol

import (
	"sort"
	"strings"
)

// ============================================================
// Add — sum of terms
// ============================================================

type Add struct{ terms []Expr }

func AddOf(terms ...Expr) Expr { return (&Add{terms: terms}).Simplify() }

func (a *Add) Simplify() Expr {
	if simplificationDepthExceeded(a) {
		return a
	}
	flat := make([]Expr, 0, len(a.terms))
	for _, t := range a.terms {
		s := t.Simplify()
		if inner, ok := s.(*Add); ok {
			flat = append(flat, inner.terms...)
		} else {
			flat = append(flat, s)
		}
		if simplificationWidthExceeded(len(flat)) {
			return &Add{terms: flat}
		}
	}
	numAccum := N(0)
	coeffs := map[string]*Num{}
	bases := map[string]Expr{}
	order := []string{}
	for _, t := range flat {
		if n, ok := t.(*Num); ok {
			numAccum = numAdd(numAccum, n)
			continue
		}
		coeff, basis := extractCoefficient(t)
		key := basis.String()
		if _, seen := coeffs[key]; !seen {
			order = append(order, key)
			coeffs[key] = N(0)
			bases[key] = basis
		}
		coeffs[key] = numAdd(coeffs[key], coeff)
	}
	result := []Expr{}
	sort.Strings(order)
	for _, key := range order {
		coeff := coeffs[key]
		if coeff.IsZero() {
			continue
		}
		basis := bases[key]
		if coeff.IsOne() {
			result = append(result, basis)
			continue
		}
		if coeff.IsNegOne() {
			result = append(result, MulOf(N(-1), basis))
		} else {
			result = append(result, MulOf(coeff, basis))
		}
	}
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

func (a *Add) Canonicalize() Expr { return Canonicalize(a) }

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
