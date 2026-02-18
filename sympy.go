// sympy.go
// Compact symbolic math engine in one file.
// Deterministic, rational-aware, rule-based, AI-embeddable.
//
// Limitations:
// - No advanced factoring
// - No symbolic matrix inversion
// - No symbolic limits beyond substitution
// - Simplification is rule-based, not canonical
// - Integration is pattern-based

package sympy

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
)

/* =======================
   Rational
======================= */

type Rational struct{ *big.Rat }

func NewInt(n int64) Rational     { return Rational{big.NewRat(n, 1)} }
func NewFrac(a, b int64) Rational { return Rational{big.NewRat(a, b)} }
func Zero() Rational              { return NewInt(0) }
func One() Rational               { return NewInt(1) }

func (r Rational) Add(o Rational) Rational { return Rational{new(big.Rat).Add(r.Rat, o.Rat)} }
func (r Rational) Sub(o Rational) Rational { return Rational{new(big.Rat).Sub(r.Rat, o.Rat)} }
func (r Rational) Mul(o Rational) Rational { return Rational{new(big.Rat).Mul(r.Rat, o.Rat)} }
func (r Rational) Div(o Rational) Rational { return Rational{new(big.Rat).Quo(r.Rat, o.Rat)} }
func (r Rational) Neg() Rational           { return Rational{new(big.Rat).Neg(r.Rat)} }
func (r Rational) IsZero() bool            { return r.Sign() == 0 }
func (r Rational) String() string          { return r.Rat.RatString() }

/* =======================
   Expr Core
======================= */

type Expr interface {
	Simplify() Expr
	String() string
	Sub(varName string, value Expr) Expr
}

/* Public Helpers */

func Simplify(e Expr) Expr { return e.Simplify() }
func String(e Expr) string { return e.Simplify().String() }

/* ---------- Num ---------- */

type Num struct{ V Rational }

func N(n int64) Expr    { return Num{NewInt(n)} }
func F(a, b int64) Expr { return Num{NewFrac(a, b)} }

func (n Num) Simplify() Expr { return n }
func (n Num) String() string { return n.V.String() }
func (n Num) Sub(string, Expr) Expr {
	return n
}

/* ---------- Sym ---------- */

type Sym struct{ Name string }

func S(name string) Expr { return Sym{Name: name} }

func (s Sym) Simplify() Expr { return s }
func (s Sym) String() string { return s.Name }
func (s Sym) Sub(v string, val Expr) Expr {
	if s.Name == v {
		return val
	}
	return s
}

/* ---------- Add ---------- */

type Add struct{ Terms []Expr }

func AddOf(terms ...Expr) Expr { return Add{terms}.Simplify() }

func (a Add) Simplify() Expr {
	var flat []Expr
	sum := Zero()

	for _, t := range a.Terms {
		t = t.Simplify()
		switch v := t.(type) {
		case Add:
			flat = append(flat, v.Terms...)
		case Num:
			sum = sum.Add(v.V)
		default:
			flat = append(flat, t)
		}
	}

	if !sum.IsZero() {
		flat = append(flat, Num{sum})
	}

	if len(flat) == 0 {
		return Num{Zero()}
	}
	if len(flat) == 1 {
		return flat[0]
	}

	sort.Slice(flat, func(i, j int) bool {
		return flat[i].String() < flat[j].String()
	})

	return Add{flat}
}

func (a Add) String() string {
	parts := make([]string, len(a.Terms))
	for i, t := range a.Terms {
		parts[i] = t.String()
	}
	return strings.Join(parts, " + ")
}

func (a Add) Sub(v string, val Expr) Expr {
	var out []Expr
	for _, t := range a.Terms {
		out = append(out, t.Sub(v, val))
	}
	return AddOf(out...)
}

/* ---------- Mul ---------- */

type Mul struct{ Factors []Expr }

func MulOf(factors ...Expr) Expr { return Mul{factors}.Simplify() }

func (m Mul) Simplify() Expr {
	var flat []Expr
	prod := One()

	for _, f := range m.Factors {
		f = f.Simplify()
		switch v := f.(type) {
		case Mul:
			flat = append(flat, v.Factors...)
		case Num:
			prod = prod.Mul(v.V)
		default:
			flat = append(flat, f)
		}
	}

	if prod.IsZero() {
		return Num{Zero()}
	}
	if prod.Cmp(big.NewRat(1, 1)) != 0 {
		flat = append(flat, Num{prod})
	}

	if len(flat) == 0 {
		return Num{prod}
	}
	if len(flat) == 1 {
		return flat[0]
	}

	sort.Slice(flat, func(i, j int) bool {
		return flat[i].String() < flat[j].String()
	})

	return Mul{flat}
}

func (m Mul) String() string {
	parts := make([]string, len(m.Factors))
	for i, f := range m.Factors {
		parts[i] = f.String()
	}
	return strings.Join(parts, "*")
}

func (m Mul) Sub(v string, val Expr) Expr {
	var out []Expr
	for _, f := range m.Factors {
		out = append(out, f.Sub(v, val))
	}
	return MulOf(out...)
}

/* ---------- Pow ---------- */

type Pow struct{ Base, Exp Expr }

func PowOf(b, e Expr) Expr { return Pow{b, e}.Simplify() }

func (p Pow) Simplify() Expr {
	b := p.Base.Simplify()
	e := p.Exp.Simplify()

	if en, ok := e.(Num); ok {
		if en.V.IsZero() {
			return Num{One()}
		}
		if en.V.Cmp(big.NewRat(1, 1)) == 0 {
			return b
		}
	}

	return Pow{b, e}
}

func (p Pow) String() string {
	return fmt.Sprintf("(%s)^%s", p.Base, p.Exp)
}

func (p Pow) Sub(v string, val Expr) Expr {
	return PowOf(p.Base.Sub(v, val), p.Exp.Sub(v, val))
}

/* =======================
   Polynomial Utilities
======================= */

func Degree(e Expr, v string) int {
	switch t := e.(type) {

	case Num:
		return 0

	case Sym:
		if t.Name == v {
			return 1
		}
		return 0

	case Add:
		max := 0
		for _, term := range t.Terms {
			d := Degree(term, v)
			if d > max {
				max = d
			}
		}
		return max

	case Mul:
		sum := 0
		for _, f := range t.Factors {
			sum += Degree(f, v)
		}
		return sum

	case Pow:
		if base, ok := t.Base.(Sym); ok && base.Name == v {
			if exp, ok := t.Exp.(Num); ok {
				i, _ := exp.V.Int64()
				return int(i)
			}
		}
	}

	return 0
}

// PolyCoeffs extracts polynomial coefficients: map[degree]Rational
func PolyCoeffs(e Expr, v string) map[int]Rational {
	coeffs := map[int]Rational{}

	var collect func(Expr)
	collect = func(ex Expr) {
		switch t := ex.(type) {

		case Add:
			for _, term := range t.Terms {
				collect(term)
			}

		case Mul:
			deg := Degree(t, v)
			c := One()
			for _, f := range t.Factors {
				if n, ok := f.(Num); ok {
					c = c.Mul(n.V)
				}
			}
			coeffs[deg] = coeffs[deg].Add(c)

		case Pow:
			deg := Degree(t, v)
			coeffs[deg] = coeffs[deg].Add(One())

		case Sym:
			if t.Name == v {
				coeffs[1] = coeffs[1].Add(One())
			}

		case Num:
			coeffs[0] = coeffs[0].Add(t.V)
		}
	}

	collect(e.Simplify())
	return coeffs
}

/* =======================
   Solvers
======================= */

func SolveLinear(a, b Rational) Rational {
	return b.Neg().Div(a)
}

func SolveQuadratic(a, b, c float64) []float64 {
	d := b*b - 4*a*c
	if d < 0 {
		return nil
	}
	s := math.Sqrt(d)
	return []float64{
		(-b + s) / (2 * a),
		(-b - s) / (2 * a),
	}
}

/* =======================
   Integration
======================= */

func Integrate(e Expr, v string) Expr {
	switch t := e.(type) {

	case Num:
		return MulOf(t, S(v))

	case Sym:
		if t.Name == v {
			return MulOf(F(1, 2), PowOf(t, N(2)))
		}
		return MulOf(t, S(v))

	case Add:
		var parts []Expr
		for _, term := range t.Terms {
			parts = append(parts, Integrate(term, v))
		}
		return AddOf(parts...)

	case Mul:
		if len(t.Factors) == 2 {
			if c, ok := t.Factors[0].(Num); ok {
				return MulOf(c, Integrate(t.Factors[1], v))
			}
		}

	case Pow:
		if base, ok := t.Base.(Sym); ok && base.Name == v {
			if exp, ok := t.Exp.(Num); ok {
				n := exp.V
				newExp := n.Add(One())
				return MulOf(
					Num{One().Div(newExp)},
					PowOf(base, Num{newExp}),
				)
			}
		}
	}

	return nil
}
