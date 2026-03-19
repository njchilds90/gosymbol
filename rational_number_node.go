package gosymbol

import (
	"fmt"
	"math/big"
)

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
func (n *Num) Canonicalize() Expr    { return Canonicalize(n) }
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
