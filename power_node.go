package gosymbol

import "math"

// ============================================================
// Pow — base^exponent
// ============================================================

type Pow struct{ base, exp Expr }

func PowOf(base, exp Expr) Expr { return (&Pow{base: base, exp: exp}).Simplify() }

func (p *Pow) Simplify() Expr {
	if simplificationDepthExceeded(p) {
		return p
	}
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
	if inner, ok := base.(*Func); ok && inner.name == "abs" {
		if en, ok2 := exp.(*Num); ok2 && en.IsInteger() && en.val.Num().Int64()%2 == 0 {
			return PowOf(inner.arg, exp)
		}
	}
	if inner, ok := base.(*Pow); ok {
		if outerExp, ok2 := exp.(*Num); ok2 && outerExp.IsOne() {
			return inner
		}
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
	if sym, ok := base.(*Sym); ok {
		if en, ok2 := exp.(*Num); ok2 && en.IsInteger() && en.val.Num().Int64()%2 == 0 && sym.assumptions.Positive {
			return PowOf(base, exp)
		}
	}
	return &Pow{base: base, exp: exp}
}

func (p *Pow) Canonicalize() Expr { return Canonicalize(p) }

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
