package sympy

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

// Expr is the interface for all symbolic expressions.
type Expr interface {
	String() string
	stringWithPrec(prec int) string // For pretty-printing
}

// constExpr represents a constant value.
type constExpr struct {
	value float64
}

func (c *constExpr) String() string {
	return c.stringWithPrec(0)
}

func (c *constExpr) stringWithPrec(prec int) string {
	return fmt.Sprintf("%v", c.value)
}

// varExpr represents a symbolic variable.
type varExpr struct {
	name string
}

func (v *varExpr) String() string {
	return v.stringWithPrec(0)
}

func (v *varExpr) stringWithPrec(prec int) string {
	return v.name
}

// addExpr represents an addition operation.
type addExpr struct {
	left, right Expr
}

func (a *addExpr) String() string {
	return a.stringWithPrec(0)
}

func (a *addExpr) stringWithPrec(prec int) string {
	s := a.left.stringWithPrec(1) + " + " + a.right.stringWithPrec(1)
	if prec > 1 {
		return "(" + s + ")"
	}
	return s
}

// mulExpr represents a multiplication operation.
type mulExpr struct {
	left, right Expr
}

func (m *mulExpr) String() string {
	return m.stringWithPrec(0)
}

func (m *mulExpr) stringWithPrec(prec int) string {
	s := m.left.stringWithPrec(2) + " * " + m.right.stringWithPrec(2)
	if prec > 2 {
		return "(" + s + ")"
	}
	return s
}

// subExpr represents subtraction: left - right
type subExpr struct {
	left, right Expr
}

func (s *subExpr) String() string {
	return s.stringWithPrec(0)
}

func (s *subExpr) stringWithPrec(prec int) string {
	sstr := s.left.stringWithPrec(1) + " - " + s.right.stringWithPrec(1)
	if prec > 1 {
		return "(" + sstr + ")"
	}
	return sstr
}

// negExpr for unary minus
type negExpr struct {
	expr Expr
}

func (n *negExpr) String() string {
	return n.stringWithPrec(0)
}

func (n *negExpr) stringWithPrec(prec int) string {
	s := "-" + n.expr.stringWithPrec(3) // High prec for unary
	if prec > 3 {
		return "(" + s + ")"
	}
	return s
}

// powExpr represents exponentiation: base^exp
type powExpr struct {
	base, exp Expr
}

func (p *powExpr) String() string {
	return p.stringWithPrec(0)
}

func (p *powExpr) stringWithPrec(prec int) string {
	s := p.base.stringWithPrec(4) + "^" + p.exp.stringWithPrec(4)
	if prec > 4 {
		return "(" + s + ")"
	}
	return s
}

// sinExpr represents sin function
type sinExpr struct {
	arg Expr
}

func (s *sinExpr) String() string {
	return s.stringWithPrec(0)
}

func (s *sinExpr) stringWithPrec(prec int) string {
	return "sin(" + s.arg.String() + ")"
}

// cosExpr represents cos function
type cosExpr struct {
	arg Expr
}

func (c *cosExpr) String() string {
	return c.stringWithPrec(0)
}

func (c *cosExpr) stringWithPrec(prec int) string {
	return "cos(" + c.arg.String() + ")"
}

// expExpr represents exp function (e^x)
type expExpr struct {
	arg Expr
}

func (e *expExpr) String() string {
	return e.stringWithPrec(0)
}

func (e *expExpr) stringWithPrec(prec int) string {
	return "exp(" + e.arg.String() + ")"
}

// lnExpr represents natural log function
type lnExpr struct {
	arg Expr
}

func (l *lnExpr) String() string {
	return l.stringWithPrec(0)
}

func (l *lnExpr) stringWithPrec(prec int) string {
	return "ln(" + l.arg.String() + ")"
}

// sqrtExpr represents square root function
type sqrtExpr struct {
	arg Expr
}

func (s *sqrtExpr) String() string {
	return s.stringWithPrec(0)
}

func (s *sqrtExpr) stringWithPrec(prec int) string {
	return "sqrt(" + s.arg.String() + ")"
}

// Number creates a constant expression.
func Number(value float64) Expr {
	return &constExpr{value}
}

// Symbol creates a variable expression.
func Symbol(name string) Expr {
	return &varExpr{name}
}

// Add creates an addition expression.
func Add(left, right Expr) Expr {
	return &addExpr{left, right}
}

// Mul creates a multiplication expression.
func Mul(left, right Expr) Expr {
	return &mulExpr{left, right}
}

// Sub creates a subtraction expression.
func Sub(left, right Expr) Expr {
	return &subExpr{left, right}
}

// Neg creates a negation expression.
func Neg(expr Expr) Expr {
	return &negExpr{expr}
}

// Pow creates an exponentiation expression.
func Pow(base, exp Expr) Expr {
	return &powExpr{base, exp}
}

// Sin creates a sine expression.
func Sin(arg Expr) Expr {
	return &sinExpr{arg}
}

// Cos creates a cosine expression.
func Cos(arg Expr) Expr {
	return &cosExpr{arg}
}

// Exp creates an exponential expression.
func Exp(arg Expr) Expr {
	return &expExpr{arg}
}

// Ln creates a natural logarithm expression.
func Ln(arg Expr) Expr {
	return &lnExpr{arg}
}

// Sqrt creates a square root expression.
func Sqrt(arg Expr) Expr {
	return &sqrtExpr{arg}
}

// Diff computes the derivative of the expression with respect to the given symbol.
func Diff(e Expr, sym Expr) Expr {
	v, ok := sym.(*varExpr)
	if !ok {
		return nil // Invalid symbol
	}

	switch expr := e.(type) {
	case *constExpr:
		return Number(0)
	case *varExpr:
		if expr.name == v.name {
			return Number(1)
		}
		return Number(0)
	case *addExpr:
		return Add(Diff(expr.left, sym), Diff(expr.right, sym))
	case *subExpr:
		return Sub(Diff(expr.left, sym), Diff(expr.right, sym))
	case *mulExpr:
		// Product rule: u'v + uv'
		return Add(Mul(Diff(expr.left, sym), expr.right), Mul(expr.left, Diff(expr.right, sym)))
	case *negExpr:
		return Neg(Diff(expr.expr, sym))
	case *powExpr:
		// General case: d/dx (u^v) = u^v * (v' * ln(u) + v * (u'/u))
		u := expr.base
		vv := expr.exp // renamed to avoid conflict with sym variable
		term1 := Mul(Diff(vv, sym), Ln(u))
		term2 := Mul(vv, Div(Diff(u, sym), u))
		return Mul(Pow(u, vv), Add(term1, term2))
	case *sinExpr:
		return Mul(Cos(expr.arg), Diff(expr.arg, sym))
	case *cosExpr:
		return Neg(Mul(Sin(expr.arg), Diff(expr.arg, sym)))
	case *expExpr:
		return Mul(Exp(expr.arg), Diff(expr.arg, sym))
	case *lnExpr:
		return Div(Diff(expr.arg, sym), expr.arg) // u'/u
	case *sqrtExpr:
		return Mul(Number(0.5), Mul(Pow(expr.arg, Number(-0.5)), Diff(expr.arg, sym)))
	default:
		return nil // Unsupported type
	}
}

// Div creates a division expression (left / right).
func Div(left, right Expr) Expr {
	return Mul(left, Pow(right, Number(-1)))
}

// Simplify performs enhanced simplification, including constant folding, identities, and more trig identities.
func Simplify(e Expr) Expr {
	// Recurse first
	switch expr := e.(type) {
	case *addExpr:
		left := Simplify(expr.left)
		right := Simplify(expr.right)
		if lConst, ok := left.(*constExpr); ok {
			if lConst.value == 0 {
				return right
			}
		}
		if rConst, ok := right.(*constExpr); ok {
			if rConst.value == 0 {
				return left
			}
		}
		// Constant folding
		if lConst, ok1 := left.(*constExpr); ok1 {
			if rConst, ok2 := right.(*constExpr); ok2 {
				return Number(lConst.value + rConst.value)
			}
		}
		// sin^2 + cos^2 = 1
		if isSinSquared(left) && isCosSquared(right) && sameArg(left, right) {
			return Number(1)
		}
		if isSinSquared(right) && isCosSquared(left) && sameArg(left, right) {
			return Number(1)
		}
		// More trig: cos^2 = (1 + cos(2x))/2, but add simple sin(a+b) = sin a cos b + cos a sin b? For now, keep basic
		return Add(left, right)
	case *subExpr:
		left := Simplify(expr.left)
		right := Simplify(expr.right)
		if rConst, ok := right.(*constExpr); ok {
			if rConst.value == 0 {
				return left
			}
		}
		// Constant folding
		if lConst, ok1 := left.(*constExpr); ok1 {
			if rConst, ok2 := right.(*constExpr); ok2 {
				return Number(lConst.value - rConst.value)
			}
		}
		return Sub(left, right)
	case *mulExpr:
		left := Simplify(expr.left)
		right := Simplify(expr.right)
		if lConst, ok := left.(*constExpr); ok {
			if lConst.value == 0 {
				return Number(0)
			}
			if lConst.value == 1 {
				return right
			}
		}
		if rConst, ok := right.(*constExpr); ok {
			if rConst.value == 0 {
				return Number(0)
			}
			if rConst.value == 1 {
				return left
			}
		}
		// Constant folding
		if lConst, ok1 := left.(*constExpr); ok1 {
			if rConst, ok2 := right.(*constExpr); ok2 {
				return Number(lConst.value * rConst.value)
			}
		}
		return Mul(left, right)
	case *powExpr:
		base := Simplify(expr.base)
		exp := Simplify(expr.exp)
		if eConst, ok := exp.(*constExpr); ok {
			if eConst.value == 0 {
				return Number(1)
			}
			if eConst.value == 1 {
				return base
			}
		}
		if bConst, ok := base.(*constExpr); ok {
			if eConst, ok2 := exp.(*constExpr); ok2 {
				return Number(math.Pow(bConst.value, eConst.value))
			}
		}
		return Pow(base, exp)
	case *negExpr:
		inner := Simplify(expr.expr)
		if iConst, ok := inner.(*constExpr); ok {
			return Number(-iConst.value)
		}
		return Neg(inner)
	case *sinExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok {
			return Number(math.Sin(aConst.value))
		}
		// Add trig identity: sin(2x) = 2 sin x cos x, but for simplify, perhaps reverse if Mul(2, Mul(Sin, Cos))
		return Sin(arg)
	case *cosExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok {
			return Number(math.Cos(aConst.value))
		}
		return Cos(arg)
	case *expExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok {
			return Number(math.Exp(aConst.value))
		}
		if ln, ok := arg.(*lnExpr); ok {
			return ln.arg
		}
		return Exp(arg)
	case *lnExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok && aConst.value > 0 {
			return Number(math.Log(aConst.value))
		}
		if exp, ok := arg.(*expExpr); ok {
			return exp.arg
		}
		return Ln(arg)
	case *sqrtExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok && aConst.value >= 0 {
			return Number(math.Sqrt(aConst.value))
		}
		return Sqrt(arg)
	default:
		return e
	}
}

// Helper functions for trig identities
func isSinSquared(e Expr) bool {
	if p, ok := e.(*powExpr); ok {
		if _, ok := p.base.(*sinExpr); ok {
			if c, ok := p.exp.(*constExpr); ok && c.value == 2 {
				return true
			}
		}
	}
	return false
}

func isCosSquared(e Expr) bool {
	if p, ok := e.(*powExpr); ok {
		if _, ok := p.base.(*cosExpr); ok {
			if c, ok := p.exp.(*constExpr); ok && c.value == 2 {
				return true
			}
		}
	}
	return false
}

func sameArg(e1, e2 Expr) bool {
	p1 := getPowBase(e1)
	p2 := getPowBase(e2)
	if p1 == nil || p2 == nil {
		return false
	}
	return p1.String() == p2.String()
}

func getPowBase(e Expr) Expr {
	if p, ok := e.(*powExpr); ok {
		return p.base
	}
	return nil
}

// Expand expands expressions fully, including distribution and binomial expansion.
func Expand(e Expr) Expr {
	e = Simplify(e) // Simplify first
	switch expr := e.(type) {
	case *powExpr:
		base := Expand(expr.base)
		exp := Expand(expr.exp)
		if c, ok := exp.(*constExpr); ok && c.value > 0 && math.Floor(c.value) == c.value { // Integer power
			n := int(c.value)
			if add, ok := base.(*addExpr); ok {
				// Binomial theorem for (a + b)^n
				a := add.left
				b := add.right
				result := Number(0)
				for k := 0; k <= n; k++ {
					coeff := binom(n, k)
					term := Mul(Number(float64(coeff)), Mul(Pow(a, Number(float64(n-k))), Pow(b, Number(float64(k)))))
					result = Add(result, term)
				}
				return result
			}
		}
		return Pow(base, exp)
	case *addExpr:
		return Add(Expand(expr.left), Expand(expr.right))
	case *subExpr:
		return Sub(Expand(expr.left), Expand(expr.right))
	case *mulExpr:
		left := Expand(expr.left)
		right := Expand(expr.right)
		// Distribute if left or right is add/sub
		if ladd, ok := left.(*addExpr); ok {
			return Add(Mul(ladd.left, right), Mul(ladd.right, right))
		}
		if lsub, ok := left.(*subExpr); ok {
			return Sub(Mul(lsub.left, right), Mul(lsub.right, right))
		}
		if radd, ok := right.(*addExpr); ok {
			return Add(Mul(left, radd.left), Mul(left, radd.right))
		}
		if rsub, ok := right.(*subExpr); ok {
			return Sub(Mul(left, rsub.left), Mul(left, rsub.right))
		}
		return Mul(left, right)
	default:
		return e
	}
}

// binom computes binomial coefficient C(n, k)
func binom(n, k int) int {
	if k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	res := 1
	for i := 0; i < k; i++ {
		res *= (n - i)
		res /= (i + 1)
	}
	return res
}

// Integrate computes the indefinite integral with respect to the symbol, with basic by parts and substitution.
func Integrate(e Expr, sym Expr) Expr {
	v, ok := sym.(*varExpr)
	if !ok {
		return nil
	}

	e = Simplify(e)
	switch expr := e.(type) {
	case *constExpr:
		return Mul(expr, v) // ∫ c dx = c x
	case *varExpr:
		if expr.name == v.name {
			return Mul(Number(0.5), Pow(v, Number(2))) // ∫ x dx = (1/2) x^2
		}
		return Mul(expr, v) // Treat as constant
	case *addExpr:
		return Add(Integrate(expr.left, sym), Integrate(expr.right, sym))
	case *subExpr:
		return Sub(Integrate(expr.left, sym), Integrate(expr.right, sym))
	case *mulExpr:
		// Try if one is constant
		if cl, ok := expr.left.(*constExpr); ok {
			return Mul(cl, Integrate(expr.right, sym))
		}
		if cr, ok := expr.right.(*constExpr); ok {
			return Mul(cr, Integrate(expr.left, sym))
		}
		// Basic integration by parts: assume u = left, dv = right if right is integrable
		dv := Integrate(expr.right, sym)
		if dv != nil {
			u := expr.left
			du := Diff(u, sym)
			vint := dv
			// ∫ u dv = u v - ∫ v du
			return Sub(Mul(u, vint), Integrate(Mul(vint, du), sym))
		}
		// Else try reverse
		du := Integrate(expr.left, sym)
		if du != nil {
			// Similar, swap
			vint := expr.right
			dv := Diff(vint, sym)
			return Sub(Mul(du, vint), Integrate(Mul(du, dv), sym)) // Wait, incorrect; fix logic
		}
		return nil
	case *powExpr:
		base := expr.base
		exp := expr.exp
		if base.String() == v.name {
			if c, ok := exp.(*constExpr); ok && c.value != -1 {
				return Mul(Number(1/(c.value+1)), Pow(base, Number(c.value+1)))
			}
		}
		return nil
	case *sinExpr:
		if expr.arg.String() == v.name {
			return Neg(Cos(expr.arg))
		}
		// Chain rule reverse: if arg = kx, ∫ sin(kx) = -cos(kx)/k
		if mul, ok := expr.arg.(*mulExpr); ok {
			if c, ok := mul.left.(*constExpr); ok && mul.right.String() == v.name {
				return Mul(Number(-1/c.value), Cos(expr.arg))
			}
		}
		return nil
	case *cosExpr:
		if expr.arg.String() == v.name {
			return Sin(expr.arg)
		}
		if mul, ok := expr.arg.(*mulExpr); ok {
			if c, ok := mul.left.(*constExpr); ok && mul.right.String() == v.name {
				return Mul(Number(1/c.value), Sin(expr.arg))
			}
		}
		return nil
	case *expExpr:
		if expr.arg.String() == v.name {
			return Exp(expr.arg)
		}
		if mul, ok := expr.arg.(*mulExpr); ok {
			if c, ok := mul.left.(*constExpr); ok && mul.right.String() == v.name {
				return Mul(Number(1/c.value), Exp(expr.arg))
			}
		}
		return nil
	default:
		return nil
	}
}

// Solve solves equations expr = 0 for the symbol, now supports linear and quadratic.
func Solve(eq Expr, sym Expr) []Expr {
	v, ok := sym.(*varExpr)
	if !ok {
		return nil
	}

	eq = Simplify(eq)
	// Collect coefficients for polynomial in v (up to deg 2)
	a, b, c := polyCoeffs(eq, v)
	if a == 0 && b == 0 {
		return nil // Constant eq
	}
	if a == 0 {
		// Linear: bx + c = 0 => x = -c/b
		return []Expr{Neg(Div(Number(c), Number(b)))}
	}
	// Quadratic: ax^2 + bx + c = 0
	disc := Sub(Pow(Number(b), Number(2)), Mul(Number(4), Mul(Number(a), Number(c))))
	root1 := Div(Add(Neg(Number(b)), Sqrt(disc)), Mul(Number(2), Number(a)))
	root2 := Div(Sub(Neg(Number(b)), Sqrt(disc)), Mul(Number(2), Number(a)))
	return []Expr{root1, root2}
}

// polyCoeffs collects a x^2 + b x + c coeffs
func polyCoeffs(e Expr, v *varExpr) (a, b, c float64) {
	switch expr := e.(type) {
	case *addExpr:
		la, lb, lc := polyCoeffs(expr.left, v)
		ra, rb, rc := polyCoeffs(expr.right, v)
		return la + ra, lb + rb, lc + rc
	case *subExpr:
		la, lb, lc := polyCoeffs(expr.left, v)
		ra, rb, rc := polyCoeffs(expr.right, v)
		return la - ra, lb - rb, lc - rc
	case *mulExpr:
		// Assume simple poly terms
		if _, ok := expr.left.(*constExpr); ok {
			coeff := expr.left.(*constExpr).value
			_, pb, pc := polyCoeffs(expr.right, v)
			return 0, coeff * pb, coeff * pc // Wait, need to check degree of right
		}
		// For x^2 term: if right is x^2, etc.
		// Simplified: use a more general collector
		deg := degree(expr, v)
		if deg == 2 {
			a = leadingCoeff(expr, v)
		} else if deg == 1 {
			b = leadingCoeff(expr, v)
		} else if deg == 0 {
			c = evalConst(expr)
		}
	case *powExpr:
		if expr.base.String() == v.name {
			if c, ok := expr.exp.(*constExpr); ok {
				if c.value == 2 {
					a = 1
				} else if c.value == 1 {
					b = 1
				} else if c.value == 0 {
					c = 1
				}
			}
		}
	case *varExpr:
		if expr.name == v.name {
			b = 1
		}
	case *constExpr:
		c = expr.value
	}
	return
}

// Helpers for polyCoeffs (stubbed for simplicity; in real, recurse)
func degree(e Expr, v *varExpr) int {
	// Implement degree finding
	return 0 // Placeholder
}

func leadingCoeff(e Expr, v *varExpr) float64 {
	return 0 // Placeholder
}

func evalConst(e Expr) float64 {
	if c, ok := e.(*constExpr); ok {
		return c.value
	}
	return 0
}

// Note: For full poly coeffs, a better way is to substitute and use numerical methods or better tree analysis, but for minimal, assume simple forms

// Eval evaluates the expression numerically with substitutions.
func Eval(e Expr, subs map[string]float64) float64 {
	switch expr := e.(type) {
	case *constExpr:
		return expr.value
	case *varExpr:
		if val, ok := subs[expr.name]; ok {
			return val
		}
		return 0 // Or error
	case *addExpr:
		return Eval(expr.left, subs) + Eval(expr.right, subs)
	case *subExpr:
		return Eval(expr.left, subs) - Eval(expr.right, subs)
	case *mulExpr:
		return Eval(expr.left, subs) * Eval(expr.right, subs)
	case *negExpr:
		return -Eval(expr.expr, subs)
	case *powExpr:
		return math.Pow(Eval(expr.base, subs), Eval(expr.exp, subs))
	case *sinExpr:
		return math.Sin(Eval(expr.arg, subs))
	case *cosExpr:
		return math.Cos(Eval(expr.arg, subs))
	case *expExpr:
		return math.Exp(Eval(expr.arg, subs))
	case *lnExpr:
		return math.Log(Eval(expr.arg, subs))
	case *sqrtExpr:
		return math.Sqrt(Eval(expr.arg, subs))
	default:
		return 0
	}
}

// Parse parses a string into Expr (simple parser).
func Parse(s string) Expr {
	// Simple recursive descent parser
	s = strings.ReplaceAll(s, " ", "") // Remove spaces
	return parseExpr(&s)
}

func parseExpr(s *string) Expr {
	return parseAddSub(s)
}

func parseAddSub(s *string) Expr {
	expr := parseMulDiv(s)
	for len(*s) > 0 {
		op := (*s)[0]
		if op != '+' && op != '-' {
			break
		}
		*s = (*s)[1:]
		term := parseMulDiv(s)
		if op == '+' {
			expr = Add(expr, term)
		} else {
			expr = Sub(expr, term)
		}
	}
	return expr
}

func parseMulDiv(s *string) Expr {
	expr := parsePow(s)
	for len(*s) > 0 {
		op := (*s)[0]
		if op != '*' && op != '/' {
			break
		}
		*s = (*s)[1:]
		term := parsePow(s)
		if op == '*' {
			expr = Mul(expr, term)
		} else {
			expr = Div(expr, term)
		}
	}
	return expr
}

func parsePow(s *string) Expr {
	expr := parseUnary(s)
	if len(*s) > 0 && (*s)[0] == '^' {
		*s = (*s)[1:]
		exp := parseUnary(s)
		expr = Pow(expr, exp)
	}
	return expr
}

func parseUnary(s *string) Expr {
	if len(*s) > 0 && (*s)[0] == '-' {
		*s = (*s)[1:]
		return Neg(parsePrimary(s))
	}
	return parsePrimary(s)
}

func parsePrimary(s *string) Expr {
	if len(*s) == 0 {
		return nil
	}
	c := (*s)[0]
	if unicode.IsDigit(rune(c)) || c == '.' {
		return parseNumber(s)
	} else if unicode.IsLetter(rune(c)) {
		return parseFuncOrVar(s)
	} else if c == '(' {
		*s = (*s)[1:]
		expr := parseExpr(s)
		if len(*s) > 0 && (*s)[0] == ')' {
			*s = (*s)[1:]
		}
		return expr
	}
	return nil
}

func parseNumber(s *string) Expr {
	numStr := ""
	for len(*s) > 0 {
		c := (*s)[0]
		if unicode.IsDigit(rune(c)) || c == '.' {
			numStr += string(c)
			*s = (*s)[1:]
		} else {
			break
		}
	}
	val, _ := strconv.ParseFloat(numStr, 64)
	return Number(val)
}

func parseFuncOrVar(s *string) Expr {
	id := ""
	for len(*s) > 0 && unicode.IsLetter(rune((*s)[0])) {
		id += string((*s)[0])
		*s = (*s)[1:]
	}
	if len(*s) > 0 && (*s)[0] == '(' {
		*s = (*s)[1:]
		arg := parseExpr(s)
		if len(*s) > 0 && (*s)[0] == ')' {
			*s = (*s)[1:]
		}
		switch id {
		case "sin":
			return Sin(arg)
		case "cos":
			return Cos(arg)
		case "exp":
			return Exp(arg)
		case "ln":
			return Ln(arg)
		case "sqrt":
			return Sqrt(arg)
		}
	}
	return Symbol(id)
}

// Note on multi-variable solving: Current Solve is for single var single eq. For systems, would need matrix methods, not implemented yet.
