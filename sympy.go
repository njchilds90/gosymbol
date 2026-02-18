package sympy

import (
	"fmt"
	"math"
)

// Expr is the interface for all symbolic expressions.
type Expr interface {
	String() string
}

// constExpr represents a constant value.
type constExpr struct {
	value float64
}

func (c *constExpr) String() string {
	return fmt.Sprintf("%v", c.value)
}

// varExpr represents a symbolic variable.
type varExpr struct {
	name string
}

func (v *varExpr) String() string {
	return v.name
}

// addExpr represents an addition operation.
type addExpr struct {
	left, right Expr
}

func (a *addExpr) String() string {
	return fmt.Sprintf("( %s + %s )", a.left.String(), a.right.String())
}

// mulExpr represents a multiplication operation.
type mulExpr struct {
	left, right Expr
}

func (m *mulExpr) String() string {
	return fmt.Sprintf("( %s * %s )", m.left.String(), m.right.String())
}

// subExpr represents subtraction: left - right
type subExpr struct {
	left, right Expr
}

func (s *subExpr) String() string {
	return fmt.Sprintf("( %s - %s )", s.left.String(), s.right.String())
}

// negExpr for unary minus
type negExpr struct {
	expr Expr
}

func (n *negExpr) String() string {
	return fmt.Sprintf("(-%s)", n.expr.String())
}

// powExpr represents exponentiation: base^exp
type powExpr struct {
	base, exp Expr
}

func (p *powExpr) String() string {
	return fmt.Sprintf("( %s ^ %s )", p.base.String(), p.exp.String())
}

// sinExpr represents sin function
type sinExpr struct {
	arg Expr
}

func (s *sinExpr) String() string {
	return fmt.Sprintf("sin(%s)", s.arg.String())
}

// cosExpr represents cos function
type cosExpr struct {
	arg Expr
}

func (c *cosExpr) String() string {
	return fmt.Sprintf("cos(%s)", c.arg.String())
}

// expExpr represents exp function (e^x)
type expExpr struct {
	arg Expr
}

func (e *expExpr) String() string {
	return fmt.Sprintf("exp(%s)", e.arg.String())
}

// lnExpr represents natural log function
type lnExpr struct {
	arg Expr
}

func (l *lnExpr) String() string {
	return fmt.Sprintf("ln(%s)", l.arg.String())
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
	default:
		return nil // Unsupported type
	}
}

// Div creates a division expression (left / right). Added for completeness in advanced features.
func Div(left, right Expr) Expr {
	return Mul(left, Pow(right, Number(-1)))
}

// Simplify performs enhanced simplification, including constant folding, identities, and basic trig identities.
func Simplify(e Expr) Expr {
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
		// Check for sin^2 + cos^2 = 1
		if isSinSquared(left) && isCosSquared(right) && sameArg(left, right) {
			return Number(1)
		}
		if isSinSquared(right) && isCosSquared(left) && sameArg(left, right) {
			return Number(1)
		}
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
		// Simplify exp(ln(x)) -> x
		if ln, ok := arg.(*lnExpr); ok {
			return ln.arg
		}
		return Exp(arg)
	case *lnExpr:
		arg := Simplify(expr.arg)
		if aConst, ok := arg.(*constExpr); ok && aConst.value > 0 {
			return Number(math.Log(aConst.value))
		}
		// Simplify ln(exp(x)) -> x
		if exp, ok := arg.(*expExpr); ok {
			return exp.arg
		}
		return Ln(arg)
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
	p1, _ := e1.(*powExpr)
	p2, _ := e2.(*powExpr)
	s1, _ := p1.base.(*sinExpr)
	s2, _ := p2.base.(*sinExpr)
	c1, _ := p1.base.(*cosExpr)
	c2, _ := p2.base.(*cosExpr)
	if s1 != nil && c2 != nil {
		return s1.arg.String() == c2.arg.String()
	}
	if c1 != nil && s2 != nil {
		return c1.arg.String() == s2.arg.String()
	}
	return false
}

// Expand expands expressions, e.g., (x + y)^n for integer n.
func Expand(e Expr) Expr {
	switch expr := e.(type) {
	case *powExpr:
		base := expr.base
		exp := expr.exp
		if add, ok := base.(*addExpr); ok {
			if c, ok := exp.(*constExpr); ok && c.value == 2 { // Simple case: (a + b)^2 = a^2 + 2ab + b^2
				a := add.left
				b := add.right
				return Add(Add(Pow(a, Number(2)), Mul(Number(2), Mul(a, b))), Pow(b, Number(2)))
			}
			// Add more binomial coefficients for higher powers if needed
		}
		return e
	// Recurse on other types
	case *addExpr:
		return Add(Expand(expr.left), Expand(expr.right))
	case *subExpr:
		return Sub(Expand(expr.left), Expand(expr.right))
	case *mulExpr:
		return Mul(Expand(expr.left), Expand(expr.right))
	default:
		return e
	}
}

// Integrate computes the indefinite integral with respect to the symbol.
func Integrate(e Expr, sym Expr) Expr {
	v, ok := sym.(*varExpr)
	if !ok {
		return nil
	}

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
		// Simple case: if one is constant
		if _, ok := expr.left.(*constExpr); ok {
			return Mul(expr.left, Integrate(expr.right, sym))
		}
		if _, ok := expr.right.(*constExpr); ok {
			return Mul(expr.right, Integrate(expr.left, sym))
		}
		return nil // Integration by parts or other methods too complex for now
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
		return nil
	case *cosExpr:
		if expr.arg.String() == v.name {
			return Sin(expr.arg)
		}
		return nil
	case *expExpr:
		if expr.arg.String() == v.name {
			return Exp(expr.arg)
		}
		return nil
	default:
		return nil
	}
}

// Solve solves simple equations of the form expr = 0 for the symbol.
// Currently supports linear equations ax + b = 0 -> x = -b/a
func Solve(eq Expr, sym Expr) Expr {
	v, ok := sym.(*varExpr)
	if !ok {
		return nil
	}

	// Assume eq is ax + b, where a and b are constants w.r.t. v
	// Collect terms
	a := coeff(eq, v)
	b := constantTerm(eq, v)
	if a == 0 {
		return nil // Not linear
	}
	return Number(-b / a)
}

// Helper: extract coefficient of v in expr
func coeff(e Expr, v *varExpr) float64 {
	switch expr := e.(type) {
	case *varExpr:
		if expr.name == v.name {
			return 1
		}
		return 0
	case *addExpr:
		return coeff(expr.left, v) + coeff(expr.right, v)
	case *subExpr:
		return coeff(expr.left, v) - coeff(expr.right, v)
	case *mulExpr:
		if _, ok := expr.left.(*constExpr); ok {
			return expr.left.(*constExpr).value * coeff(expr.right, v)
		}
		if _, ok := expr.right.(*constExpr); ok {
			return expr.right.(*constExpr).value * coeff(expr.left, v)
		}
		return 0 // Non-linear
	case *constExpr:
		return 0
	default:
		return 0
	}
}

// Helper: extract constant term (independent of v)
func constantTerm(e Expr, v *varExpr) float64 {
	switch expr := e.(type) {
	case *varExpr:
		return 0
	case *addExpr:
		return constantTerm(expr.left, v) + constantTerm(expr.right, v)
	case *subExpr:
		return constantTerm(expr.left, v) - constantTerm(expr.right, v)
	case *mulExpr:
		// Only if both sides constant
		lConst := constantTerm(expr.left, v)
		rConst := constantTerm(expr.right, v)
		if lConst != 0 && rConst != 0 {
			return lConst * rConst
		}
		return 0
	case *constExpr:
		return expr.value
	default:
		return 0
	}
}
