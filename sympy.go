package sympy

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

type Expr interface {
	String() string
	LaTeX() string
	Eval(map[string]float64) float64
	Diff(Expr) Expr
	Simplify() Expr
}

type constExpr struct{ value float64 }
func (c *constExpr) String() string { return fmt.Sprintf("%g", c.value) }
func (c *constExpr) LaTeX() string  { return fmt.Sprintf("%.10g", c.value) }
func (c *constExpr) Eval(_ map[string]float64) float64 { return c.value }
func (c *constExpr) Diff(_ Expr) Expr { return Number(0) }
func (c *constExpr) Simplify() Expr { return c }

type varExpr struct{ name string }
func (v *varExpr) String() string { return v.name }
func (v *varExpr) LaTeX() string  { return v.name }
func (v *varExpr) Eval(s map[string]float64) float64 { if val, ok := s[v.name]; ok { return val }; return math.NaN() }
func (v *varExpr) Diff(sym Expr) Expr { if vv, ok := sym.(*varExpr); ok && vv.name == v.name { return Number(1) }; return Number(0) }
func (v *varExpr) Simplify() Expr { return v }

type binOp struct { op string; l, r Expr; prec int }
func (b *binOp) String() string {
	ls, rs := b.l.String(), b.r.String()
	if lb, ok := b.l.(*binOp); ok && lb.prec < b.prec { ls = "(" + ls + ")" }
	if rb, ok := b.r.(*binOp); ok && rb.prec <= b.prec { rs = "(" + rs + ")" }
	return ls + b.op + rs
}
func (b *binOp) LaTeX() string {
	ls, rs := b.l.LaTeX(), b.r.LaTeX()
	switch b.op {
	case "+": return ls + " + " + rs
	case "-": return ls + " - " + rs
	case "*": return ls + "\\cdot " + rs
	case "/": return "\\frac{" + ls + "}{" + rs + "}"
	case "^": return ls + "^{" + rs + "}"
	}
	return b.String()
}
func (b *binOp) Eval(s map[string]float64) float64 {
	lv, rv := b.l.Eval(s), b.r.Eval(s)
	switch b.op {
	case "+": return lv + rv
	case "-": return lv - rv
	case "*": return lv * rv
	case "/": return lv / rv
	case "^": return math.Pow(lv, rv)
	}
	return math.NaN()
}
func (b *binOp) Diff(sym Expr) Expr {
	ld, rd := b.l.Diff(sym), b.r.Diff(sym)
	switch b.op {
	case "+": return Add(ld, rd)
	case "-": return Sub(ld, rd)
	case "*": return Add(Mul(ld, b.r), Mul(b.l, rd))
	case "^": return Mul(b, Add(Mul(rd, Ln(b.l)), Mul(b.r, Div(ld, b.l))))
	}
	return Number(0)
}
func (b *binOp) Simplify() Expr {
	l, r := b.l.Simplify(), b.r.Simplify()
	if lc, ok := l.(*constExpr); ok {
		if rc, ok := r.(*constExpr); ok {
			switch b.op {
			case "+": return Number(lc.value + rc.value)
			case "-": return Number(lc.value - rc.value)
			case "*": return Number(lc.value * rc.value)
			case "/": return Number(lc.value / rc.value)
			case "^": return Number(math.Pow(lc.value, rc.value))
			}
		}
		if lc.value == 0 && b.op == "+" { return r }
		if lc.value == 0 && b.op == "-" { return Neg(r) }
		if lc.value == 1 && b.op == "*" { return r }
		if lc.value == 0 && b.op == "*" { return Number(0) }
	}
	if rc, ok := r.(*constExpr); ok {
		if rc.value == 0 && b.op == "+" { return l }
		if rc.value == 1 && b.op == "*" { return l }
		if rc.value == 0 && b.op == "*" { return Number(0) }
	}
	if b.op == "+" && isTrigId(l, r) { return Number(1) }
	if b.op == "-" && l.String() == r.String() { return Number(0) }
	if b.op == "/" && l.String() == r.String() { return Number(1) }
	if b.op == "*" && isZero(l) || isZero(r) { return Number(0) }
	return &binOp{b.op, l, r, b.prec}
}

type unary struct { op string; e Expr }
func (u *unary) String() string { return u.op + u.e.String() }
func (u *unary) LaTeX() string  { return "-" + u.e.LaTeX() }
func (u *unary) Eval(s map[string]float64) float64 { return -u.e.Eval(s) }
func (u *unary) Diff(sym Expr) Expr { return Neg(u.e.Diff(sym)) }
func (u *unary) Simplify() Expr { return Neg(u.e.Simplify()) }

type fexpr struct { name string; arg Expr }
func (f *fexpr) String() string { return f.name + "(" + f.arg.String() + ")" }
func (f *fexpr) LaTeX() string {
	argL := f.arg.LaTeX()
	switch f.name {
	case "sin": return "\\sin(" + argL + ")"
	case "cos": return "\\cos(" + argL + ")"
	case "exp": return "e^{" + argL + "}"
	case "ln": return "\\ln(" + argL + ")"
	case "sqrt": return "\\sqrt{" + argL + "}"
	}
	return f.String()
}
func (f *fexpr) Eval(s map[string]float64) float64 {
	a := f.arg.Eval(s)
	switch f.name {
	case "sin": return math.Sin(a)
	case "cos": return math.Cos(a)
	case "exp": return math.Exp(a)
	case "ln": return math.Log(a)
	case "sqrt": return math.Sqrt(a)
	}
	return math.NaN()
}
func (f *fexpr) Diff(sym Expr) Expr {
	d := f.arg.Diff(sym)
	switch f.name {
	case "sin": return Mul(Cos(f.arg), d)
	case "cos": return Neg(Mul(Sin(f.arg), d))
	case "exp": return Mul(Exp(f.arg), d)
	case "ln": return Div(d, f.arg)
	case "sqrt": return Mul(Number(0.5), Div(d, f))
	}
	return Number(0)
}
func (f *fexpr) Simplify() Expr {
	a := f.arg.Simplify()
	if c, ok := a.(*constExpr); ok {
		switch f.name {
		case "sin": return Number(math.Sin(c.value))
		case "cos": return Number(math.Cos(c.value))
		case "exp": return Number(math.Exp(c.value))
		case "ln": if c.value > 0 { return Number(math.Log(c.value)) }
		case "sqrt": if c.value >= 0 { return Number(math.Sqrt(c.value)) }
		}
	}
	if f.name == "exp" && isInverse(f.arg, "ln") { return f.arg.(*fexpr).arg }
	if f.name == "ln" && isInverse(f.arg, "exp") { return f.arg.(*fexpr).arg }
	return &fexpr{f.name, a}
}

func Number(v float64) Expr     { return &constExpr{v} }
func Symbol(n string) Expr      { return &varExpr{n} }
func Add(a, b Expr) Expr        { return &binOp{"+", a, b, 1} }
func Sub(a, b Expr) Expr        { return &binOp{"-", a, b, 1} }
func Mul(a, b Expr) Expr        { return &binOp{"*", a, b, 2} }
func Div(a, b Expr) Expr        { return Mul(a, Pow(b, Number(-1))) }
func Pow(a, b Expr) Expr        { return &binOp{"^", a, b, 3} }
func Neg(e Expr) Expr           { return &unary{"-", e} }
func Sin(e Expr) Expr           { return &fexpr{"sin", e} }
func Cos(e Expr) Expr           { return &fexpr{"cos", e} }
func Exp(e Expr) Expr           { return &fexpr{"exp", e} }
func Ln(e Expr) Expr            { return &fexpr{"ln", e} }
func Sqrt(e Expr) Expr          { return &fexpr{"sqrt", e} }

func isTrigId(a, b Expr) bool {
	return (isPow2(a, "sin") && isPow2(b, "cos")) || (isPow2(a, "cos") && isPow2(b, "sin"))
}
func isPow2(e Expr, fn string) bool {
	if p, ok := e.(*binOp); ok && p.op == "^" {
		if f, ok := p.l.(*fexpr); ok && f.name == fn {
			if c, ok := p.r.(*constExpr); ok && c.value == 2 { return true }
		}
	}
	return false
}
func isInverse(e Expr, name string) bool {
	f, ok := e.(*fexpr)
	return ok && f.name == name
}

func Expand(e Expr) Expr {
	e = e.Simplify()
	if p, ok := e.(*binOp); ok && p.op == "^" {
		if c, ok := p.r.(*constExpr); ok && c.value == float64(int(c.value)) && c.value > 0 {
			n := int(c.value)
			if add, ok := p.l.(*binOp); ok && add.op == "+" {
				res := Number(0)
				for k := 0; k <= n; k++ {
					coef := float64(binom(n, k))
					term := Mul(Number(coef), Mul(Pow(add.l, Number(float64(n-k))), Pow(add.r, Number(float64(k)))))
					res = Add(res, term)
				}
				return res
			}
		}
	}
	if m, ok := e.(*binOp); ok && m.op == "*" {
		if l, ok := m.l.(*binOp); ok && l.op == "+" { return Add(Mul(l.l, m.r), Mul(l.r, m.r)) }
		if r, ok := m.r.(*binOp); ok && r.op == "+" { return Add(Mul(m.l, r.l), Mul(m.l, r.r)) }
	}
	return e
}

func binom(n, k int) int {
	if k > n-k { k = n - k }
	res := 1
	for i := 0; i < k; i++ { res = res * (n - i) / (i + 1) }
	return res
}

func Integrate(e Expr, sym Expr) Expr {
	switch ex := e.(type) {
	case *constExpr: return Mul(e, sym)
	case *varExpr:
		if ex.name == sym.(*varExpr).name { return Mul(Number(0.5), Pow(sym, Number(2))) }
		return Mul(e, sym)
	case *binOp:
		if ex.op == "+" { return Add(Integrate(ex.l, sym), Integrate(ex.r, sym)) }
		if ex.op == "-" { return Sub(Integrate(ex.l, sym), Integrate(ex.r, sym)) }
		if ex.op == "*" {
			if c, ok := ex.l.(*constExpr); ok { return Mul(c, Integrate(ex.r, sym)) }
			if dv := Integrate(ex.r, sym); dv != nil {
				u := ex.l
				du := u.Diff(sym)
				return Sub(Mul(u, dv), Integrate(Mul(dv, du), sym))
			}
		}
	case *fexpr:
		switch ex.name {
		case "sin": return Neg(Cos(ex.arg))
		case "cos": return Sin(ex.arg)
		case "exp": return Exp(ex.arg)
		}
	}
	return nil
}

func IntegrateDefinite(e Expr, sym Expr, a, b float64) float64 {
	intg := Integrate(e, sym)
	if intg == nil { return math.NaN() }
	return intg.Eval(map[string]float64{sym.(*varExpr).name: b}) - intg.Eval(map[string]float64{sym.(*varExpr).name: a})
}

func Solve(e Expr, sym Expr) []Expr {
	v := sym.(*varExpr)
	degree := deg(e, v)
	if degree == 1 { return []Expr{ solveLinear(e, v) } }
	if degree == 2 { return solveQuadratic(e, v) }
	// higher degree stub: rational root theorem
	possibleRoots := []float64{1, -1, 2, -2, 3, -3, 0.5, -0.5} // expand for robustness
	for _, root := range possibleRoots {
		if e.Eval(map[string]float64{v.name: root}) == 0 {
			factor := Sub(sym, Number(root))
			quot := polyDiv(e, factor, v)
			return append([]Expr{Number(root)}, Solve(quot, sym)...)
		}
	}
	return nil
}

func solveLinear(e Expr, v *varExpr) Expr {
	a, b := linearCoeffs(e, v)
	return Div(Neg(Number(b)), Number(a))
}

func linearCoeffs(e Expr, v *varExpr) (a, b float64) {
	switch ex := e.(type) {
	case *binOp:
		if ex.op == "+" || ex.op == "-" {
			la, lb := linearCoeffs(ex.l, v)
			ra, rb := linearCoeffs(ex.r, v)
			if ex.op == "+" { return la + ra, lb + rb }
			return la - ra, lb - rb
		}
	case *varExpr: a = 1
	case *constExpr: b = ex.value
	}
	return
}

func solveQuadratic(e Expr, v *varExpr) []Expr {
	a, b, c := polyCoeffs(e, v)
	disc := Sub(Pow(Number(b), Number(2)), Mul(Number(4), Mul(Number(a), Number(c))))
	sqrtD := Sqrt(disc)
	twoA := Mul(Number(2), Number(a))
	return []Expr{
		Div(Add(Neg(Number(b)), sqrtD), twoA),
		Div(Sub(Neg(Number(b)), sqrtD), twoA),
	}
}

func polyCoeffs(e Expr, v *varExpr) (a, b, c float64) {
	switch ex := e.(type) {
	case *binOp:
		if ex.op == "+" || ex.op == "-" {
			la, lb, lc := polyCoeffs(ex.l, v)
			ra, rb, rc := polyCoeffs(ex.r, v)
			if ex.op == "+" { return la + ra, lb + rb, lc + rc }
			return la - ra, lb - rb, lc - rc
		}
		if ex.op == "^" {
			if vv, ok := ex.l.(*varExpr); ok && vv.name == v.name {
				if p, ok := ex.r.(*constExpr); ok {
					if p.value == 2 { a = 1 }
					if p.value == 1 { b = 1 }
					if p.value == 0 { c = 1 }
				}
			}
		}
	case *varExpr: b = 1
	case *constExpr: c = ex.value
	}
	return
}

func deg(e Expr, v *varExpr) int {
	switch ex := e.(type) {
	case *binOp:
		if ex.op == "+" || ex.op == "-" { return max(deg(ex.l, v), deg(ex.r, v)) }
		if ex.op == "*" { return deg(ex.l, v) + deg(ex.r, v) }
	case *varExpr: return 1
	}
	return 0
}

func max(a, b int) int { if a > b { return a }; return b }

func polyDiv(e, factor Expr, v *varExpr) Expr { return e } // stub

func Taylor(f, x, point Expr, n int) Expr {
	res := f.Simplify()
	fac := Number(1)
	df := f
	for k := 1; k < n; k++ {
		df = df.Diff(x).Simplify()
		fac = Mul(fac, Number(float64(k)))
		term := Mul(Div(df.Subst(x, point), fac), Pow(Sub(x, point), Number(float64(k))))
		res = Add(res, term)
	}
	return res
}

type Matrix [][]Expr
func (m Matrix) String() string {
	rows := []string{}
	for _, row := range m {
		els := []string{}
		for _, e := row { els = append(els, e.String()) }
		rows = append(rows, "["+strings.Join(els, ", ")+"]")
	}
	return "[" + strings.Join(rows, "; ") + "]"
}
func (m Matrix) LaTeX() string { return m.String() } // stub
func MatrixAdd(a, b Matrix) Matrix {
	res := make(Matrix, len(a))
	for i := range a {
		res[i] = make([]Expr, len(a[i]))
		for j := range a[i] { res[i][j] = Add(a[i][j], b[i][j]) }
	}
	return res
}
func MatrixMul(a, b Matrix) Matrix {
	res := make(Matrix, len(a))
	for i := range a {
		res[i] = make([]Expr, len(b[0]))
		for j := range b[0] {
			sum := Number(0)
			for k := range a[i] { sum = Add(sum, Mul(a[i][k], b[k][j])) }
			res[i][j] = sum
		}
	}
	return res
}

func Limit(f, x, a Expr) Expr { return f.Subst(x, a).Simplify() }

func Parse(s string) Expr {
	s = strings.ReplaceAll(s, " ", "")
	return parseExpr(&s)
}

func parseExpr(s *string) Expr { return parseAddSub(s) }

func parseAddSub(s *string) Expr {
	e := parseMul(s)
	for len(*s) > 0 {
		op := (*s)[0]
		if op != '+' && op != '-' { break }
		*s = (*s)[1:]
		t := parseMul(s)
		if op == '+' { e = Add(e, t) } else { e = Sub(e, t) }
	}
	return e
}

func parseMul(s *string) Expr {
	e := parsePow(s)
	for len(*s) > 0 {
		op := (*s)[0]
		if op != '*' && op != '/' { break }
		*s = (*s)[1:]
		t := parsePow(s)
		if op == '*' { e = Mul(e, t) } else { e = Div(e, t) }
	}
	return e
}

func parsePow(s *string) Expr {
	e := parseAtom(s)
	if len(*s) > 0 && (*s)[0] == '^' {
		*s = (*s)[1:]
		e = Pow(e, parseAtom(s))
	}
	return e
}

func parseAtom(s *string) Expr {
	if len(*s) == 0 { return nil }
	c := (*s)[0]
	if unicode.IsDigit(rune(c)) || c == '.' { return parseNum(s) }
	if unicode.IsLetter(rune(c)) { return parseId(s) }
	if c == '(' { *s = (*s)[1:]; e := parseExpr(s); if len(*s) > 0 && (*s)[0] == ')' { *s = (*s)[1:] }; return e }
	if c == '-' { *s = (*s)[1:]; return Neg(parseAtom(s)) }
	return nil
}

func parseNum(s *string) Expr {
	n := ""
	for len(*s) > 0 {
		c := (*s)[0]
		if unicode.IsDigit(rune(c)) || c == '.' { n += string(c); *s = (*s)[1:] } else { break }
	}
	v, _ := strconv.ParseFloat(n, 64)
	return Number(v)
}

func parseId(s *string) Expr {
	id := ""
	for len(*s) > 0 && unicode.IsLetter(rune((*s)[0])) { id += string((*s)[0]); *s = (*s)[1:] }
	if len(*s) > 0 && (*s)[0] == '(' {
		*s = (*s)[1:]
		arg := parseExpr(s)
		if len(*s) > 0 && (*s)[0] == ')' { *s = (*s)[1:] }
		switch id {
		case "sin": return Sin(arg)
		case "cos": return Cos(arg)
		case "exp": return Exp(arg)
		case "ln": return Ln(arg)
		case "sqrt": return Sqrt(arg)
		}
	}
	return Symbol(id)
}
