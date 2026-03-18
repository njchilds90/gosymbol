// Package gosymbol provides a deterministic symbolic math kernel for Go.
//
// Design goals:
//   - Zero external dependencies
//   - Exact rational arithmetic (math/big.Rat)
//   - Deterministic simplification and stable output
//   - AI/LLM friendly: JSON, LaTeX, infix parsing, and MCP-ready APIs
//   - Embeddable in Go services, CLI tools, and agent backends
package gosymbol

import (
	"fmt"
	"strings"
)

// Expr is the common interface implemented by every symbolic expression node.
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
		if len(newFactors) == 3 {
			if leadingCoefficient, ok := newFactors[0].(*Num); ok && leadingCoefficient.Equal(N(2)) {
				if sineFunction, ok := newFactors[1].(*Func); ok && sineFunction.name == "sin" {
					if cosineFunction, ok := newFactors[2].(*Func); ok && cosineFunction.name == "cos" && sineFunction.arg.Equal(cosineFunction.arg) {
						return SinOf(MulOf(N(2), sineFunction.arg))
					}
				}
			}
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
					if fn.name == "sin" || fn.name == "cos" || fn.name == "sinh" || fn.name == "cosh" {
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
				if (ti.funcName == "sinh" && tj.funcName == "cosh") || (ti.funcName == "cosh" && tj.funcName == "sinh") {
					continue
				}
				newTerms := []Expr{}
				for idx, t := range add.terms {
					if idx != ti.idx && idx != tj.idx {
						newTerms = append(newTerms, t)
					}
				}
				newTerms = append(newTerms, ti.coeff)
				return AddOf(newTerms...).Simplify()
			}
			if ti.argStr == tj.argStr && numCmp(ti.coeff, tj.coeff) == 0 {
				if (ti.funcName == "cosh" && tj.funcName == "sinh") || (ti.funcName == "sinh" && tj.funcName == "cosh") {
					newTerms := []Expr{}
					for idx, t := range add.terms {
						if idx != ti.idx && idx != tj.idx {
							newTerms = append(newTerms, t)
						}
					}
					if ti.funcName == "cosh" {
						newTerms = append(newTerms, ti.coeff)
					} else {
						newTerms = append(newTerms, numNeg(ti.coeff))
					}
					return AddOf(newTerms...).Simplify()
				}
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
