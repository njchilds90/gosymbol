package gosymbol

import (
	"fmt"
	"strings"
)

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

// SymbolicMatrix is the full-name public alias for Matrix.
type SymbolicMatrix = Matrix

// CreateSymbolicMatrix creates a zero-filled symbolic matrix.
func CreateSymbolicMatrix(rowCount, columnCount int) *SymbolicMatrix {
	return NewMatrix(rowCount, columnCount)
}

// CreateSymbolicMatrixFromEntries creates a symbolic matrix from row-major entries.
func CreateSymbolicMatrixFromEntries(rowCount, columnCount int, entries []Expr) *SymbolicMatrix {
	return MatrixFromSlice(rowCount, columnCount, entries)
}

// CreateIdentitySymbolicMatrix creates an identity matrix.
func CreateIdentitySymbolicMatrix(size int) *SymbolicMatrix {
	return Identity(size)
}
