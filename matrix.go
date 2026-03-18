package gosymbol

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
