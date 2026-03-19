package gosymbol

// ComplexNumberNode represents a symbolic complex value with real and imaginary parts.
type ComplexNumberNode struct {
	realPart      Expr
	imaginaryPart Expr
}

// CreateComplexNumber constructs a symbolic complex number.
func CreateComplexNumber(realPart Expr, imaginaryPart Expr) *ComplexNumberNode {
	return &ComplexNumberNode{realPart: realPart.Simplify(), imaginaryPart: imaginaryPart.Simplify()}
}

// CreateImaginaryUnit returns the symbolic imaginary unit.
func CreateImaginaryUnit() *ComplexNumberNode {
	return CreateComplexNumber(N(0), N(1))
}

func (complexNumber *ComplexNumberNode) Simplify() Expr {
	if simplificationDepthExceeded(complexNumber) {
		return complexNumber
	}
	if imaginaryNumber, ok := complexNumber.imaginaryPart.(*Num); ok && imaginaryNumber.IsZero() {
		return complexNumber.realPart.Simplify()
	}
	return &ComplexNumberNode{
		realPart:      complexNumber.realPart.Simplify(),
		imaginaryPart: complexNumber.imaginaryPart.Simplify(),
	}
}

func (complexNumber *ComplexNumberNode) Canonicalize() Expr {
	return Canonicalize(complexNumber)
}

func (complexNumber *ComplexNumberNode) String() string {
	if imaginaryNumber, ok := complexNumber.imaginaryPart.(*Num); ok {
		if imaginaryNumber.IsZero() {
			return complexNumber.realPart.String()
		}
		if imaginaryNumber.IsOne() {
			return complexNumber.realPart.String() + " + i"
		}
		if imaginaryNumber.IsNegOne() {
			return complexNumber.realPart.String() + " + -i"
		}
	}
	return complexNumber.realPart.String() + " + " + complexNumber.imaginaryPart.String() + "*i"
}

func (complexNumber *ComplexNumberNode) LaTeX() string {
	if imaginaryNumber, ok := complexNumber.imaginaryPart.(*Num); ok {
		if imaginaryNumber.IsZero() {
			return complexNumber.realPart.LaTeX()
		}
		if imaginaryNumber.IsOne() {
			return complexNumber.realPart.LaTeX() + " + i"
		}
		if imaginaryNumber.IsNegOne() {
			return complexNumber.realPart.LaTeX() + " - i"
		}
	}
	return complexNumber.realPart.LaTeX() + " + " + complexNumber.imaginaryPart.LaTeX() + " i"
}

func (complexNumber *ComplexNumberNode) Sub(variableName string, value Expr) Expr {
	return CreateComplexNumber(complexNumber.realPart.Sub(variableName, value), complexNumber.imaginaryPart.Sub(variableName, value)).Simplify()
}

func (complexNumber *ComplexNumberNode) Diff(variableName string) Expr {
	return CreateComplexNumber(complexNumber.realPart.Diff(variableName), complexNumber.imaginaryPart.Diff(variableName)).Simplify()
}

func (complexNumber *ComplexNumberNode) Eval() (*Num, bool) {
	if imaginaryNumber, ok := complexNumber.imaginaryPart.Eval(); ok && imaginaryNumber.IsZero() {
		return complexNumber.realPart.Eval()
	}
	return nil, false
}

func (complexNumber *ComplexNumberNode) Equal(other Expr) bool {
	otherComplexNumber, isComplexNumber := other.(*ComplexNumberNode)
	return isComplexNumber &&
		complexNumber.realPart.Equal(otherComplexNumber.realPart) &&
		complexNumber.imaginaryPart.Equal(otherComplexNumber.imaginaryPart)
}

func (complexNumber *ComplexNumberNode) exprType() string { return "complex" }

func (complexNumber *ComplexNumberNode) toJSON() map[string]interface{} {
	return map[string]interface{}{
		"type":      "complex",
		"real":      complexNumber.realPart.toJSON(),
		"imaginary": complexNumber.imaginaryPart.toJSON(),
	}
}

// RealPart returns the symbolic real component.
func (complexNumber *ComplexNumberNode) RealPart() Expr { return complexNumber.realPart }

// ImaginaryPart returns the symbolic imaginary component.
func (complexNumber *ComplexNumberNode) ImaginaryPart() Expr { return complexNumber.imaginaryPart }

// AddComplexNumbers adds two symbolic complex numbers.
func AddComplexNumbers(leftComplexNumber, rightComplexNumber *ComplexNumberNode) *ComplexNumberNode {
	return CreateComplexNumber(
		AddOf(leftComplexNumber.realPart, rightComplexNumber.realPart),
		AddOf(leftComplexNumber.imaginaryPart, rightComplexNumber.imaginaryPart),
	)
}

// MultiplyComplexNumbers multiplies two symbolic complex numbers.
func MultiplyComplexNumbers(leftComplexNumber, rightComplexNumber *ComplexNumberNode) *ComplexNumberNode {
	return CreateComplexNumber(
		AddOf(
			MulOf(leftComplexNumber.realPart, rightComplexNumber.realPart),
			MulOf(N(-1), leftComplexNumber.imaginaryPart, rightComplexNumber.imaginaryPart),
		),
		AddOf(
			MulOf(leftComplexNumber.realPart, rightComplexNumber.imaginaryPart),
			MulOf(leftComplexNumber.imaginaryPart, rightComplexNumber.realPart),
		),
	)
}
