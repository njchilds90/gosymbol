package gosymbol

import (
	"context"
	"fmt"
	"math/big"
	"sort"
	"strings"
)

// ============================================================
// Canonicalization, Collect, Cancel
// ============================================================

// Canonicalize expands and deeply simplifies an expression.
func Canonicalize(e Expr) Expr { return DeepSimplify(Expand(e).Simplify()) }

// Collect groups terms by powers of varName.
func Collect(expr Expr, varName string) Expr {
	coeffs := PolyCoeffs(expr, varName)
	if len(coeffs) == 0 {
		return N(0)
	}
	degrees := make([]int, 0, len(coeffs))
	for d := range coeffs {
		degrees = append(degrees, d)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(degrees)))
	terms := make([]Expr, 0, len(degrees))
	for _, d := range degrees {
		c := coeffs[d]
		if cn, ok := c.(*Num); ok && cn.IsZero() {
			continue
		}
		switch d {
		case 0:
			terms = append(terms, c)
		case 1:
			terms = append(terms, MulOf(c, S(varName)))
		default:
			terms = append(terms, MulOf(c, PowOf(S(varName), N(int64(d)))))
		}
	}
	if len(terms) == 0 {
		return N(0)
	}
	return AddOf(terms...).Simplify()
}

// Cancel simplifies a rational expression num/denom.
func Cancel(num, denom Expr) Expr {
	num = num.Simplify()
	denom = denom.Simplify()
	if nn, ok := num.Eval(); ok {
		if dn, ok2 := denom.Eval(); ok2 {
			if dn.IsZero() {
				panic("gosymbol: Cancel: zero denominator")
			}
			return numDiv(nn, dn)
		}
	}
	if dn, ok := denom.(*Num); ok && dn.IsOne() {
		return num
	}
	if dn, ok := denom.(*Num); ok && dn.IsNegOne() {
		return MulOf(N(-1), num).Simplify()
	}
	numCoeff, numRest := extractCoefficient(num)
	denCoeff, denRest := extractCoefficient(denom)
	if numRest.Equal(denRest) {
		return numDiv(numCoeff, denCoeff)
	}
	return MulOf(num, PowOf(denom, N(-1))).Simplify()
}

// ============================================================
// Symbolic Factoring
// ============================================================

// FactorResult holds the result of a factoring attempt.
type FactorResult struct {
	Factors []Expr
	Success bool
}

// Factor attempts to factor a polynomial or rational expression in varName.
// It extracts numeric content, applies exact rational-root-based deflation for
// polynomial factors, and preserves irreducible remainder factors when it
// cannot factor further over the rationals.
func Factor(expr Expr, varName string) FactorResult {
	expr = expr.Simplify()
	if numeratorExpression, denominatorExpression, isQuotient := extractQuotient(expr); isQuotient {
		numeratorFactorResult := Factor(numeratorExpression, varName)
		denominatorFactorResult := Factor(denominatorExpression, varName)
		factors := append([]Expr{}, numeratorFactorResult.Factors...)
		for _, denominatorFactor := range denominatorFactorResult.Factors {
			factors = append(factors, PowOf(denominatorFactor, N(-1)).Simplify())
		}
		if len(factors) == 0 {
			factors = []Expr{expr}
		}
		return FactorResult{Factors: factors, Success: numeratorFactorResult.Success || denominatorFactorResult.Success}
	}
	expr = Collect(expr, varName).Simplify()

	numericCoefficients, isNumericPolynomial := polynomialNumericCoefficients(expr, varName)
	if !isNumericPolynomial {
		return factorNonNumericPolynomial(expr, varName)
	}
	degree := highestPolynomialDegree(numericCoefficients)
	if degree <= 1 {
		return FactorResult{Factors: []Expr{expr}, Success: false}
	}

	primitiveContent := extractPolynomialContent(numericCoefficients)
	primitiveCoefficients := dividePolynomialCoefficients(numericCoefficients, primitiveContent)
	remainingCoefficients := copyPolynomialCoefficients(primitiveCoefficients)
	factors := make([]Expr, 0, degree+1)
	if !primitiveContent.IsOne() {
		factors = append(factors, primitiveContent)
	}

	for {
		remainingDegree := highestPolynomialDegree(remainingCoefficients)
		if remainingDegree <= 2 {
			break
		}
		rationalRoot, found := findRationalPolynomialRoot(remainingCoefficients)
		if !found {
			break
		}
		factors = append(factors, linearFactorFromRoot(varName, rationalRoot))
		remainingCoefficients = dividePolynomialByLinearFactor(remainingCoefficients, rationalRoot)
	}

	remainingDegree := highestPolynomialDegree(remainingCoefficients)
	switch remainingDegree {
	case 0:
		if constantTerm := remainingCoefficients[0]; !constantTerm.IsOne() {
			factors = append(factors, constantTerm)
		}
	case 1:
		linearFactor, linearCoefficient := buildLinearPolynomialFactor(varName, remainingCoefficients)
		if !linearCoefficient.IsOne() {
			factors = append(factors, linearCoefficient)
		}
		factors = append(factors, linearFactor)
	case 2:
		quadraticFactors, solved := factorQuadraticPolynomial(varName, remainingCoefficients)
		if solved {
			factors = append(factors, quadraticFactors...)
		} else {
			quadraticFactor, quadraticCoefficient := buildPolynomialFactorFromCoefficients(varName, remainingCoefficients)
			if !quadraticCoefficient.IsOne() {
				factors = append(factors, quadraticCoefficient)
			}
			factors = append(factors, quadraticFactor)
		}
	default:
		irreducibleFactor, leadingCoefficient := buildPolynomialFactorFromCoefficients(varName, remainingCoefficients)
		if !leadingCoefficient.IsOne() {
			factors = append(factors, leadingCoefficient)
		}
		factors = append(factors, irreducibleFactor)
	}

	filteredFactors := make([]Expr, 0, len(factors))
	for _, factor := range factors {
		if numberFactor, isNumber := factor.(*Num); isNumber && numberFactor.IsOne() {
			continue
		}
		filteredFactors = append(filteredFactors, factor.Simplify())
	}
	if len(filteredFactors) == 0 {
		filteredFactors = []Expr{expr}
	}
	return FactorResult{Factors: filteredFactors, Success: len(filteredFactors) > 1 || !filteredFactors[0].Equal(expr)}
}

func factorNonNumericPolynomial(expr Expr, varName string) FactorResult {
	expr = Collect(expr, varName).Simplify()
	coefficients := PolyCoeffs(expr, varName)
	commonFactor := N(0)
	for _, coefficientExpression := range coefficients {
		coefficientNumber, isNumber := coefficientExpression.(*Num)
		if !isNumber || !coefficientNumber.IsInteger() {
			commonFactor = N(1)
			break
		}
		if commonFactor.IsZero() {
			commonFactor = numAbs(coefficientNumber)
			continue
		}
		commonFactor = N(gcdInt(commonFactor.val.Num().Int64(), numAbs(coefficientNumber).val.Num().Int64()))
	}
	if commonFactor.IsZero() {
		commonFactor = N(1)
	}
	if !commonFactor.IsOne() {
		return FactorResult{
			Factors: []Expr{commonFactor, MulOf(PowOf(commonFactor, N(-1)), expr).Simplify()},
			Success: true,
		}
	}
	return FactorResult{Factors: []Expr{expr}, Success: false}
}

func polynomialNumericCoefficients(expr Expr, varName string) (map[int]*Num, bool) {
	rawCoefficients := PolyCoeffs(expr, varName)
	if len(rawCoefficients) == 0 {
		return nil, false
	}
	numericCoefficients := make(map[int]*Num, len(rawCoefficients))
	for degree, coefficientExpression := range rawCoefficients {
		coefficientNumber, isNumber := coefficientExpression.(*Num)
		if !isNumber {
			return nil, false
		}
		numericCoefficients[degree] = coefficientNumber
	}
	return numericCoefficients, true
}

func highestPolynomialDegree(coefficients map[int]*Num) int {
	maximumDegree := -1
	for degree, coefficient := range coefficients {
		if coefficient == nil || coefficient.IsZero() {
			continue
		}
		if degree > maximumDegree {
			maximumDegree = degree
		}
	}
	return maximumDegree
}

func extractPolynomialContent(coefficients map[int]*Num) *Num {
	numeratorGreatestCommonDivisor := big.NewInt(0)
	denominatorLeastCommonMultiple := big.NewInt(1)
	for _, coefficient := range coefficients {
		if coefficient == nil || coefficient.IsZero() {
			continue
		}
		absoluteNumerator := new(big.Int).Abs(coefficient.val.Num())
		if numeratorGreatestCommonDivisor.Sign() == 0 {
			numeratorGreatestCommonDivisor.Set(absoluteNumerator)
		} else {
			numeratorGreatestCommonDivisor.GCD(nil, nil, numeratorGreatestCommonDivisor, absoluteNumerator)
		}
		denominatorLeastCommonMultiple = leastCommonMultipleBigInt(denominatorLeastCommonMultiple, coefficient.val.Denom())
	}
	if numeratorGreatestCommonDivisor.Sign() == 0 {
		return N(1)
	}
	return &Num{val: new(big.Rat).SetFrac(numeratorGreatestCommonDivisor, denominatorLeastCommonMultiple)}
}

func leastCommonMultipleBigInt(left, right *big.Int) *big.Int {
	if left.Sign() == 0 {
		return new(big.Int).Abs(right)
	}
	if right.Sign() == 0 {
		return new(big.Int).Abs(left)
	}
	greatestCommonDivisor := new(big.Int)
	greatestCommonDivisor.GCD(nil, nil, left, right)
	result := new(big.Int).Mul(new(big.Int).Abs(left), new(big.Int).Abs(right))
	return result.Div(result, greatestCommonDivisor)
}

func dividePolynomialCoefficients(coefficients map[int]*Num, divisor *Num) map[int]*Num {
	result := make(map[int]*Num, len(coefficients))
	for degree, coefficient := range coefficients {
		result[degree] = numDiv(coefficient, divisor)
	}
	return result
}

func copyPolynomialCoefficients(coefficients map[int]*Num) map[int]*Num {
	result := make(map[int]*Num, len(coefficients))
	for degree, coefficient := range coefficients {
		result[degree] = &Num{val: coefficient.Rat()}
	}
	return result
}

func normalizePolynomialToIntegers(coefficients map[int]*Num) map[int]*big.Int {
	commonDenominator := big.NewInt(1)
	for _, coefficient := range coefficients {
		if coefficient == nil || coefficient.IsZero() {
			continue
		}
		commonDenominator = leastCommonMultipleBigInt(commonDenominator, coefficient.val.Denom())
	}
	integerCoefficients := make(map[int]*big.Int, len(coefficients))
	for degree, coefficient := range coefficients {
		scaledNumerator := new(big.Int).Mul(coefficient.val.Num(), new(big.Int).Quo(commonDenominator, coefficient.val.Denom()))
		integerCoefficients[degree] = scaledNumerator
	}
	return integerCoefficients
}

func findRationalPolynomialRoot(coefficients map[int]*Num) (*Num, bool) {
	integerCoefficients := normalizePolynomialToIntegers(coefficients)
	degree := highestPolynomialDegree(coefficients)
	if degree <= 0 {
		return nil, false
	}
	leadingCoefficient := integerCoefficients[degree]
	constantCoefficient, hasConstant := integerCoefficients[0]
	if !hasConstant {
		constantCoefficient = big.NewInt(0)
	}
	if constantCoefficient.Sign() == 0 {
		return N(0), true
	}
	numeratorCandidates := integerDivisors(constantCoefficient)
	denominatorCandidates := integerDivisors(leadingCoefficient)
	for _, numeratorCandidate := range numeratorCandidates {
		for _, denominatorCandidate := range denominatorCandidates {
			if denominatorCandidate.Sign() == 0 {
				continue
			}
			root := &Num{val: new(big.Rat).SetFrac(numeratorCandidate, denominatorCandidate)}
			if polynomialEvaluatesToZero(coefficients, root.val) {
				return root, true
			}
			negativeRoot := numNeg(root)
			if polynomialEvaluatesToZero(coefficients, negativeRoot.val) {
				return negativeRoot, true
			}
		}
	}
	return nil, false
}

func integerDivisors(value *big.Int) []*big.Int {
	absoluteValue := new(big.Int).Abs(value)
	if absoluteValue.Sign() == 0 {
		return []*big.Int{big.NewInt(0)}
	}
	if absoluteValue.BitLen() > 31 {
		return []*big.Int{new(big.Int).Set(absoluteValue)}
	}
	intValue := absoluteValue.Int64()
	divisors := make([]*big.Int, 0)
	for candidate := int64(1); candidate*candidate <= intValue; candidate++ {
		if intValue%candidate != 0 {
			continue
		}
		divisors = append(divisors, big.NewInt(candidate))
		pairedDivisor := intValue / candidate
		if pairedDivisor != candidate {
			divisors = append(divisors, big.NewInt(pairedDivisor))
		}
	}
	return divisors
}

func polynomialEvaluatesToZero(coefficients map[int]*Num, point *big.Rat) bool {
	maximumDegree := highestPolynomialDegree(coefficients)
	accumulator := new(big.Rat)
	for degree := maximumDegree; degree >= 0; degree-- {
		accumulator.Mul(accumulator, point)
		if coefficient, hasCoefficient := coefficients[degree]; hasCoefficient {
			accumulator.Add(accumulator, coefficient.Rat())
		}
	}
	return accumulator.Sign() == 0
}

func dividePolynomialByLinearFactor(coefficients map[int]*Num, root *Num) map[int]*Num {
	maximumDegree := highestPolynomialDegree(coefficients)
	reducedCoefficients := make(map[int]*Num, maximumDegree)
	synthetic := make([]*big.Rat, maximumDegree+1)
	for degree := maximumDegree; degree >= 0; degree-- {
		if coefficient, hasCoefficient := coefficients[degree]; hasCoefficient {
			synthetic[maximumDegree-degree] = coefficient.Rat()
		} else {
			synthetic[maximumDegree-degree] = new(big.Rat)
		}
	}
	result := make([]*big.Rat, maximumDegree)
	result[0] = new(big.Rat).Set(synthetic[0])
	for index := 1; index < len(synthetic)-1; index++ {
		nextCoefficient := new(big.Rat).Mul(result[index-1], root.Rat())
		nextCoefficient.Add(nextCoefficient, synthetic[index])
		result[index] = nextCoefficient
	}
	for index, coefficient := range result {
		reducedDegree := maximumDegree - 1 - index
		reducedCoefficients[reducedDegree] = ratToNum(coefficient)
	}
	return reducedCoefficients
}

func linearFactorFromRoot(varName string, root *Num) Expr {
	return AddOf(S(varName), numNeg(root)).Simplify()
}

func factorQuadraticPolynomial(varName string, coefficients map[int]*Num) ([]Expr, bool) {
	quadraticCoefficient := coefficients[2]
	linearCoefficient := coefficients[1]
	if linearCoefficient == nil {
		linearCoefficient = N(0)
	}
	constantCoefficient := coefficients[0]
	if constantCoefficient == nil {
		constantCoefficient = N(0)
	}
	discriminant := numSub(numMul(linearCoefficient, linearCoefficient), numMul(N(4), numMul(quadraticCoefficient, constantCoefficient)))
	discriminantSquareRoot, hasExactSquareRoot := exactSquareRoot(discriminant)
	if !hasExactSquareRoot {
		return nil, false
	}
	twoA := numMul(N(2), quadraticCoefficient)
	rootOne := numDiv(numAdd(numNeg(linearCoefficient), discriminantSquareRoot), twoA)
	rootTwo := numDiv(numSub(numNeg(linearCoefficient), discriminantSquareRoot), twoA)
	return []Expr{
		quadraticCoefficient,
		linearFactorFromRoot(varName, rootOne),
		linearFactorFromRoot(varName, rootTwo),
	}, true
}

func exactSquareRoot(value *Num) (*Num, bool) {
	if value.IsNegative() {
		return nil, false
	}
	numeratorSquareRoot := integerSquareRoot(value.val.Num())
	denominatorSquareRoot := integerSquareRoot(value.val.Denom())
	if numeratorSquareRoot == nil || denominatorSquareRoot == nil {
		return nil, false
	}
	return &Num{val: new(big.Rat).SetFrac(numeratorSquareRoot, denominatorSquareRoot)}, true
}

func integerSquareRoot(value *big.Int) *big.Int {
	if value.Sign() < 0 {
		return nil
	}
	root := new(big.Int).Sqrt(value)
	if new(big.Int).Mul(root, root).Cmp(value) != 0 {
		return nil
	}
	return root
}

func buildLinearPolynomialFactor(varName string, coefficients map[int]*Num) (Expr, *Num) {
	return buildPolynomialFactorFromCoefficients(varName, coefficients)
}

func buildPolynomialFactorFromCoefficients(varName string, coefficients map[int]*Num) (Expr, *Num) {
	degrees := make([]int, 0, len(coefficients))
	for degree, coefficient := range coefficients {
		if coefficient == nil || coefficient.IsZero() {
			continue
		}
		degrees = append(degrees, degree)
	}
	sort.Sort(sort.Reverse(sort.IntSlice(degrees)))
	leadingDegree := degrees[0]
	leadingCoefficient := coefficients[leadingDegree]
	terms := make([]Expr, 0, len(degrees))
	for _, degree := range degrees {
		coefficient := numDiv(coefficients[degree], leadingCoefficient)
		switch degree {
		case 0:
			terms = append(terms, coefficient)
		case 1:
			terms = append(terms, MulOf(coefficient, S(varName)).Simplify())
		default:
			terms = append(terms, MulOf(coefficient, PowOf(S(varName), N(int64(degree)))).Simplify())
		}
	}
	return AddOf(terms...).Simplify(), leadingCoefficient
}

func ratToNum(value *big.Rat) *Num {
	return &Num{val: new(big.Rat).Set(value)}
}

func extractQuotient(e Expr) (num, denom Expr, ok bool) {
	m, isMul := e.(*Mul)
	if !isMul {
		return nil, nil, false
	}
	var numFactors, denomFactors []Expr
	for _, f := range m.factors {
		if p, isPow := f.(*Pow); isPow {
			if en, isNum := p.exp.(*Num); isNum && en.IsNegOne() {
				denomFactors = append(denomFactors, p.base)
				continue
			}
		}
		numFactors = append(numFactors, f)
	}
	if len(denomFactors) == 0 {
		return nil, nil, false
	}
	var n, d Expr
	switch len(numFactors) {
	case 0:
		n = N(1)
	case 1:
		n = numFactors[0]
	default:
		n = &Mul{factors: numFactors}
	}
	if len(denomFactors) == 1 {
		d = denomFactors[0]
	} else {
		d = &Mul{factors: denomFactors}
	}
	return n, d, true
}

// ============================================================
// Partial Derivatives and Vector Calculus
// ============================================================

// PDiff computes the partial derivative ∂/∂varName of expr.
func PDiff(expr Expr, varName string) Expr { return Diff(expr, varName) }

// Gradient returns the gradient ∇f as a slice of partial derivatives.
func Gradient(expr Expr, varNames []string) []Expr {
	result := make([]Expr, len(varNames))
	for i, v := range varNames {
		result[i] = PDiff(expr, v)
	}
	return result
}

// Jacobian returns the m×n Jacobian matrix.
func Jacobian(exprs []Expr, varNames []string) *Matrix {
	mat := NewMatrix(len(exprs), len(varNames))
	for i, e := range exprs {
		for j, v := range varNames {
			mat.Set(i, j, PDiff(e, v))
		}
	}
	return mat
}

// Hessian returns the n×n matrix of second partial derivatives.
func Hessian(expr Expr, varNames []string) *Matrix {
	n := len(varNames)
	mat := NewMatrix(n, n)
	for i, vi := range varNames {
		for j, vj := range varNames {
			mat.Set(i, j, PDiff(PDiff(expr, vi), vj))
		}
	}
	return mat
}

// Laplacian returns ∇²f = sum of second partial derivatives.
func Laplacian(expr Expr, varNames []string) Expr {
	terms := make([]Expr, len(varNames))
	for i, v := range varNames {
		terms[i] = PDiff(PDiff(expr, v), v)
	}
	return AddOf(terms...).Simplify()
}

// Divergence returns ∇·F for a vector field.
func Divergence(exprs []Expr, varNames []string) Expr {
	if len(exprs) != len(varNames) {
		panic("gosymbol: Divergence requires len(exprs) == len(varNames)")
	}
	terms := make([]Expr, len(exprs))
	for i := range exprs {
		terms[i] = PDiff(exprs[i], varNames[i])
	}
	return AddOf(terms...).Simplify()
}

// Curl returns ∇×F for a 3D vector field.
func Curl(field [3]Expr, vars [3]string) [3]Expr {
	return [3]Expr{
		AddOf(PDiff(field[2], vars[1]), MulOf(N(-1), PDiff(field[1], vars[2]))).Simplify(),
		AddOf(PDiff(field[0], vars[2]), MulOf(N(-1), PDiff(field[2], vars[0]))).Simplify(),
		AddOf(PDiff(field[1], vars[0]), MulOf(N(-1), PDiff(field[0], vars[1]))).Simplify(),
	}
}

// ============================================================
// Top-level convenience functions (original API)
// ============================================================

func Simplify(e Expr) Expr { return e.Simplify() }
func String(e Expr) string { return e.String() }
func LaTeX(e Expr) string  { return e.LaTeX() }

func Sub(expr Expr, varName string, value Expr) Expr {
	return expr.Sub(varName, value).Simplify()
}

func Diff(expr Expr, varName string) Expr {
	return expr.Diff(varName).Simplify()
}

func Diff2(expr Expr, varName string) Expr {
	return Diff(Diff(expr, varName), varName)
}

func DiffN(expr Expr, varName string, n int) Expr {
	if n < 0 {
		panic("gosymbol: DiffN requires n >= 0")
	}
	result := expr
	for i := 0; i < n; i++ {
		result = Diff(result, varName)
	}
	return result
}

func Expand(e Expr) Expr { return expandExpr(e).Simplify() }

func expandExpr(e Expr) Expr {
	switch v := e.(type) {
	case *Mul:
		expanded := make([]Expr, len(v.factors))
		for i, f := range v.factors {
			expanded[i] = expandExpr(f)
		}
		for i, f := range expanded {
			if a, ok := f.(*Add); ok {
				rest := make([]Expr, 0, len(expanded)-1)
				for j, ef := range expanded {
					if j != i {
						rest = append(rest, ef)
					}
				}
				terms := make([]Expr, len(a.terms))
				for k, t := range a.terms {
					terms[k] = expandExpr(MulOf(append([]Expr{t}, rest...)...))
				}
				return expandExpr(AddOf(terms...))
			}
		}
		return MulOf(expanded...)
	case *Add:
		newTerms := make([]Expr, len(v.terms))
		for i, t := range v.terms {
			newTerms[i] = expandExpr(t)
		}
		return AddOf(newTerms...)
	case *Pow:
		if n, ok := v.exp.(*Num); ok && n.IsInteger() {
			exp := n.val.Num().Int64()
			if exp >= 0 && exp <= 10 {
				result := Expr(N(1))
				base := expandExpr(v.base)
				for i := int64(0); i < exp; i++ {
					result = expandExpr(&Mul{factors: []Expr{result, base}})
				}
				return result
			}
		}
		return &Pow{base: expandExpr(v.base), exp: expandExpr(v.exp)}
	}
	return e
}

// ============================================================
// Free Symbols
// ============================================================

func FreeSymbols(e Expr) map[string]struct{} {
	result := map[string]struct{}{}
	collectSymbols(e, result)
	return result
}

func collectSymbols(e Expr, out map[string]struct{}) {
	switch v := e.(type) {
	case *Sym:
		out[v.name] = struct{}{}
	case *ConstantNode:
		return
	case *Add:
		for _, t := range v.terms {
			collectSymbols(t, out)
		}
	case *Mul:
		for _, f := range v.factors {
			collectSymbols(f, out)
		}
	case *Pow:
		collectSymbols(v.base, out)
		collectSymbols(v.exp, out)
	case *Func:
		collectSymbols(v.arg, out)
	case *PiecewiseExpression:
		for _, piecewiseCase := range v.Cases {
			collectSymbols(piecewiseCase.Expression, out)
		}
		if v.DefaultExpression != nil {
			collectSymbols(v.DefaultExpression, out)
		}
	}
}

// ============================================================
// Polynomial utilities
// ============================================================

func Degree(expr Expr, varName string) int {
	expr = expr.Simplify()
	switch v := expr.(type) {
	case *Num:
		return 0
	case *ConstantNode:
		return 0
	case *Sym:
		if v.name == varName {
			return 1
		}
		return 0
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 && n.IsInteger() {
				return int(n.val.Num().Int64())
			}
		}
		return 0
	case *Add:
		maxDeg := 0
		for _, t := range v.terms {
			if d := Degree(t, varName); d > maxDeg {
				maxDeg = d
			}
		}
		return maxDeg
	case *Mul:
		totalDeg := 0
		for _, f := range v.factors {
			totalDeg += Degree(f, varName)
		}
		return totalDeg
	}
	return 0
}

type PolyCoeffsResult map[int]Expr

func PolyCoeffs(expr Expr, varName string) PolyCoeffsResult {
	result := PolyCoeffsResult{}
	extractCoeffs(expr.Simplify(), varName, result)
	return result
}

func extractCoeffs(e Expr, varName string, out PolyCoeffsResult) {
	switch v := e.(type) {
	case *Num:
		addCoeff(out, 0, v)
	case *ConstantNode:
		addCoeff(out, 0, v)
	case *Sym:
		if v.name == varName {
			addCoeff(out, 1, N(1))
		} else {
			addCoeff(out, 0, v)
		}
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 && n.IsInteger() {
				addCoeff(out, int(n.val.Num().Int64()), N(1))
				return
			}
		}
		addCoeff(out, 0, e)
	case *Mul:
		deg := 0
		coeffFactors := []Expr{}
		for _, f := range v.factors {
			if d := Degree(f, varName); d > 0 {
				deg += d
			} else {
				coeffFactors = append(coeffFactors, f)
			}
		}
		var coeff Expr
		switch len(coeffFactors) {
		case 0:
			coeff = N(1)
		case 1:
			coeff = coeffFactors[0]
		default:
			coeff = MulOf(coeffFactors...)
		}
		addCoeff(out, deg, coeff)
	case *Add:
		for _, t := range v.terms {
			extractCoeffs(t, varName, out)
		}
	case *PiecewiseExpression:
		addCoeff(out, 0, v)
	}
}

func addCoeff(out PolyCoeffsResult, deg int, val Expr) {
	if existing, ok := out[deg]; ok {
		out[deg] = AddOf(existing, val).Simplify()
	} else {
		out[deg] = val.Simplify()
	}
}

type ApartResult struct {
	Terms []Expr
	Error string
}

func Apart(num, denom Expr, varName string) ApartResult {
	num = num.Simplify()
	denom = denom.Simplify()
	fr := Factor(denom, varName)
	if !fr.Success || len(fr.Factors) < 2 {
		return ApartResult{
			Terms: []Expr{MulOf(num, PowOf(denom, N(-1))).Simplify()},
			Error: "denominator could not be factored",
		}
	}
	dDen := Diff(denom, varName)
	var linearFactors, roots []Expr
	for _, factor := range fr.Factors {
		if Degree(factor, varName) != 1 {
			continue
		}
		coeffs := PolyCoeffs(factor, varName)
		a1 := N(1)
		b0 := N(0)
		if c, ok := coeffs[1]; ok {
			if cn, ok2 := c.(*Num); ok2 {
				a1 = cn
			}
		}
		if c, ok := coeffs[0]; ok {
			if cn, ok2 := c.(*Num); ok2 {
				b0 = cn
			}
		}
		if !a1.IsZero() {
			root := numDiv(numNeg(b0), a1)
			linearFactors = append(linearFactors, factor)
			roots = append(roots, root)
		}
	}
	if len(linearFactors) == 0 {
		return ApartResult{
			Terms: []Expr{MulOf(num, PowOf(denom, N(-1))).Simplify()},
			Error: "no linear factors found in denominator",
		}
	}
	terms := make([]Expr, len(roots))
	for i, root := range roots {
		residue := MulOf(Sub(num, varName, root), PowOf(Sub(dDen, varName, root), N(-1))).Simplify()
		terms[i] = MulOf(residue, PowOf(linearFactors[i], N(-1))).Simplify()
	}
	return ApartResult{Terms: terms}
}

// PatternRewriteRule describes a deterministic rewrite rule.
type PatternRewriteRule struct {
	Pattern     Expr
	Replacement Expr
}

// MatchExpressionPattern matches a pattern expression against a concrete expression.
// Symbolic variables whose names start with an underscore are treated as placeholders.
func MatchExpressionPattern(pattern Expr, expression Expr) (map[string]Expr, bool) {
	bindings := map[string]Expr{}
	if !matchExpressionPatternIntoBindings(pattern, expression, bindings) {
		return nil, false
	}
	return bindings, true
}

func matchExpressionPatternIntoBindings(pattern Expr, expression Expr, bindings map[string]Expr) bool {
	if patternSymbolicVariable, isVariable := pattern.(*Sym); isVariable && len(patternSymbolicVariable.name) > 0 && patternSymbolicVariable.name[0] == '_' {
		if existingBinding, hasBinding := bindings[patternSymbolicVariable.name]; hasBinding {
			return existingBinding.Equal(expression)
		}
		bindings[patternSymbolicVariable.name] = expression
		return true
	}
	switch typedPattern := pattern.(type) {
	case *Num, *ConstantNode, *Sym:
		return pattern.Equal(expression)
	case *Add:
		typedExpression, isAddition := expression.(*Add)
		if !isAddition || len(typedPattern.terms) != len(typedExpression.terms) {
			return false
		}
		for termIndex := range typedPattern.terms {
			if !matchExpressionPatternIntoBindings(typedPattern.terms[termIndex], typedExpression.terms[termIndex], bindings) {
				return false
			}
		}
		return true
	case *Mul:
		typedExpression, isMultiplication := expression.(*Mul)
		if !isMultiplication || len(typedPattern.factors) != len(typedExpression.factors) {
			return false
		}
		for factorIndex := range typedPattern.factors {
			if !matchExpressionPatternIntoBindings(typedPattern.factors[factorIndex], typedExpression.factors[factorIndex], bindings) {
				return false
			}
		}
		return true
	case *Pow:
		typedExpression, isPower := expression.(*Pow)
		return isPower &&
			matchExpressionPatternIntoBindings(typedPattern.base, typedExpression.base, bindings) &&
			matchExpressionPatternIntoBindings(typedPattern.exp, typedExpression.exp, bindings)
	case *Func:
		typedExpression, isFunction := expression.(*Func)
		return isFunction && typedPattern.name == typedExpression.name && matchExpressionPatternIntoBindings(typedPattern.arg, typedExpression.arg, bindings)
	default:
		return pattern.Equal(expression)
	}
}

// RewriteExpressionByPatternRules applies rewrite rules repeatedly until stable.
func RewriteExpressionByPatternRules(expression Expr, rules []PatternRewriteRule) Expr {
	currentExpression := expression
	for iterationIndex := 0; iterationIndex < 10; iterationIndex++ {
		replacedExpression, changed := rewriteExpressionByPatternRulesOnce(currentExpression, rules)
		if !changed {
			return currentExpression.Simplify()
		}
		currentExpression = replacedExpression.Simplify()
	}
	return currentExpression.Simplify()
}

func rewriteExpressionByPatternRulesOnce(expression Expr, rules []PatternRewriteRule) (Expr, bool) {
	for _, rule := range rules {
		if bindings, matched := MatchExpressionPattern(rule.Pattern, expression); matched {
			return substitutePatternBindings(rule.Replacement, bindings).Simplify(), true
		}
	}
	switch typedExpression := expression.(type) {
	case *Add:
		rewrittenTerms := make([]Expr, len(typedExpression.terms))
		changed := false
		for termIndex, term := range typedExpression.terms {
			rewrittenTerm, termChanged := rewriteExpressionByPatternRulesOnce(term, rules)
			rewrittenTerms[termIndex] = rewrittenTerm
			changed = changed || termChanged
		}
		if changed {
			return AddOf(rewrittenTerms...), true
		}
	case *Mul:
		rewrittenFactors := make([]Expr, len(typedExpression.factors))
		changed := false
		for factorIndex, factor := range typedExpression.factors {
			rewrittenFactor, factorChanged := rewriteExpressionByPatternRulesOnce(factor, rules)
			rewrittenFactors[factorIndex] = rewrittenFactor
			changed = changed || factorChanged
		}
		if changed {
			return MulOf(rewrittenFactors...), true
		}
	case *Pow:
		rewrittenBase, baseChanged := rewriteExpressionByPatternRulesOnce(typedExpression.base, rules)
		rewrittenExponent, exponentChanged := rewriteExpressionByPatternRulesOnce(typedExpression.exp, rules)
		if baseChanged || exponentChanged {
			return PowOf(rewrittenBase, rewrittenExponent), true
		}
	case *Func:
		rewrittenArgument, argumentChanged := rewriteExpressionByPatternRulesOnce(typedExpression.arg, rules)
		if argumentChanged {
			return funcOf(typedExpression.name, rewrittenArgument).Simplify(), true
		}
	}
	return expression, false
}

func substitutePatternBindings(expression Expr, bindings map[string]Expr) Expr {
	if symbolicVariable, isVariable := expression.(*Sym); isVariable {
		if replacement, hasReplacement := bindings[symbolicVariable.name]; hasReplacement {
			return replacement
		}
	}
	switch typedExpression := expression.(type) {
	case *Add:
		replacedTerms := make([]Expr, len(typedExpression.terms))
		for termIndex, term := range typedExpression.terms {
			replacedTerms[termIndex] = substitutePatternBindings(term, bindings)
		}
		return AddOf(replacedTerms...)
	case *Mul:
		replacedFactors := make([]Expr, len(typedExpression.factors))
		for factorIndex, factor := range typedExpression.factors {
			replacedFactors[factorIndex] = substitutePatternBindings(factor, bindings)
		}
		return MulOf(replacedFactors...)
	case *Pow:
		return PowOf(substitutePatternBindings(typedExpression.base, bindings), substitutePatternBindings(typedExpression.exp, bindings))
	case *Func:
		return funcOf(typedExpression.name, substitutePatternBindings(typedExpression.arg, bindings)).Simplify()
	default:
		return expression
	}
}

// ExpandTrigonometric applies common sum-angle and multiple-angle expansions.
func ExpandTrigonometric(expression Expr) Expr {
	switch typedExpression := expression.(type) {
	case *Func:
		expandedArgument := ExpandTrigonometric(typedExpression.arg)
		if additionArgument, isAddition := expandedArgument.(*Add); isAddition && len(additionArgument.terms) == 2 {
			leftTerm := additionArgument.terms[0]
			rightTerm := additionArgument.terms[1]
			switch typedExpression.name {
			case "sin":
				return AddOf(MulOf(SinOf(leftTerm), CosOf(rightTerm)), MulOf(CosOf(leftTerm), SinOf(rightTerm))).Simplify()
			case "cos":
				return AddOf(MulOf(CosOf(leftTerm), CosOf(rightTerm)), MulOf(N(-1), SinOf(leftTerm), SinOf(rightTerm))).Simplify()
			}
		}
		return funcOf(typedExpression.name, expandedArgument).Simplify()
	case *Pow:
		return PowOf(ExpandTrigonometric(typedExpression.base), ExpandTrigonometric(typedExpression.exp)).Simplify()
	case *Add:
		expandedTerms := make([]Expr, len(typedExpression.terms))
		for termIndex, term := range typedExpression.terms {
			expandedTerms[termIndex] = ExpandTrigonometric(term)
		}
		return AddOf(expandedTerms...).Simplify()
	case *Mul:
		expandedFactors := make([]Expr, len(typedExpression.factors))
		for factorIndex, factor := range typedExpression.factors {
			expandedFactors[factorIndex] = ExpandTrigonometric(factor)
		}
		return MulOf(expandedFactors...).Simplify()
	default:
		return expression
	}
}

// ExpandLogarithmic applies common logarithmic product and power expansions.
func ExpandLogarithmic(expression Expr) Expr {
	switch typedExpression := expression.(type) {
	case *Func:
		if typedExpression.name == "ln" {
			switch logarithmicArgument := typedExpression.arg.(type) {
			case *Mul:
				expandedTerms := make([]Expr, len(logarithmicArgument.factors))
				for factorIndex, factor := range logarithmicArgument.factors {
					expandedTerms[factorIndex] = LnOf(ExpandLogarithmic(factor))
				}
				return AddOf(expandedTerms...).Simplify()
			case *Pow:
				return MulOf(logarithmicArgument.exp, LnOf(ExpandLogarithmic(logarithmicArgument.base))).Simplify()
			}
		}
		return funcOf(typedExpression.name, ExpandLogarithmic(typedExpression.arg)).Simplify()
	case *Add:
		expandedTerms := make([]Expr, len(typedExpression.terms))
		for termIndex, term := range typedExpression.terms {
			expandedTerms[termIndex] = ExpandLogarithmic(term)
		}
		return AddOf(expandedTerms...).Simplify()
	case *Mul:
		expandedFactors := make([]Expr, len(typedExpression.factors))
		for factorIndex, factor := range typedExpression.factors {
			expandedFactors[factorIndex] = ExpandLogarithmic(factor)
		}
		return MulOf(expandedFactors...).Simplify()
	case *Pow:
		return PowOf(ExpandLogarithmic(typedExpression.base), ExpandLogarithmic(typedExpression.exp)).Simplify()
	default:
		return expression
	}
}

// ComputeGroebnerBasis computes a small Buchberger basis for exact multivariate polynomials.
func ComputeGroebnerBasis(polynomials []Expr, variableNames []string) []Expr {
	return ComputeGroebnerBasisWithContext(context.Background(), polynomials, variableNames)
}

// ComputeGroebnerBasisWithContext computes a small Buchberger basis with cancellation support.
func ComputeGroebnerBasisWithContext(operationContext context.Context, polynomials []Expr, variableNames []string) []Expr {
	if operationContext != nil && operationContext.Err() != nil {
		return nil
	}
	if len(polynomials) == 0 || len(variableNames) == 0 {
		return nil
	}
	basis := make([]map[string]*Num, 0, len(polynomials))
	for _, polynomial := range polynomials {
		if operationContext != nil && operationContext.Err() != nil {
			break
		}
		polynomialTerms, converted := convertExpressionToMultivariatePolynomial(polynomial, variableNames)
		if !converted {
			continue
		}
		basis = append(basis, polynomialTerms)
	}
	changed := true
	for changed {
		if operationContext != nil && operationContext.Err() != nil {
			return nil
		}
		changed = false
		for leftIndex := 0; leftIndex < len(basis); leftIndex++ {
			for rightIndex := leftIndex + 1; rightIndex < len(basis); rightIndex++ {
				if operationContext != nil && operationContext.Err() != nil {
					return nil
				}
				sPolynomial := computeSPolynomial(basis[leftIndex], basis[rightIndex], variableNames)
				reducedPolynomial := reduceMultivariatePolynomial(sPolynomial, basis, variableNames)
				if len(reducedPolynomial) == 0 {
					continue
				}
				basis = append(basis, reducedPolynomial)
				changed = true
			}
		}
	}
	result := make([]Expr, len(basis))
	for basisIndex, polynomialTerms := range basis {
		result[basisIndex] = convertMultivariatePolynomialToExpression(polynomialTerms, variableNames)
	}
	return result
}

func convertExpressionToMultivariatePolynomial(expression Expr, variableNames []string) (map[string]*Num, bool) {
	terms := map[string]*Num{}
	switch typedExpression := Expand(expression).Simplify().(type) {
	case *Add:
		for _, term := range typedExpression.terms {
			exponents, coefficient, converted := convertExpressionTermToMonomial(term, variableNames)
			if !converted {
				return nil, false
			}
			terms[stringifyExponentVector(exponents)] = addMultivariateCoefficient(terms[stringifyExponentVector(exponents)], coefficient)
		}
	default:
		exponents, coefficient, converted := convertExpressionTermToMonomial(typedExpression, variableNames)
		if !converted {
			return nil, false
		}
		terms[stringifyExponentVector(exponents)] = coefficient
	}
	for exponentKey, coefficient := range terms {
		if coefficient.IsZero() {
			delete(terms, exponentKey)
		}
	}
	return terms, true
}

func convertExpressionTermToMonomial(expression Expr, variableNames []string) ([]int, *Num, bool) {
	exponents := make([]int, len(variableNames))
	coefficient := N(1)
	var collectFactor func(Expr) bool
	collectFactor = func(factor Expr) bool {
		switch typedFactor := factor.(type) {
		case *Num:
			coefficient = numMul(coefficient, typedFactor)
			return true
		case *Sym:
			for variableIndex, variableName := range variableNames {
				if typedFactor.name == variableName {
					exponents[variableIndex]++
					return true
				}
			}
			return false
		case *Pow:
			symbolicVariable, isVariable := typedFactor.base.(*Sym)
			exponentNumber, isNumber := typedFactor.exp.(*Num)
			if !isVariable || !isNumber || !exponentNumber.IsInteger() {
				return false
			}
			for variableIndex, variableName := range variableNames {
				if symbolicVariable.name == variableName {
					exponents[variableIndex] += int(exponentNumber.val.Num().Int64())
					return true
				}
			}
			return false
		case *Mul:
			for _, innerFactor := range typedFactor.factors {
				if !collectFactor(innerFactor) {
					return false
				}
			}
			return true
		default:
			return false
		}
	}
	if !collectFactor(expression) {
		return nil, nil, false
	}
	return exponents, coefficient, true
}

func stringifyExponentVector(exponents []int) string {
	parts := make([]string, len(exponents))
	for exponentIndex, exponent := range exponents {
		parts[exponentIndex] = fmt.Sprintf("%d", exponent)
	}
	return strings.Join(parts, ",")
}

func parseExponentVector(serializedExponents string) []int {
	if serializedExponents == "" {
		return nil
	}
	parts := strings.Split(serializedExponents, ",")
	exponents := make([]int, len(parts))
	for exponentIndex, part := range parts {
		fmt.Sscanf(part, "%d", &exponents[exponentIndex])
	}
	return exponents
}

func addMultivariateCoefficient(left, right *Num) *Num {
	if left == nil {
		return right
	}
	return numAdd(left, right)
}

func leadingMultivariateTerm(polynomial map[string]*Num) ([]int, *Num, string) {
	var leadingKey string
	var leadingExponents []int
	for exponentKey := range polynomial {
		exponents := parseExponentVector(exponentKey)
		if leadingExponents == nil || compareExponentVectors(exponents, leadingExponents) > 0 {
			leadingExponents = exponents
			leadingKey = exponentKey
		}
	}
	return leadingExponents, polynomial[leadingKey], leadingKey
}

func compareExponentVectors(left, right []int) int {
	for exponentIndex := range left {
		if left[exponentIndex] != right[exponentIndex] {
			if left[exponentIndex] > right[exponentIndex] {
				return 1
			}
			return -1
		}
	}
	return 0
}

func computeSPolynomial(leftPolynomial, rightPolynomial map[string]*Num, variableNames []string) map[string]*Num {
	leftLeadingExponents, leftLeadingCoefficient, _ := leadingMultivariateTerm(leftPolynomial)
	rightLeadingExponents, rightLeadingCoefficient, _ := leadingMultivariateTerm(rightPolynomial)
	leastCommonMultipleExponents := make([]int, len(leftLeadingExponents))
	for exponentIndex := range leastCommonMultipleExponents {
		leastCommonMultipleExponents[exponentIndex] = leftLeadingExponents[exponentIndex]
		if rightLeadingExponents[exponentIndex] > leastCommonMultipleExponents[exponentIndex] {
			leastCommonMultipleExponents[exponentIndex] = rightLeadingExponents[exponentIndex]
		}
	}
	leftMultiplier := subtractExponentVectors(leastCommonMultipleExponents, leftLeadingExponents)
	rightMultiplier := subtractExponentVectors(leastCommonMultipleExponents, rightLeadingExponents)
	leftScaledPolynomial := scaleMultivariatePolynomial(leftPolynomial, leftMultiplier, numRecip(leftLeadingCoefficient))
	rightScaledPolynomial := scaleMultivariatePolynomial(rightPolynomial, rightMultiplier, numRecip(rightLeadingCoefficient))
	return subtractMultivariatePolynomials(leftScaledPolynomial, rightScaledPolynomial)
}

func subtractExponentVectors(left, right []int) []int {
	result := make([]int, len(left))
	for exponentIndex := range left {
		result[exponentIndex] = left[exponentIndex] - right[exponentIndex]
	}
	return result
}

func scaleMultivariatePolynomial(polynomial map[string]*Num, exponentOffset []int, coefficientScale *Num) map[string]*Num {
	result := map[string]*Num{}
	for exponentKey, coefficient := range polynomial {
		exponents := parseExponentVector(exponentKey)
		for exponentIndex := range exponents {
			exponents[exponentIndex] += exponentOffset[exponentIndex]
		}
		result[stringifyExponentVector(exponents)] = numMul(coefficient, coefficientScale)
	}
	return result
}

func subtractMultivariatePolynomials(leftPolynomial, rightPolynomial map[string]*Num) map[string]*Num {
	result := map[string]*Num{}
	for exponentKey, coefficient := range leftPolynomial {
		result[exponentKey] = coefficient
	}
	for exponentKey, coefficient := range rightPolynomial {
		if existingCoefficient, hasExistingCoefficient := result[exponentKey]; hasExistingCoefficient {
			result[exponentKey] = numSub(existingCoefficient, coefficient)
		} else {
			result[exponentKey] = numNeg(coefficient)
		}
	}
	for exponentKey, coefficient := range result {
		if coefficient.IsZero() {
			delete(result, exponentKey)
		}
	}
	return result
}

func reduceMultivariatePolynomial(polynomial map[string]*Num, basis []map[string]*Num, variableNames []string) map[string]*Num {
	remainder := map[string]*Num{}
	workingPolynomial := polynomial
	for len(workingPolynomial) > 0 {
		reduced := false
		workingLeadingExponents, workingLeadingCoefficient, workingLeadingKey := leadingMultivariateTerm(workingPolynomial)
		for _, basisPolynomial := range basis {
			basisLeadingExponents, basisLeadingCoefficient, _ := leadingMultivariateTerm(basisPolynomial)
			if exponentVectorDivides(basisLeadingExponents, workingLeadingExponents) {
				exponentDifference := subtractExponentVectors(workingLeadingExponents, basisLeadingExponents)
				scale := numDiv(workingLeadingCoefficient, basisLeadingCoefficient)
				workingPolynomial = subtractMultivariatePolynomials(workingPolynomial, scaleMultivariatePolynomial(basisPolynomial, exponentDifference, scale))
				reduced = true
				break
			}
		}
		if reduced {
			continue
		}
		remainder[workingLeadingKey] = workingLeadingCoefficient
		delete(workingPolynomial, workingLeadingKey)
	}
	return remainder
}

func exponentVectorDivides(divisor, dividend []int) bool {
	for exponentIndex := range divisor {
		if divisor[exponentIndex] > dividend[exponentIndex] {
			return false
		}
	}
	return true
}

func convertMultivariatePolynomialToExpression(polynomial map[string]*Num, variableNames []string) Expr {
	if len(polynomial) == 0 {
		return N(0)
	}
	terms := make([]Expr, 0, len(polynomial))
	for exponentKey, coefficient := range polynomial {
		exponents := parseExponentVector(exponentKey)
		factors := []Expr{coefficient}
		for exponentIndex, exponent := range exponents {
			if exponent == 0 {
				continue
			}
			if exponent == 1 {
				factors = append(factors, S(variableNames[exponentIndex]))
			} else {
				factors = append(factors, PowOf(S(variableNames[exponentIndex]), N(int64(exponent))))
			}
		}
		terms = append(terms, MulOf(factors...).Simplify())
	}
	return AddOf(terms...).Simplify()
}
