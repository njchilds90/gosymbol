package gosymbol

import (
	"context"
	"math"
)

// ============================================================
// Integration (rule-based symbolic + numerical)
// ============================================================

func Integrate(expr Expr, varName string) (Expr, bool) {
	expr = expr.Simplify()
	switch v := expr.(type) {
	case *Num:
		return MulOf(v, S(varName)), true
	case *Sym:
		if v.name == varName {
			return MulOf(F(1, 2), PowOf(S(varName), N(2))), true
		}
		return MulOf(v, S(varName)), true
	case *Pow:
		if sym, ok := v.base.(*Sym); ok && sym.name == varName {
			if n, ok2 := v.exp.(*Num); ok2 {
				if n.IsNegOne() {
					return LnOf(AbsOf(S(varName))), true
				}
				newExp := numAdd(n, N(1))
				return MulOf(numRecip(newExp), PowOf(S(varName), newExp)), true
			}
		}
		if sym, ok := v.exp.(*Sym); ok && sym.name == varName {
			if _, ok2 := v.base.(*Num); ok2 {
				return MulOf(PowOf(v.base, S(varName)), PowOf(LnOf(v.base), N(-1))), true
			}
		}
		return nil, false
	case *Mul:
		coeff := N(1)
		rest := []Expr{}
		foundNumericCoefficient := false
		for _, f := range v.factors {
			if n, ok := f.(*Num); ok {
				coeff = numMul(coeff, n)
				foundNumericCoefficient = true
			} else {
				rest = append(rest, f)
			}
		}
		if !foundNumericCoefficient {
			return nil, false
		}
		var inner Expr
		if len(rest) == 1 {
			inner = rest[0]
		} else {
			inner = &Mul{factors: rest}
		}
		intInner, ok := Integrate(inner, varName)
		if !ok {
			return nil, false
		}
		return MulOf(coeff, intInner).Simplify(), true
	case *Add:
		terms := make([]Expr, len(v.terms))
		for i, t := range v.terms {
			intT, ok := Integrate(t, varName)
			if !ok {
				return nil, false
			}
			terms[i] = intT
		}
		return AddOf(terms...).Simplify(), true
	case *Func:
		if linearCoefficient, isLinear := linearArgumentCoefficient(v.arg, varName); isLinear {
			inverseLinearCoefficient := numRecip(linearCoefficient)
			switch v.name {
			case "sin":
				return MulOf(N(-1), inverseLinearCoefficient, CosOf(v.arg)).Simplify(), true
			case "cos":
				return MulOf(inverseLinearCoefficient, SinOf(v.arg)).Simplify(), true
			case "sinh":
				return MulOf(inverseLinearCoefficient, CoshOf(v.arg)).Simplify(), true
			case "cosh":
				return MulOf(inverseLinearCoefficient, SinhOf(v.arg)).Simplify(), true
			case "exp":
				return MulOf(inverseLinearCoefficient, ExpOf(v.arg)).Simplify(), true
			}
		}
		switch v.name {
		case "sin":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return MulOf(N(-1), CosOf(S(varName))), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(N(-1), numRecip(coeff), CosOf(v.arg)), true
					}
				}
			}
		case "cos":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return SinOf(S(varName)), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(numRecip(coeff), SinOf(v.arg)), true
					}
				}
			}
		case "exp":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return ExpOf(S(varName)), true
			}
			if m, ok := v.arg.(*Mul); ok && len(m.factors) == 2 {
				if coeff, ok2 := m.factors[0].(*Num); ok2 {
					if sym, ok3 := m.factors[1].(*Sym); ok3 && sym.name == varName {
						return MulOf(numRecip(coeff), ExpOf(v.arg)), true
					}
				}
			}
		case "ln":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(MulOf(S(varName), LnOf(S(varName))), MulOf(N(-1), S(varName))), true
			}
		case "asin":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AsinOf(S(varName))),
					SqrtOf(AddOf(N(1), MulOf(N(-1), PowOf(S(varName), N(2))))),
				), true
			}
		case "acos":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AcosOf(S(varName))),
					MulOf(N(-1), SqrtOf(AddOf(N(1), MulOf(N(-1), PowOf(S(varName), N(2)))))),
				), true
			}
		case "atan":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AtanOf(S(varName))),
					MulOf(N(-1), F(1, 2), LnOf(AddOf(N(1), PowOf(S(varName), N(2))))),
				), true
			}
		case "sinh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return CoshOf(S(varName)), true
			}
		case "cosh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return SinhOf(S(varName)), true
			}
		case "tanh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return LnOf(CoshOf(S(varName))), true
			}
		case "asinh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AsinhOf(S(varName))),
					MulOf(N(-1), SqrtOf(AddOf(PowOf(S(varName), N(2)), N(1)))),
				), true
			}
		case "acosh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AcoshOf(S(varName))),
					MulOf(
						N(-1),
						SqrtOf(AddOf(S(varName), N(-1))),
						SqrtOf(AddOf(S(varName), N(1))),
					),
				), true
			}
		case "atanh":
			if sym, ok := v.arg.(*Sym); ok && sym.name == varName {
				return AddOf(
					MulOf(S(varName), AtanhOf(S(varName))),
					MulOf(F(1, 2), LnOf(AddOf(N(1), MulOf(N(-1), PowOf(S(varName), N(2)))))),
				), true
			}
		}
		return nil, false
	}
	return nil, false
}

// IntegrateWithConstant returns an antiderivative with an explicit +C term.
func IntegrateWithConstant(expr Expr, varName string) (Expr, bool) {
	result, ok := Integrate(expr, varName)
	if !ok {
		return nil, false
	}
	return AddOf(result, CreateArbitraryIntegrationConstant()), true
}

func linearArgumentCoefficient(argument Expr, varName string) (*Num, bool) {
	switch linearArgument := argument.Simplify().(type) {
	case *Sym:
		if linearArgument.name == varName {
			return N(1), true
		}
	case *Mul:
		if len(linearArgument.factors) == 2 {
			if coefficient, isNumber := linearArgument.factors[0].(*Num); isNumber {
				if variable, isVariable := linearArgument.factors[1].(*Sym); isVariable && variable.name == varName {
					return coefficient, true
				}
			}
		}
	}
	return nil, false
}

func DefiniteIntegrate(expr Expr, varName string, a, b float64) float64 {
	nodes := []float64{
		-0.9739065285, -0.8650633667, -0.6794095683,
		-0.4333953941, -0.1488743390, 0.1488743390,
		0.4333953941, 0.6794095683, 0.8650633667, 0.9739065285,
	}
	weights := []float64{
		0.0666713443, 0.1494513492, 0.2190863625,
		0.2692667193, 0.2955242247, 0.2955242247,
		0.2692667193, 0.2190863625, 0.1494513492, 0.0666713443,
	}
	sum := 0.0
	mid := (a + b) / 2
	half := (b - a) / 2
	for i, t := range nodes {
		xi := mid + half*t
		subbed := expr.Sub(varName, NFloat(xi))
		if v, ok := subbed.Eval(); ok {
			f, _ := v.val.Float64()
			sum += weights[i] * f
		}
	}
	return half * sum
}

// ============================================================
// Limits
// ============================================================

// LimitResult holds the result of a limit computation.
type LimitResult struct {
	Value   Expr
	Success bool
	Error   string
}

// Limit computes a two-sided symbolic limit.
func Limit(expr Expr, varName string, point Expr) LimitResult {
	return LimitWithDirection(expr, varName, point, "")
}

// LimitWithDirection computes a symbolic limit with optional one-sided or
// infinity-aware direction handling. Supported directions are "", "both", "+",
// "-", "left", and "right".
func LimitWithDirection(expr Expr, varName string, point Expr, direction string) LimitResult {
	normalizedDirection := normalizeLimitDirection(direction)
	if normalizedDirection == "both" {
		leftLimitResult := limitRecursive(expr, varName, point, "-", 5)
		rightLimitResult := limitRecursive(expr, varName, point, "+", 5)
		if leftLimitResult.Success && rightLimitResult.Success && leftLimitResult.Value.Equal(rightLimitResult.Value) {
			return leftLimitResult
		}
		if leftLimitResult.Success && rightLimitResult.Success && leftLimitResult.Value.String() == rightLimitResult.Value.String() {
			return LimitResult{Value: leftLimitResult.Value, Success: true}
		}
		return LimitResult{
			Error:   "two-sided limit does not exist: left and right limits differ",
			Success: false,
		}
	}
	return limitRecursive(expr, varName, point, normalizedDirection, 5)
}

func normalizeLimitDirection(direction string) string {
	switch direction {
	case "", "both":
		return "both"
	case "+", "right":
		return "+"
	case "-", "left":
		return "-"
	default:
		return direction
	}
}

func limitRecursive(expr Expr, varName string, point Expr, direction string, maxLhopitalApplications int) LimitResult {
	expr = DeepSimplify(expr)
	if isPositiveInfinityExpression(point) || isNegativeInfinityExpression(point) {
		return limitAtInfinity(expr, varName, point)
	}
	if numeratorExpression, denominatorExpression, isQuotient := extractQuotient(expr); isQuotient {
		numeratorAtPoint := numeratorExpression.Sub(varName, point).Simplify()
		denominatorAtPoint := denominatorExpression.Sub(varName, point).Simplify()
		if isZeroExpression(denominatorAtPoint) {
			if isZeroExpression(numeratorAtPoint) && maxLhopitalApplications > 0 {
				return limitRecursive(
					MulOf(Diff(numeratorExpression, varName), PowOf(Diff(denominatorExpression, varName), N(-1))).Simplify(),
					varName,
					point,
					direction,
					maxLhopitalApplications-1,
				)
			}
			if pointNumber, ok := point.Eval(); ok {
				return numericFiniteLimit(expr, varName, pointNumber.Float64(), direction)
			}
		}
	}
	substitutedExpression := expr.Sub(varName, point).Simplify()
	if numericValue, ok := substitutedExpression.Eval(); ok {
		floatingPointValue := numericValue.Float64()
		if !hasFiniteSingularity(substitutedExpression) && !math.IsNaN(floatingPointValue) && !math.IsInf(floatingPointValue, 0) {
			return LimitResult{Value: substitutedExpression, Success: true}
		}
	}
	freeVariableSet := FreeSymbols(substitutedExpression)
	if _, dependsOnVariable := freeVariableSet[varName]; !dependsOnVariable && !hasFiniteSingularity(substitutedExpression) {
		return LimitResult{Value: substitutedExpression, Success: true}
	}
	if pointNumber, ok := point.Eval(); ok {
		series := TaylorSeries(expr, varName, point, 4)
		seriesValue := series.Sub(varName, point).Simplify()
		if seriesNumericValue, ok := seriesValue.Eval(); ok {
			floatingPointValue := seriesNumericValue.Float64()
			if !math.IsNaN(floatingPointValue) && !math.IsInf(floatingPointValue, 0) {
				return LimitResult{Value: seriesValue, Success: true}
			}
		}
		return numericFiniteLimit(expr, varName, pointNumber.Float64(), direction)
	}
	return LimitResult{
		Error:   "limit could not be determined: " + expr.String() + " as " + varName + " -> " + point.String(),
		Success: false,
	}
}

func limitAtInfinity(expr Expr, varName string, point Expr) LimitResult {
	polynomialNumerator, polynomialDenominator, isQuotient := extractQuotient(expr)
	if !isQuotient {
		polynomialNumerator = expr
		polynomialDenominator = N(1)
	}
	numeratorCoefficients, numeratorIsPolynomial := polynomialNumericCoefficients(Collect(polynomialNumerator, varName).Simplify(), varName)
	denominatorCoefficients, denominatorIsPolynomial := polynomialNumericCoefficients(Collect(polynomialDenominator, varName).Simplify(), varName)
	if numeratorIsPolynomial && denominatorIsPolynomial {
		numeratorDegree := highestPolynomialDegree(numeratorCoefficients)
		denominatorDegree := highestPolynomialDegree(denominatorCoefficients)
		numeratorLeadingCoefficient := numeratorCoefficients[numeratorDegree]
		denominatorLeadingCoefficient := denominatorCoefficients[denominatorDegree]
		switch {
		case numeratorDegree < denominatorDegree:
			return LimitResult{Value: N(0), Success: true}
		case numeratorDegree == denominatorDegree:
			return LimitResult{Value: numDiv(numeratorLeadingCoefficient, denominatorLeadingCoefficient), Success: true}
		default:
			sign := infinitySignFromLeadingTerm(numDiv(numeratorLeadingCoefficient, denominatorLeadingCoefficient), numeratorDegree-denominatorDegree, point)
			return LimitResult{Value: infinityExpression(sign), Success: true}
		}
	}
	return numericInfinityLimit(expr, varName, point)
}

func infinitySignFromLeadingTerm(leadingRatio *Num, parityDifference int, point Expr) int {
	sign := 1
	if leadingRatio.IsNegative() {
		sign = -1
	}
	if isNegativeInfinityExpression(point) && parityDifference%2 != 0 {
		sign *= -1
	}
	return sign
}

func numericFiniteLimit(expr Expr, varName string, point float64, direction string) LimitResult {
	samplePoints := limitProbePoints(point, direction)
	samples := make([]float64, 0, len(samplePoints))
	for _, samplePoint := range samplePoints {
		evaluatedExpression := expr.Sub(varName, NFloat(samplePoint)).Simplify()
		numericValue, ok := evaluatedExpression.Eval()
		if !ok {
			continue
		}
		samples = append(samples, numericValue.Float64())
	}
	return summarizeLimitSamples(samples)
}

func numericInfinityLimit(expr Expr, varName string, point Expr) LimitResult {
	sign := 1.0
	if isNegativeInfinityExpression(point) {
		sign = -1.0
	}
	probeMagnitudes := []float64{1e3, 1e4, 1e5}
	samples := make([]float64, 0, len(probeMagnitudes))
	for _, probeMagnitude := range probeMagnitudes {
		evaluatedExpression := expr.Sub(varName, NFloat(sign*probeMagnitude)).Simplify()
		numericValue, ok := evaluatedExpression.Eval()
		if !ok {
			continue
		}
		samples = append(samples, numericValue.Float64())
	}
	return summarizeLimitSamples(samples)
}

func limitProbePoints(point float64, direction string) []float64 {
	offsets := []float64{1e-3, 1e-5, 1e-7}
	points := make([]float64, 0, len(offsets))
	for _, offset := range offsets {
		switch direction {
		case "+":
			points = append(points, point+offset)
		case "-":
			points = append(points, point-offset)
		default:
			points = append(points, point-offset, point+offset)
		}
	}
	return points
}

func summarizeLimitSamples(samples []float64) LimitResult {
	if len(samples) == 0 {
		return LimitResult{Error: "limit could not be determined numerically", Success: false}
	}
	allLargePositive := true
	allLargeNegative := true
	strictlyIncreasingMagnitude := true
	maximumDeviation := 0.0
	referenceValue := samples[len(samples)-1]
	for index, sample := range samples {
		if math.IsNaN(sample) {
			return LimitResult{Error: "limit produced NaN during probing", Success: false}
		}
		if sample <= 1e5 {
			allLargePositive = false
		}
		if sample >= -1e5 {
			allLargeNegative = false
		}
		if index > 0 && math.Abs(sample) <= math.Abs(samples[index-1]) {
			strictlyIncreasingMagnitude = false
		}
		maximumDeviation = math.Max(maximumDeviation, math.Abs(sample-referenceValue))
	}
	switch {
	case allLargePositive:
		return LimitResult{Value: infinityExpression(1), Success: true}
	case allLargeNegative:
		return LimitResult{Value: infinityExpression(-1), Success: true}
	case strictlyIncreasingMagnitude && referenceValue > 0 && math.Abs(referenceValue) > 1e5:
		return LimitResult{Value: infinityExpression(1), Success: true}
	case strictlyIncreasingMagnitude && referenceValue < 0 && math.Abs(referenceValue) > 1e5:
		return LimitResult{Value: infinityExpression(-1), Success: true}
	case math.Abs(referenceValue) < 1e-8:
		return LimitResult{Value: N(0), Success: true}
	case maximumDeviation < 1e-6:
		return LimitResult{Value: NFloat(referenceValue).Simplify(), Success: true}
	default:
		return LimitResult{Error: "limit could not be determined from stable probes", Success: false}
	}
}

func isZeroExpression(expr Expr) bool {
	numericValue, ok := expr.Eval()
	return ok && numericValue.IsZero()
}

func hasFiniteSingularity(expr Expr) bool {
	switch singularityCandidate := expr.(type) {
	case *Pow:
		if exponent, isNumber := singularityCandidate.exp.(*Num); isNumber && exponent.IsNegative() {
			if baseValue, ok := singularityCandidate.base.Eval(); ok && baseValue.IsZero() {
				return true
			}
		}
		return hasFiniteSingularity(singularityCandidate.base) || hasFiniteSingularity(singularityCandidate.exp)
	case *Add:
		for _, term := range singularityCandidate.terms {
			if hasFiniteSingularity(term) {
				return true
			}
		}
	case *Mul:
		for _, factor := range singularityCandidate.factors {
			if hasFiniteSingularity(factor) {
				return true
			}
		}
	case *Func:
		return hasFiniteSingularity(singularityCandidate.arg)
	}
	return false
}

func infinityExpression(sign int) Expr {
	if sign < 0 {
		return CreateConstantNode("-inf")
	}
	return CreateConstantNode("inf")
}

func isPositiveInfinityExpression(expr Expr) bool {
	switch infinityCandidate := expr.(type) {
	case *ConstantNode:
		return infinityCandidate.name == "inf" || infinityCandidate.name == "+inf"
	case *Sym:
		return infinityCandidate.name == "inf" || infinityCandidate.name == "+inf" || infinityCandidate.name == "infinity"
	}
	return false
}

func isNegativeInfinityExpression(expr Expr) bool {
	switch infinityCandidate := expr.(type) {
	case *ConstantNode:
		return infinityCandidate.name == "-inf"
	case *Sym:
		return infinityCandidate.name == "-inf"
	}
	return false
}

// ============================================================
// Taylor / Maclaurin series
// ============================================================

func TaylorSeries(expr Expr, varName string, a Expr, order int) Expr {
	terms := []Expr{}
	current := expr
	factorial := N(1)
	for k := 0; k <= order; k++ {
		if k > 0 {
			factorial = numMul(factorial, N(int64(k)))
		}
		coeff := MulOf(current.Sub(varName, a), PowOf(factorial, N(-1))).Simplify()
		if n, ok := coeff.(*Num); ok && n.IsZero() {
			current = Diff(current, varName)
			continue
		}
		var xTerm Expr
		if k == 0 {
			xTerm = coeff
		} else if k == 1 {
			xTerm = MulOf(coeff, AddOf(S(varName), MulOf(N(-1), a)))
		} else {
			xTerm = MulOf(coeff, PowOf(AddOf(S(varName), MulOf(N(-1), a)), N(int64(k))))
		}
		terms = append(terms, xTerm)
		current = Diff(current, varName)
	}
	return AddOf(terms...).Simplify()
}

func TaylorSeriesWithRemainder(expr Expr, varName string, a Expr, order int) Expr {
	series := TaylorSeries(expr, varName, a, order)
	remainder := OTerm(varName, order+1)
	if add, ok := series.(*Add); ok {
		return &Add{terms: append(add.terms, remainder)}
	}
	return &Add{terms: []Expr{series, remainder}}
}

func MaclaurinSeries(expr Expr, varName string, order int) Expr {
	return TaylorSeries(expr, varName, N(0), order)
}

func MaclaurinSeriesWithRemainder(expr Expr, varName string, order int) Expr {
	return TaylorSeriesWithRemainder(expr, varName, N(0), order)
}

func factorialNumber(order int) *Num {
	factorial := N(1)
	for factor := 2; factor <= order; factor++ {
		factorial = numMul(factorial, N(int64(factor)))
	}
	return factorial
}

// TaylorSeriesWithContext computes a Taylor series while honoring context cancellation.
func TaylorSeriesWithContext(operationContext context.Context, expr Expr, varName string, a Expr, order int) (Expr, error) {
	terms := make([]Expr, 0, order+1)
	current := expr
	for derivativeOrder := 0; derivativeOrder <= order; derivativeOrder++ {
		if operationContext != nil && operationContext.Err() != nil {
			return nil, operationContext.Err()
		}
		factorial := factorialNumber(derivativeOrder)
		coefficient := MulOf(current.Sub(varName, a), PowOf(factorial, N(-1))).Simplify()
		if numericCoefficient, isNumber := coefficient.(*Num); isNumber && numericCoefficient.IsZero() {
			current = Diff(current, varName)
			continue
		}
		switch derivativeOrder {
		case 0:
			terms = append(terms, coefficient)
		case 1:
			terms = append(terms, MulOf(coefficient, AddOf(S(varName), MulOf(N(-1), a))).Simplify())
		default:
			terms = append(terms, MulOf(coefficient, PowOf(AddOf(S(varName), MulOf(N(-1), a)), N(int64(derivativeOrder)))).Simplify())
		}
		current = Diff(current, varName)
	}
	return AddOf(terms...).Simplify(), nil
}

// PerformRischTranscendentalIntegration attempts a deterministic transcendental integration pass
// before falling back to the existing rule-based integrator.
func PerformRischTranscendentalIntegration(expression Expr, variableName string) (Expr, bool) {
	result, success, _ := PerformRischTranscendentalIntegrationWithContext(context.Background(), expression, variableName)
	return result, success
}

// PerformRischTranscendentalIntegrationWithContext attempts a deterministic transcendental
// integration pass and allows cancellation for longer-running callers.
func PerformRischTranscendentalIntegrationWithContext(operationContext context.Context, expression Expr, variableName string) (Expr, bool, error) {
	if operationContext != nil && operationContext.Err() != nil {
		return nil, false, operationContext.Err()
	}
	if integratedExpression, integrationSucceeded := Integrate(expression, variableName); integrationSucceeded {
		return integratedExpression, true, nil
	}
	switch typedExpression := expression.Simplify().(type) {
	case *Mul:
		if len(typedExpression.factors) == 2 {
			if logarithmicFunction, isFunction := typedExpression.factors[0].(*Func); isFunction && logarithmicFunction.name == "ln" {
				if inverseVariablePower, isPower := typedExpression.factors[1].(*Pow); isPower {
					if symbolicVariable, isVariable := inverseVariablePower.base.(*Sym); isVariable && symbolicVariable.name == variableName {
						if exponentNumber, isNumber := inverseVariablePower.exp.(*Num); isNumber && exponentNumber.IsNegOne() {
							return MulOf(F(1, 2), PowOf(LnOf(S(variableName)), N(2))).Simplify(), true, nil
						}
					}
				}
			}
		}
	case *Pow:
		if functionExpression, isFunction := typedExpression.base.(*Func); isFunction && functionExpression.name == "exp" {
			if exponentNumber, isNumber := typedExpression.exp.(*Num); isNumber && exponentNumber.IsInteger() && exponentNumber.val.Num().Int64() == 2 {
				if linearCoefficient, isLinear := linearArgumentCoefficient(functionExpression.arg, variableName); isLinear {
					return MulOf(numRecip(numMul(N(2), linearCoefficient)), PowOf(ExpOf(functionExpression.arg), N(2))).Simplify(), true, nil
				}
			}
		}
	}
	return nil, false, nil
}

// SolveFirstOrderLinearOrdinaryDifferentialEquation solves y' + p(x) y = q(x) by integrating factor.
func SolveFirstOrderLinearOrdinaryDifferentialEquation(variableName string, coefficientOfUnknownFunction Expr, rightHandSide Expr) (Expr, bool) {
	integratingFactorExponent, exponentSucceeded := PerformRischTranscendentalIntegration(coefficientOfUnknownFunction, variableName)
	if !exponentSucceeded {
		return nil, false
	}
	integratingFactor := ExpOf(integratingFactorExponent)
	weightedRightHandSide := MulOf(integratingFactor, rightHandSide).Simplify()
	weightedIntegral, integralSucceeded := PerformRischTranscendentalIntegration(weightedRightHandSide, variableName)
	if !integralSucceeded {
		return nil, false
	}
	return MulOf(
		PowOf(integratingFactor, N(-1)),
		AddOf(weightedIntegral, CreateArbitraryIntegrationConstant()),
	).Simplify(), true
}
