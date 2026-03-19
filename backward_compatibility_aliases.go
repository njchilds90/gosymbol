package gosymbol

import "sort"

// Expression is the full-name public alias for Expr.
type Expression = Expr

// RationalNumberNode is the full-name public alias for Num.
type RationalNumberNode = Num

// SymbolicVariableNode is the full-name public alias for Sym.
type SymbolicVariableNode = Sym

// AdditionNode is the full-name public alias for Add.
type AdditionNode = Add

// MultiplicationNode is the full-name public alias for Mul.
type MultiplicationNode = Mul

// PowerNode is the full-name public alias for Pow.
type PowerNode = Pow

// FunctionNode is the full-name public alias for Func.
type FunctionNode = Func

// SymbolicEquation is the full-name public alias for Equation.
type SymbolicEquation = Equation

// CreateRationalNumber constructs an integer-valued rational number node.
func CreateRationalNumber(integerValue int64) *RationalNumberNode { return N(integerValue) }

// CreateRationalFraction constructs an exact rational fraction node.
func CreateRationalFraction(numerator, denominator int64) *RationalNumberNode {
	return F(numerator, denominator)
}

// CreateApproximateRationalNumber constructs a floating-point-backed rational node.
func CreateApproximateRationalNumber(floatingPointValue float64) *RationalNumberNode {
	return NFloat(floatingPointValue)
}

// CreateSymbolicVariable constructs a symbolic variable node.
func CreateSymbolicVariable(variableName string) *SymbolicVariableNode { return S(variableName) }

// CreateSymbolicVariableWithAssumptions constructs a symbolic variable with assumptions.
func CreateSymbolicVariableWithAssumptions(variableName string, assumptions Assumptions) *SymbolicVariableNode {
	return SAssume(variableName, assumptions)
}

func CreateAddition(terms ...Expression) Expression         { return AddOf(terms...) }
func CreateMultiplication(factors ...Expression) Expression { return MulOf(factors...) }
func CreatePower(baseExpression, exponentExpression Expression) Expression {
	return PowOf(baseExpression, exponentExpression)
}

func CreateSine(argument Expression) Expression              { return SinOf(argument) }
func CreateCosine(argument Expression) Expression            { return CosOf(argument) }
func CreateTangent(argument Expression) Expression           { return TanOf(argument) }
func CreateExponential(argument Expression) Expression       { return ExpOf(argument) }
func CreateNaturalLogarithm(argument Expression) Expression  { return LnOf(argument) }
func CreateAbsoluteValue(argument Expression) Expression     { return AbsOf(argument) }
func CreateArcSine(argument Expression) Expression           { return AsinOf(argument) }
func CreateArcCosine(argument Expression) Expression         { return AcosOf(argument) }
func CreateArcTangent(argument Expression) Expression        { return AtanOf(argument) }
func CreateHyperbolicSine(argument Expression) Expression    { return SinhOf(argument) }
func CreateHyperbolicCosine(argument Expression) Expression  { return CoshOf(argument) }
func CreateHyperbolicTangent(argument Expression) Expression { return TanhOf(argument) }

func SimplifyExpression(expression Expression) Expression       { return Simplify(expression) }
func DeeplySimplifyExpression(expression Expression) Expression { return DeepSimplify(expression) }
func DifferentiateExpression(expression Expression, variableName string) Expression {
	return Diff(expression, variableName)
}
func DifferentiateExpressionTwice(expression Expression, variableName string) Expression {
	return Diff2(expression, variableName)
}
func DifferentiateExpressionRepeatedly(expression Expression, variableName string, count int) Expression {
	return DiffN(expression, variableName, count)
}
func SubstituteExpression(expression Expression, variableName string, value Expression) Expression {
	return Sub(expression, variableName, value)
}
func ExpandExpression(expression Expression) Expression { return Expand(expression) }

// SymbolicIntegration returns an antiderivative with an explicit arbitrary constant.
func SymbolicIntegration(expression Expression, variableName string) (Expression, error) {
	antiderivative, succeeded := IntegrateWithConstant(expression, variableName)
	if !succeeded {
		return nil, newSymbolicMathematicsError("symbolic integration", "unsupported integrand", nil)
	}
	return antiderivative, nil
}

// PerformSymbolicIntegration preserves the descriptive full-name integration entry point.
func PerformSymbolicIntegration(expression Expression, variableName string) (Expression, error) {
	return SymbolicIntegration(expression, variableName)
}

// LimitExpression computes a symbolic limit using the existing limit engine.
func LimitExpression(expression Expression, variable *SymbolicVariableNode, point Expression, direction string) (Expression, error) {
	if variable == nil {
		return nil, newSymbolicMathematicsError("limit", "variable must not be nil", nil)
	}
	limitResult := LimitWithDirection(expression, variable.Name(), point, direction)
	if !limitResult.Success {
		return nil, newSymbolicMathematicsError("limit", limitResult.Error, nil)
	}
	return limitResult.Value, nil
}

func TaylorSeriesExpansion(expression Expression, variableName string, center Expression, order int) Expression {
	return TaylorSeries(expression, variableName, center, order)
}
func FreeSymbolicVariables(expression Expression) map[string]struct{} { return FreeSymbols(expression) }
func PolynomialDegree(expression Expression, variableName string) int {
	return Degree(expression, variableName)
}
func PolynomialCoefficients(expression Expression, variableName string) PolyCoeffsResult {
	return PolyCoeffs(expression, variableName)
}

func SolveLinearEquation(a, b Expression) SolveResult       { return SolveLinear(a, b) }
func SolveQuadraticEquation(a, b, c Expression) SolveResult { return SolveQuadraticExact(a, b, c) }
func SolveSymbolicEquation(equation *SymbolicEquation, variableName string) SolveResult {
	return SolveEquation(equation, variableName)
}

func LatexOutput(expression Expression) string { return LaTeX(expression) }
func JavaScriptObjectNotationString(expression Expression) (string, error) {
	return ToJSON(expression)
}
func JavaScriptObjectNotationToExpression(data map[string]interface{}) (Expression, error) {
	return FromJSON(data)
}
func HandleModelContextProtocolToolCall(request ToolRequest) ToolResponse {
	return HandleToolCall(request)
}
func ModelContextProtocolToolSpecification() string { return MCPToolSpec() }

// CollectLikeTerms performs deterministic term collection.
func CollectLikeTerms(expression Expression) Expression {
	return expression.Simplify()
}

// CancelCommonFactors simplifies a rational expression if it can recognize a quotient.
func CancelCommonFactors(rationalExpression Expression) Expression {
	numeratorExpression, denominatorExpression, isQuotient := extractQuotient(rationalExpression)
	if !isQuotient {
		return rationalExpression.Simplify()
	}
	return Cancel(numeratorExpression, denominatorExpression)
}

// FactorExpression factors an expression using the first free variable in
// lexicographic order when available.
func FactorExpression(expression Expression) Expression {
	freeVariableSet := FreeSymbols(expression)
	if len(freeVariableSet) == 0 {
		return expression.Simplify()
	}
	freeVariableNames := make([]string, 0, len(freeVariableSet))
	for freeVariableName := range freeVariableSet {
		freeVariableNames = append(freeVariableNames, freeVariableName)
	}
	sort.Strings(freeVariableNames)
	factorResult := Factor(expression, freeVariableNames[0])
	return MulOf(factorResult.Factors...).Simplify()
}

// PartialFractionDecomposition rewrites a rational expression as a sum of simpler fractions.
func PartialFractionDecomposition(rationalExpression Expression, variable *SymbolicVariableNode) Expression {
	if variable == nil {
		return rationalExpression.Simplify()
	}
	numeratorExpression, denominatorExpression, isQuotient := extractQuotient(rationalExpression)
	if !isQuotient {
		return rationalExpression.Simplify()
	}
	partialFractionResult := Apart(numeratorExpression, denominatorExpression, variable.Name())
	if len(partialFractionResult.Terms) == 0 {
		return rationalExpression.Simplify()
	}
	return AddOf(partialFractionResult.Terms...).Simplify()
}

// AsciiPrettyPrintExpression is a compatibility wrapper for AsciiPrettyPrint.
func AsciiPrettyPrintExpression(expression Expression) string { return AsciiPrettyPrint(expression) }

// Solve provides a method-oriented equation solver entry point.
func (equation *Equation) Solve(variableName string) SolveResult {
	return SolveEquation(equation, variableName)
}
