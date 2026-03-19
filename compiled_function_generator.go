package gosymbol

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// LambdifyToGoFunction converts an expression into a native Go closure over a
// variable-value map.
func LambdifyToGoFunction(expression Expr) (func(map[string]float64) float64, error) {
	if expression == nil {
		return nil, newSymbolicMathematicsError("lambdify", "expression must not be nil", nil)
	}
	return func(variableValues map[string]float64) float64 {
		value, evaluationError := evaluateExpressionAsFloatingPoint(expression, variableValues)
		if evaluationError != nil {
			return math.NaN()
		}
		return value
	}, nil
}

// GenerateRunnableGoFunctionSource renders a compilable Go function for the expression.
func GenerateRunnableGoFunctionSource(expression Expr, functionName string) (string, error) {
	if expression == nil {
		return "", newSymbolicMathematicsError("lambdify", "expression must not be nil", nil)
	}
	if functionName == "" {
		functionName = "EvaluateSymbolicExpression"
	}
	requiredVariables := FreeSymbols(expression)
	orderedVariableNames := make([]string, 0, len(requiredVariables))
	for requiredVariableName := range requiredVariables {
		orderedVariableNames = append(orderedVariableNames, requiredVariableName)
	}
	sort.Strings(orderedVariableNames)
	var builder strings.Builder
	builder.WriteString("package main\n\nimport \"math\"\n\n")
	builder.WriteString("func ")
	builder.WriteString(functionName)
	builder.WriteString("(")
	for variableIndex, variableName := range orderedVariableNames {
		if variableIndex > 0 {
			builder.WriteString(", ")
		}
		builder.WriteString(variableName)
		builder.WriteString(" float64")
	}
	builder.WriteString(") float64 {\n\treturn ")
	goExpressionSource, sourceError := renderExpressionAsGoSource(expression)
	if sourceError != nil {
		return "", sourceError
	}
	builder.WriteString(goExpressionSource)
	builder.WriteString("\n}\n")
	return builder.String(), nil
}

// AsciiPrettyPrint renders an expression tree in a console-friendly format.
func AsciiPrettyPrint(expression Expr) string {
	return buildAsciiPrettyPrint(expression, "", true)
}

func buildAsciiPrettyPrint(expression Expr, indentation string, isLastChild bool) string {
	if expression == nil {
		return indentation + "nil\n"
	}
	linePrefix := "├── "
	nextIndentation := indentation + "│   "
	if isLastChild {
		linePrefix = "└── "
		nextIndentation = indentation + "    "
	}
	var builder strings.Builder
	builder.WriteString(indentation)
	builder.WriteString(linePrefix)
	builder.WriteString(expression.exprType())
	builder.WriteString(": ")
	builder.WriteString(expression.String())
	builder.WriteString("\n")

	children := expressionChildren(expression)
	for childIndex, childExpression := range children {
		builder.WriteString(buildAsciiPrettyPrint(childExpression, nextIndentation, childIndex == len(children)-1))
	}
	return builder.String()
}

func expressionChildren(expression Expr) []Expr {
	switch typedExpression := expression.(type) {
	case *Add:
		return typedExpression.terms
	case *Mul:
		return typedExpression.factors
	case *Pow:
		return []Expr{typedExpression.base, typedExpression.exp}
	case *Func:
		return []Expr{typedExpression.arg}
	case *PiecewiseExpression:
		children := make([]Expr, 0, len(typedExpression.Cases)+1)
		for _, piecewiseCase := range typedExpression.Cases {
			children = append(children, piecewiseCase.Expression)
		}
		if typedExpression.DefaultExpression != nil {
			children = append(children, typedExpression.DefaultExpression)
		}
		return children
	default:
		return nil
	}
}

func evaluateExpressionAsFloatingPoint(expression Expr, variableValues map[string]float64) (float64, error) {
	switch typedExpression := expression.(type) {
	case *Num:
		return typedExpression.Float64(), nil
	case *ConstantNode:
		return math.NaN(), newSymbolicMathematicsError("evaluation", "constant nodes require symbolic treatment", nil)
	case *Sym:
		value, hasValue := variableValues[typedExpression.Name()]
		if !hasValue {
			return math.NaN(), newSymbolicMathematicsError("evaluation", fmt.Sprintf("missing value for variable %s", typedExpression.Name()), nil)
		}
		return value, nil
	case *Add:
		total := 0.0
		for _, term := range typedExpression.terms {
			termValue, evaluationError := evaluateExpressionAsFloatingPoint(term, variableValues)
			if evaluationError != nil {
				return math.NaN(), evaluationError
			}
			total += termValue
		}
		return total, nil
	case *Mul:
		total := 1.0
		for _, factor := range typedExpression.factors {
			factorValue, evaluationError := evaluateExpressionAsFloatingPoint(factor, variableValues)
			if evaluationError != nil {
				return math.NaN(), evaluationError
			}
			total *= factorValue
		}
		return total, nil
	case *Pow:
		baseValue, baseError := evaluateExpressionAsFloatingPoint(typedExpression.base, variableValues)
		if baseError != nil {
			return math.NaN(), baseError
		}
		exponentValue, exponentError := evaluateExpressionAsFloatingPoint(typedExpression.exp, variableValues)
		if exponentError != nil {
			return math.NaN(), exponentError
		}
		return math.Pow(baseValue, exponentValue), nil
	case *Func:
		argumentValue, argumentError := evaluateExpressionAsFloatingPoint(typedExpression.arg, variableValues)
		if argumentError != nil {
			return math.NaN(), argumentError
		}
		switch typedExpression.name {
		case "sin":
			return math.Sin(argumentValue), nil
		case "cos":
			return math.Cos(argumentValue), nil
		case "tan":
			return math.Tan(argumentValue), nil
		case "exp":
			return math.Exp(argumentValue), nil
		case "ln":
			return math.Log(argumentValue), nil
		case "abs":
			return math.Abs(argumentValue), nil
		case "asin":
			return math.Asin(argumentValue), nil
		case "acos":
			return math.Acos(argumentValue), nil
		case "atan":
			return math.Atan(argumentValue), nil
		case "sinh":
			return math.Sinh(argumentValue), nil
		case "cosh":
			return math.Cosh(argumentValue), nil
		case "tanh":
			return math.Tanh(argumentValue), nil
		case "asinh":
			return math.Asinh(argumentValue), nil
		case "acosh":
			return math.Acosh(argumentValue), nil
		case "atanh":
			return math.Atanh(argumentValue), nil
		default:
			return math.NaN(), newSymbolicMathematicsError("evaluation", "unsupported function "+typedExpression.name, nil)
		}
	case *PiecewiseExpression:
		return math.NaN(), newSymbolicMathematicsError("evaluation", "piecewise evaluation requires an external condition resolver", nil)
	default:
		return math.NaN(), newSymbolicMathematicsError("evaluation", "unsupported expression type "+expression.exprType(), nil)
	}
}

func renderExpressionAsGoSource(expression Expr) (string, error) {
	switch typedExpression := expression.(type) {
	case *Num:
		return typedExpression.String(), nil
	case *Sym:
		return typedExpression.Name(), nil
	case *Add:
		termSources := make([]string, len(typedExpression.terms))
		for termIndex, term := range typedExpression.terms {
			termSource, termError := renderExpressionAsGoSource(term)
			if termError != nil {
				return "", termError
			}
			termSources[termIndex] = "(" + termSource + ")"
		}
		return strings.Join(termSources, " + "), nil
	case *Mul:
		factorSources := make([]string, len(typedExpression.factors))
		for factorIndex, factor := range typedExpression.factors {
			factorSource, factorError := renderExpressionAsGoSource(factor)
			if factorError != nil {
				return "", factorError
			}
			factorSources[factorIndex] = "(" + factorSource + ")"
		}
		return strings.Join(factorSources, " * "), nil
	case *Pow:
		baseSource, baseError := renderExpressionAsGoSource(typedExpression.base)
		if baseError != nil {
			return "", baseError
		}
		exponentSource, exponentError := renderExpressionAsGoSource(typedExpression.exp)
		if exponentError != nil {
			return "", exponentError
		}
		return "math.Pow(" + baseSource + ", " + exponentSource + ")", nil
	case *Func:
		argumentSource, argumentError := renderExpressionAsGoSource(typedExpression.arg)
		if argumentError != nil {
			return "", argumentError
		}
		switch typedExpression.name {
		case "sin":
			return "math.Sin(" + argumentSource + ")", nil
		case "cos":
			return "math.Cos(" + argumentSource + ")", nil
		case "tan":
			return "math.Tan(" + argumentSource + ")", nil
		case "exp":
			return "math.Exp(" + argumentSource + ")", nil
		case "ln":
			return "math.Log(" + argumentSource + ")", nil
		case "abs":
			return "math.Abs(" + argumentSource + ")", nil
		case "asin":
			return "math.Asin(" + argumentSource + ")", nil
		case "acos":
			return "math.Acos(" + argumentSource + ")", nil
		case "atan":
			return "math.Atan(" + argumentSource + ")", nil
		case "sinh":
			return "math.Sinh(" + argumentSource + ")", nil
		case "cosh":
			return "math.Cosh(" + argumentSource + ")", nil
		case "tanh":
			return "math.Tanh(" + argumentSource + ")", nil
		case "asinh":
			return "math.Asinh(" + argumentSource + ")", nil
		case "acosh":
			return "math.Acosh(" + argumentSource + ")", nil
		case "atanh":
			return "math.Atanh(" + argumentSource + ")", nil
		default:
			return "", newSymbolicMathematicsError("lambdify", "unsupported function "+typedExpression.name, nil)
		}
	default:
		return "", newSymbolicMathematicsError("lambdify", "unsupported expression type "+expression.exprType(), nil)
	}
}
