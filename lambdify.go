package gosymbol

import (
	"fmt"
	"math"
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
