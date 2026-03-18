package gosymbol

import "strings"

// PiecewiseCase stores a condition and the expression used when it matches.
type PiecewiseCase struct {
	Condition  string
	Expression Expr
}

// PiecewiseExpression represents a deterministic piecewise symbolic expression.
type PiecewiseExpression struct {
	Cases             []PiecewiseCase
	DefaultExpression Expr
}

// CreatePiecewiseExpression constructs a piecewise expression.
func CreatePiecewiseExpression(cases []PiecewiseCase, defaultExpression Expr) *PiecewiseExpression {
	return &PiecewiseExpression{Cases: cases, DefaultExpression: defaultExpression}
}

func (piecewiseExpression *PiecewiseExpression) Simplify() Expr {
	simplifiedCases := make([]PiecewiseCase, len(piecewiseExpression.Cases))
	for caseIndex, piecewiseCase := range piecewiseExpression.Cases {
		simplifiedCases[caseIndex] = PiecewiseCase{
			Condition:  piecewiseCase.Condition,
			Expression: piecewiseCase.Expression.Simplify(),
		}
	}
	defaultExpression := Expr(nil)
	if piecewiseExpression.DefaultExpression != nil {
		defaultExpression = piecewiseExpression.DefaultExpression.Simplify()
	}
	return &PiecewiseExpression{
		Cases:             simplifiedCases,
		DefaultExpression: defaultExpression,
	}
}

func (piecewiseExpression *PiecewiseExpression) String() string {
	parts := make([]string, 0, len(piecewiseExpression.Cases)+1)
	for _, piecewiseCase := range piecewiseExpression.Cases {
		parts = append(parts, piecewiseCase.Condition+" => "+piecewiseCase.Expression.String())
	}
	if piecewiseExpression.DefaultExpression != nil {
		parts = append(parts, "otherwise => "+piecewiseExpression.DefaultExpression.String())
	}
	return "piecewise(" + strings.Join(parts, "; ") + ")"
}

func (piecewiseExpression *PiecewiseExpression) LaTeX() string {
	return `\operatorname{piecewise}\left(` + piecewiseExpression.String() + `\right)`
}

func (piecewiseExpression *PiecewiseExpression) Sub(variableName string, value Expr) Expr {
	updatedCases := make([]PiecewiseCase, len(piecewiseExpression.Cases))
	for caseIndex, piecewiseCase := range piecewiseExpression.Cases {
		updatedCases[caseIndex] = PiecewiseCase{
			Condition:  piecewiseCase.Condition,
			Expression: piecewiseCase.Expression.Sub(variableName, value),
		}
	}
	defaultExpression := Expr(nil)
	if piecewiseExpression.DefaultExpression != nil {
		defaultExpression = piecewiseExpression.DefaultExpression.Sub(variableName, value)
	}
	return CreatePiecewiseExpression(updatedCases, defaultExpression).Simplify()
}

func (piecewiseExpression *PiecewiseExpression) Diff(variableName string) Expr {
	updatedCases := make([]PiecewiseCase, len(piecewiseExpression.Cases))
	for caseIndex, piecewiseCase := range piecewiseExpression.Cases {
		updatedCases[caseIndex] = PiecewiseCase{
			Condition:  piecewiseCase.Condition,
			Expression: piecewiseCase.Expression.Diff(variableName),
		}
	}
	defaultExpression := Expr(nil)
	if piecewiseExpression.DefaultExpression != nil {
		defaultExpression = piecewiseExpression.DefaultExpression.Diff(variableName)
	}
	return CreatePiecewiseExpression(updatedCases, defaultExpression).Simplify()
}

func (piecewiseExpression *PiecewiseExpression) Eval() (*Num, bool) { return nil, false }
func (piecewiseExpression *PiecewiseExpression) Equal(other Expr) bool {
	otherPiecewiseExpression, isPiecewiseExpression := other.(*PiecewiseExpression)
	if !isPiecewiseExpression || len(piecewiseExpression.Cases) != len(otherPiecewiseExpression.Cases) {
		return false
	}
	for caseIndex, piecewiseCase := range piecewiseExpression.Cases {
		if piecewiseCase.Condition != otherPiecewiseExpression.Cases[caseIndex].Condition {
			return false
		}
		if !piecewiseCase.Expression.Equal(otherPiecewiseExpression.Cases[caseIndex].Expression) {
			return false
		}
	}
	if piecewiseExpression.DefaultExpression == nil || otherPiecewiseExpression.DefaultExpression == nil {
		return piecewiseExpression.DefaultExpression == nil && otherPiecewiseExpression.DefaultExpression == nil
	}
	return piecewiseExpression.DefaultExpression.Equal(otherPiecewiseExpression.DefaultExpression)
}
func (piecewiseExpression *PiecewiseExpression) exprType() string { return "piecewise" }
func (piecewiseExpression *PiecewiseExpression) toJSON() map[string]interface{} {
	cases := make([]map[string]interface{}, len(piecewiseExpression.Cases))
	for caseIndex, piecewiseCase := range piecewiseExpression.Cases {
		cases[caseIndex] = map[string]interface{}{
			"condition":  piecewiseCase.Condition,
			"expression": piecewiseCase.Expression.toJSON(),
		}
	}
	result := map[string]interface{}{
		"type":  "piecewise",
		"cases": cases,
	}
	if piecewiseExpression.DefaultExpression != nil {
		result["default"] = piecewiseExpression.DefaultExpression.toJSON()
	}
	return result
}
