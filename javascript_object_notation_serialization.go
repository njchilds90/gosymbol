package gosymbol

import (
	"encoding/json"
	"fmt"
	"math/big"
)

// ============================================================
// JSON Serialization
// ============================================================

func ToJSON(e Expr) (string, error) {
	b, err := json.Marshal(e.toJSON())
	return string(b), err
}

func FromJSON(data map[string]interface{}) (Expr, error) {
	if data == nil {
		return nil, fmt.Errorf("expression must be an object")
	}
	typAny, ok := data["type"]
	if !ok {
		return nil, fmt.Errorf("missing 'type' field")
	}
	typ, ok := typAny.(string)
	if !ok || typ == "" {
		return nil, fmt.Errorf("field 'type' must be a non-empty string")
	}

	subObj := func(field string) (map[string]interface{}, error) {
		v, ok := data[field]
		if !ok {
			return nil, fmt.Errorf("%s: missing %q", typ, field)
		}
		m, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: %q must be an object", typ, field)
		}
		return m, nil
	}

	subObjArray := func(field string) ([]map[string]interface{}, error) {
		v, ok := data[field]
		if !ok {
			return nil, fmt.Errorf("%s: missing %q", typ, field)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("%s: %q must be an array", typ, field)
		}
		out := make([]map[string]interface{}, len(raw))
		for i, it := range raw {
			m, ok := it.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("%s: %q[%d] must be an object", typ, field, i)
			}
			out[i] = m
		}
		return out, nil
	}

	subString := func(field string) (string, error) {
		v, ok := data[field]
		if !ok {
			return "", fmt.Errorf("%s: missing %q", typ, field)
		}
		s, ok := v.(string)
		if !ok || s == "" {
			return "", fmt.Errorf("%s: %q must be a non-empty string", typ, field)
		}
		return s, nil
	}

	subNumberAsInt := func(field string) (int, error) {
		v, ok := data[field]
		if !ok {
			return 0, fmt.Errorf("%s: missing %q", typ, field)
		}
		n, ok := v.(float64)
		if !ok {
			return 0, fmt.Errorf("%s: %q must be a number", typ, field)
		}
		return int(n), nil
	}

	switch typ {
	case "num":
		valAny, ok := data["value"]
		if !ok {
			return nil, fmt.Errorf("num: missing 'value'")
		}
		val, ok := valAny.(string)
		if !ok || val == "" {
			return nil, fmt.Errorf("num: 'value' must be a non-empty string")
		}
		r := new(big.Rat)
		if _, ok := r.SetString(val); !ok {
			return nil, fmt.Errorf("invalid num value: %s", val)
		}
		return &Num{val: r}, nil

	case "sym":
		name, err := subString("name")
		if err != nil {
			return nil, err
		}
		sym := S(name)
		if rawAssumptions, ok := data["assumptions"].(map[string]interface{}); ok {
			sym.assumptions = Assumptions{
				Real:     boolValue(rawAssumptions["real"]),
				Positive: boolValue(rawAssumptions["positive"]),
				Negative: boolValue(rawAssumptions["negative"]),
				Integer:  boolValue(rawAssumptions["integer"]),
				NonZero:  boolValue(rawAssumptions["non_zero"]),
				Natural:  boolValue(rawAssumptions["natural"]),
			}
		}
		return sym, nil

	case "constant":
		name, err := subString("name")
		if err != nil {
			return nil, err
		}
		return CreateConstantNode(name), nil

	case "add":
		objs, err := subObjArray("terms")
		if err != nil {
			return nil, err
		}
		terms := make([]Expr, len(objs))
		for i, o := range objs {
			e, err := FromJSON(o)
			if err != nil {
				return nil, fmt.Errorf("add: terms[%d]: %w", i, err)
			}
			terms[i] = e
		}
		return AddOf(terms...), nil

	case "mul":
		objs, err := subObjArray("factors")
		if err != nil {
			return nil, err
		}
		factors := make([]Expr, len(objs))
		for i, o := range objs {
			e, err := FromJSON(o)
			if err != nil {
				return nil, fmt.Errorf("mul: factors[%d]: %w", i, err)
			}
			factors[i] = e
		}
		return MulOf(factors...), nil

	case "pow":
		baseM, err := subObj("base")
		if err != nil {
			return nil, err
		}
		expM, err := subObj("exp")
		if err != nil {
			return nil, err
		}
		base, err := FromJSON(baseM)
		if err != nil {
			return nil, fmt.Errorf("pow: base: %w", err)
		}
		exp, err := FromJSON(expM)
		if err != nil {
			return nil, fmt.Errorf("pow: exp: %w", err)
		}
		return PowOf(base, exp), nil

	case "func":
		name, err := subString("name")
		if err != nil {
			return nil, err
		}
		argM, err := subObj("arg")
		if err != nil {
			return nil, err
		}
		arg, err := FromJSON(argM)
		if err != nil {
			return nil, fmt.Errorf("func: arg: %w", err)
		}
		return funcOf(name, arg).Simplify(), nil

	case "bigo":
		v, err := subString("var")
		if err != nil {
			return nil, err
		}
		order, err := subNumberAsInt("order")
		if err != nil {
			return nil, err
		}
		return OTerm(v, order), nil

	case "piecewise":
		rawCases, ok := data["cases"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("piecewise: 'cases' must be an array")
		}
		cases := make([]PiecewiseCase, len(rawCases))
		for caseIndex, rawCase := range rawCases {
			caseMap, ok := rawCase.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("piecewise: cases[%d] must be an object", caseIndex)
			}
			condition, ok := caseMap["condition"].(string)
			if !ok {
				return nil, fmt.Errorf("piecewise: cases[%d].condition must be a string", caseIndex)
			}
			expressionMap, ok := caseMap["expression"].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("piecewise: cases[%d].expression must be an object", caseIndex)
			}
			expression, err := FromJSON(expressionMap)
			if err != nil {
				return nil, err
			}
			cases[caseIndex] = PiecewiseCase{Condition: condition, Expression: expression}
		}
		var defaultExpression Expr
		if rawDefault, hasDefault := data["default"].(map[string]interface{}); hasDefault {
			parsedDefault, err := FromJSON(rawDefault)
			if err != nil {
				return nil, err
			}
			defaultExpression = parsedDefault
		}
		return CreatePiecewiseExpression(cases, defaultExpression), nil
	}
	return nil, fmt.Errorf("unknown expression type: %s", typ)
}

func boolValue(v interface{}) bool {
	b, _ := v.(bool)
	return b
}
