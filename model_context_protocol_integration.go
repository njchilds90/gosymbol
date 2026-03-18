package gosymbol

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ============================================================
// MCP Tool Interface
// ============================================================

type ToolRequest struct {
	Tool   string                 `json:"tool"`
	Params map[string]interface{} `json:"params"`
}

type ToolResponse struct {
	Result interface{} `json:"result,omitempty"`
	LaTeX  string      `json:"latex,omitempty"`
	String string      `json:"string,omitempty"`
	Error  string      `json:"error,omitempty"`
}

func HandleToolCall(req ToolRequest) ToolResponse {
	getExpr := func(key string) (Expr, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		if raw, ok := v.(string); ok {
			return ParseWithError(raw)
		}
		val, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid type for param %s: want expression object or infix string", key)
		}
		return FromJSON(val)
	}
	getString := func(key string) (string, error) {
		v, ok := req.Params[key]
		if !ok {
			return "", fmt.Errorf("missing param: %s", key)
		}
		s, ok := v.(string)
		if !ok {
			return "", fmt.Errorf("param %s must be a string", key)
		}
		return s, nil
	}
	getStrings := func(key string) ([]string, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be array", key)
		}
		result := make([]string, len(raw))
		for i, r := range raw {
			s, ok := r.(string)
			if !ok {
				return nil, fmt.Errorf("param %s[%d] must be string", key, i)
			}
			result[i] = s
		}
		return result, nil
	}
	getExprList := func(key string) ([]Expr, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.([]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be array", key)
		}
		result := make([]Expr, len(raw))
		for i, r := range raw {
			m, ok := r.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("param %s[%d] must be expression object", key, i)
			}
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			result[i] = e
		}
		return result, nil
	}
	getMatrix := func(key string) (*Matrix, error) {
		v, ok := req.Params[key]
		if !ok {
			return nil, fmt.Errorf("missing param: %s", key)
		}
		raw, ok := v.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("param %s must be matrix object", key)
		}

		rowsF, ok := raw["rows"].(float64)
		if !ok {
			return nil, fmt.Errorf("matrix.rows must be a number")
		}
		colsF, ok := raw["cols"].(float64)
		if !ok {
			return nil, fmt.Errorf("matrix.cols must be a number")
		}
		rows, cols := int(rowsF), int(colsF)
		if rows <= 0 || cols <= 0 {
			return nil, fmt.Errorf("matrix dimensions must be positive")
		}

		entriesRawAny, ok := raw["entries"]
		if !ok {
			return nil, fmt.Errorf("matrix.entries missing")
		}
		entriesRaw, ok := entriesRawAny.([]interface{})
		if !ok {
			return nil, fmt.Errorf("matrix.entries must be an array")
		}
		if len(entriesRaw) != rows*cols {
			return nil, fmt.Errorf("matrix entries count mismatch")
		}
		entries := make([]Expr, rows*cols)
		for i, er := range entriesRaw {
			m, ok := er.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("matrix entry %d must be expression", i)
			}
			e, err := FromJSON(m)
			if err != nil {
				return nil, err
			}
			entries[i] = e
		}
		return MatrixFromSlice(rows, cols, entries), nil
	}
	respond := func(e Expr) ToolResponse {
		return ToolResponse{Result: e.toJSON(), LaTeX: LaTeX(e), String: String(e)}
	}
	respondMatrix := func(mat *Matrix) ToolResponse {
		return ToolResponse{
			Result: map[string]interface{}{"rows": mat.rows, "cols": mat.cols, "string": mat.String()},
			LaTeX:  mat.LaTeX(),
			String: mat.String(),
		}
	}
	solvesTool := func(res SolveResult) ToolResponse {
		if res.Error != "" && len(res.Solutions) == 0 {
			return ToolResponse{Error: res.Error}
		}
		strs := make([]string, len(res.Solutions))
		for i, s := range res.Solutions {
			strs[i] = String(s)
		}
		resp := ToolResponse{Result: strs, String: strings.Join(strs, ", ")}
		if res.Error != "" {
			resp.Error = res.Error
		}
		return resp
	}

	switch req.Tool {
	case "parse":
		input, err := getString("input")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		e, err := ParseWithError(input)
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(e)

	case "simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Simplify(e))

	case "deep_simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(DeepSimplify(e))

	case "trig_simplify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(TrigSimplify(e))

	case "canonicalize":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Canonicalize(e))

	case "diff":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Diff(e, v))

	case "diff2":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Diff2(e, v))

	case "diffn":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		nAny, ok := req.Params["n"]
		if !ok {
			return ToolResponse{Error: "missing param: n"}
		}
		nF, ok := nAny.(float64)
		if !ok {
			return ToolResponse{Error: "param n must be a number"}
		}
		n := int(nF)
		if n < 0 {
			return ToolResponse{Error: "param n must be >= 0"}
		}
		return respond(DiffN(e, v, n))

	case "pdiff":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(PDiff(e, v))

	case "gradient":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		grad := Gradient(e, vars)
		strs := make([]string, len(grad))
		latexStrs := make([]string, len(grad))
		for i, g := range grad {
			strs[i] = String(g)
			latexStrs[i] = LaTeX(g)
		}
		return ToolResponse{
			Result: strs,
			String: "[" + strings.Join(strs, ", ") + "]",
			LaTeX:  "[" + strings.Join(latexStrs, ", ") + "]",
		}

	case "integrate":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		result, ok := Integrate(e, v)
		if !ok {
			return ToolResponse{Error: "integration failed: unsupported form"}
		}
		return respond(result)

	case "integrate_with_constant":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		result, ok := IntegrateWithConstant(e, v)
		if !ok {
			return ToolResponse{Error: "integration failed: unsupported form"}
		}
		return respond(result)

	case "definite_integrate":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		aF, ok := req.Params["a"].(float64)
		if !ok {
			return ToolResponse{Error: "param a must be a number"}
		}
		bF, ok := req.Params["b"].(float64)
		if !ok {
			return ToolResponse{Error: "param b must be a number"}
		}
		result := DefiniteIntegrate(e, v, aF, bF)
		return ToolResponse{Result: result, String: fmt.Sprintf("%.10g", result)}

	case "expand":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Expand(e))

	case "substitute":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		val, err := getExpr("value")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Sub(e, v, val))

	case "to_latex":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{LaTeX: LaTeX(e), String: String(e)}

	case "pretty_print":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		formatted := AsciiPrettyPrint(e)
		return ToolResponse{Result: formatted, String: formatted}

	case "lambdify":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		_, err = LambdifyToGoFunction(e)
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		freeVariables := FreeSymbols(e)
		names := make([]string, 0, len(freeVariables))
		for freeVariableName := range freeVariables {
			names = append(names, freeVariableName)
		}
		sort.Strings(names)
		return ToolResponse{
			Result: map[string]interface{}{
				"variables":  names,
				"expression": String(e),
			},
			String: "lambdify ready for variables: " + strings.Join(names, ", "),
		}

	case "free_symbols":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		syms := FreeSymbols(e)
		names := make([]string, 0, len(syms))
		for n := range syms {
			names = append(names, n)
		}
		sort.Strings(names)
		return ToolResponse{Result: names}

	case "degree":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{Result: Degree(e, v)}

	case "poly_coeffs":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		coeffs := PolyCoeffs(e, v)
		result := map[string]string{}
		for deg, c := range coeffs {
			result[fmt.Sprintf("%d", deg)] = String(c)
		}
		return ToolResponse{Result: result}

	case "collect":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Collect(e, v))

	case "cancel":
		num, err := getExpr("num")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		denom, err := getExpr("denom")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Cancel(num, denom))

	case "apart":
		num, err := getExpr("num")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		denom, err := getExpr("denom")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		result := Apart(num, denom, v)
		strs := make([]string, len(result.Terms))
		for i, t := range result.Terms {
			strs[i] = String(t)
		}
		return ToolResponse{Result: strs, String: strings.Join(strs, " + "), Error: result.Error}

	case "factor":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		fr := Factor(e, v)
		strs := make([]string, len(fr.Factors))
		for i, f := range fr.Factors {
			strs[i] = String(f)
		}
		return ToolResponse{Result: strs, String: strings.Join(strs, " * ")}

	case "solve_linear":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		res := SolveLinear(a, b)
		if res.Error != "" {
			return ToolResponse{Error: res.Error}
		}
		sols := make([]map[string]interface{}, len(res.Solutions))
		latexSols := make([]string, len(res.Solutions))
		strSols := make([]string, len(res.Solutions))
		for i, s := range res.Solutions {
			sols[i] = s.toJSON()
			latexSols[i] = LaTeX(s)
			strSols[i] = String(s)
		}
		return ToolResponse{
			Result: sols,
			LaTeX:  strings.Join(latexSols, ", "),
			String: strings.Join(strSols, ", "),
		}

	case "solve_quadratic":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c, err := getExpr("c")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return solvesTool(SolveQuadratic(a, b, c))

	case "solve_quadratic_exact":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c, err := getExpr("c")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return solvesTool(SolveQuadraticExact(a, b, c))

	case "solve_equation":
		lhs, err := getExpr("lhs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		rhs, err := getExpr("rhs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return solvesTool(SolveEquation(Eq(lhs, rhs), v))

	case "solve_cubic":
		a, err := getExpr("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b, err := getExpr("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c, err := getExpr("c")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		d, err := getExpr("d")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return solvesTool(SolveCubic(a, b, c, d))

	case "solve_polynomial_newton":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		rangeF, _ := req.Params["range"].(float64)
		tolF, _ := req.Params["tol"].(float64)
		iterF, _ := req.Params["max_iter"].(float64)
		return solvesTool(SolvePolynomialNewton(e, v, rangeF, tolF, int(iterF)))

	case "solve_system_2x2":
		a1, err := getExpr("a1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b1, err := getExpr("b1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c1, err := getExpr("c1")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		a2, err := getExpr("a2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		b2, err := getExpr("b2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		c2, err := getExpr("c2")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		x, y, err := SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2)
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return ToolResponse{
			Result: map[string]string{"x": String(x), "y": String(y)},
			String: "x=" + String(x) + ", y=" + String(y),
			LaTeX:  "x=" + LaTeX(x) + ",\\ y=" + LaTeX(y),
		}

	case "limit":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		pt, err := getExpr("point")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		res := Limit(e, v, pt)
		if !res.Success {
			return ToolResponse{Error: res.Error}
		}
		return respond(res.Value)

	case "taylor":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		aExpr, err := getExpr("around")
		if err != nil {
			aExpr = N(0)
		}
		orderFloat, _ := req.Params["order"].(float64)
		order := int(orderFloat)
		if order <= 0 {
			order = 5
		}
		return respond(TaylorSeries(e, v, aExpr, order))

	case "taylor_remainder":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		aExpr, err := getExpr("around")
		if err != nil {
			aExpr = N(0)
		}
		orderFloat, _ := req.Params["order"].(float64)
		order := int(orderFloat)
		if order <= 0 {
			order = 5
		}
		return respond(TaylorSeriesWithRemainder(e, v, aExpr, order))

	case "maclaurin":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		v, err := getString("var")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		orderFloat, _ := req.Params["order"].(float64)
		order := int(orderFloat)
		if order <= 0 {
			order = 5
		}
		return respond(MaclaurinSeries(e, v, order))

	case "jacobian":
		exprs, err := getExprList("exprs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(Jacobian(exprs, vars))

	case "hessian":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(Hessian(e, vars))

	case "laplacian":
		e, err := getExpr("expr")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Laplacian(e, vars))

	case "divergence":
		exprs, err := getExprList("exprs")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		vars, err := getStrings("vars")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(Divergence(exprs, vars))

	case "matrix_det":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(m.Det())

	case "matrix_inv":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		inv, err := m.Inverse()
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(inv)

	case "matrix_trace":
		m, err := getMatrix("matrix")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respond(m.Trace())

	case "matrix_mul":
		m1, err := getMatrix("a")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		m2, err := getMatrix("b")
		if err != nil {
			return ToolResponse{Error: err.Error()}
		}
		return respondMatrix(m1.MatMul(m2))

	case "mcp_spec":
		return ToolResponse{Result: MCPToolSpec(), String: "MCP tool specification"}
	}

	return ToolResponse{Error: fmt.Sprintf("unknown tool: %s", req.Tool)}
}

// ============================================================
// MCP spec
// ============================================================

func MCPToolSpec() string {
	tools := []map[string]interface{}{
		ts("parse", "Parse an infix string expression into JSON/tree form", []string{"input"}, map[string]string{"input": "string"}),
		ts("simplify", "Simplify a symbolic expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("deep_simplify", "Apply multiple simplification passes including trig identities", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("trig_simplify", "Apply trig identities (sin²+cos²=1, exp(ln(x))=x, etc.)", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("canonicalize", "Expand and canonicalize expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("expand", "Algebraically expand expression", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("collect", "Collect terms by powers of variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("cancel", "Simplify rational num/denom", []string{"num", "denom"}, map[string]string{"num": "object", "denom": "object"}),
		ts("factor", "Factor polynomial in variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("apart", "Partial fraction decomposition", []string{"num", "denom", "var"}, map[string]string{"num": "object", "denom": "object", "var": "string"}),
		ts("substitute", "Substitute var with value", []string{"expr", "var", "value"}, map[string]string{"expr": "object", "var": "string", "value": "object"}),
		ts("to_latex", "Convert to LaTeX", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("pretty_print", "Render a console-friendly symbolic tree", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("lambdify", "Validate conversion to a Go closure and return required variables", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("free_symbols", "Return free symbol names", []string{"expr"}, map[string]string{"expr": "object"}),
		ts("diff", "First derivative d/dx", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("diff2", "Second derivative d²/dx²", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("diffn", "nth derivative. Requires n (int)", []string{"expr", "var", "n"}, map[string]string{"expr": "object", "var": "string", "n": "integer"}),
		ts("pdiff", "Partial derivative ∂/∂var", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("gradient", "Gradient vector ∇f. Requires vars (string[])", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("jacobian", "Jacobian matrix. Requires exprs (array) and vars (array)", []string{"exprs", "vars"}, map[string]string{"exprs": "array", "vars": "array"}),
		ts("hessian", "Hessian matrix of second partials", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("laplacian", "Laplacian ∇²f", []string{"expr", "vars"}, map[string]string{"expr": "object", "vars": "array"}),
		ts("divergence", "Divergence ∇·F", []string{"exprs", "vars"}, map[string]string{"exprs": "array", "vars": "array"}),
		ts("integrate", "Symbolic integration (rule-based)", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("integrate_with_constant", "Symbolic integration with explicit +C", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("definite_integrate", "Numerical ∫_a^b. Requires a,b (numbers)", []string{"expr", "var", "a", "b"}, map[string]string{"expr": "object", "var": "string", "a": "number", "b": "number"}),
		ts("taylor", "Taylor series", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "around": "object", "order": "integer"}),
		ts("taylor_remainder", "Taylor series with BigO remainder", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "order": "integer"}),
		ts("maclaurin", "Maclaurin series (Taylor around 0)", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string", "order": "integer"}),
		ts("degree", "Polynomial degree in variable", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("poly_coeffs", "Extract polynomial coefficients by degree", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("limit", "lim_{var->point} expr", []string{"expr", "var", "point"}, map[string]string{"expr": "object", "var": "string", "point": "object"}),
		ts("solve_linear", "Solve a*x+b=0 exactly", []string{"a", "b"}, map[string]string{"a": "object", "b": "object"}),
		ts("solve_quadratic", "Solve a*x²+b*x+c=0 (float)", []string{"a", "b", "c"}, map[string]string{"a": "object", "b": "object", "c": "object"}),
		ts("solve_quadratic_exact", "Solve a*x²+b*x+c=0 with exact roots when possible", []string{"a", "b", "c"}, map[string]string{"a": "object", "b": "object", "c": "object"}),
		ts("solve_equation", "Solve lhs = rhs for a variable", []string{"lhs", "rhs", "var"}, map[string]string{"lhs": "object", "rhs": "object", "var": "string"}),
		ts("solve_cubic", "Solve a*x³+b*x²+c*x+d=0 (Cardano)", []string{"a", "b", "c", "d"}, map[string]string{}),
		ts("solve_polynomial_newton", "Numerical root finding. Optional: range, tol, max_iter", []string{"expr", "var"}, map[string]string{"expr": "object", "var": "string"}),
		ts("solve_system_2x2", "2×2 linear system: a1*x+b1*y=c1, a2*x+b2*y=c2", []string{"a1", "b1", "c1", "a2", "b2", "c2"}, map[string]string{}),
		ts("matrix_det", "Matrix det. matrix={rows,cols,entries:[expr,...]}", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_inv", "Symbolic matrix inverse", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_trace", "Matrix trace", []string{"matrix"}, map[string]string{"matrix": "object"}),
		ts("matrix_mul", "Matrix multiply a*b", []string{"a", "b"}, map[string]string{"a": "object", "b": "object"}),
		ts("mcp_spec", "Return this tool schema", []string{}, map[string]string{}),
	}
	spec := map[string]interface{}{"tools": tools}
	b, _ := json.MarshalIndent(spec, "", "  ")
	return string(b)
}

func ts(name, description string, required []string, props map[string]string) map[string]interface{} {
	properties := map[string]interface{}{}
	for k, typ := range props {
		properties[k] = map[string]interface{}{"type": typ}
	}
	return map[string]interface{}{
		"name":        name,
		"description": description,
		"inputSchema": map[string]interface{}{
			"type":       "object",
			"properties": properties,
			"required":   required,
		},
	}
}
