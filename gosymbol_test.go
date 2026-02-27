package gosymbol_test

import (
	"encoding/json"
	"strings"
	"testing"

	gosymbol "github.com/njchilds90/gosymbol"
)

// ============================================================
// Num tests
// ============================================================

func TestNum_Integer(t *testing.T) {
	n := gosymbol.N(42)
	if n.String() != "42" {
		t.Errorf("want 42, got %s", n.String())
	}
}

func TestNum_Rational(t *testing.T) {
	n := gosymbol.F(1, 3)
	if n.String() != "1/3" {
		t.Errorf("want 1/3, got %s", n.String())
	}
}

func TestNum_LaTeX_Rational(t *testing.T) {
	n := gosymbol.F(2, 5)
	if n.LaTeX() != `\frac{2}{5}` {
		t.Errorf("want \\frac{2}{5}, got %s", n.LaTeX())
	}
}

func TestNum_Diff_IsZero(t *testing.T) {
	result := gosymbol.N(5).Diff("x")
	if gosymbol.String(result) != "0" {
		t.Errorf("d/dx(5) should be 0, got %s", gosymbol.String(result))
	}
}

func TestNum_Eval(t *testing.T) {
	n, ok := gosymbol.N(7).Eval()
	if !ok || n.String() != "7" {
		t.Errorf("Num.Eval() should succeed with same value")
	}
}

// ============================================================
// Sym tests
// ============================================================

func TestSym_String(t *testing.T) {
	x := gosymbol.S("x")
	if x.String() != "x" {
		t.Errorf("want x, got %s", x.String())
	}
}

func TestSym_Sub_Match(t *testing.T) {
	x := gosymbol.S("x")
	result := x.Sub("x", gosymbol.N(3))
	if gosymbol.String(result) != "3" {
		t.Errorf("want 3, got %s", gosymbol.String(result))
	}
}

func TestSym_Sub_NoMatch(t *testing.T) {
	x := gosymbol.S("x")
	result := x.Sub("y", gosymbol.N(3))
	if gosymbol.String(result) != "x" {
		t.Errorf("want x, got %s", gosymbol.String(result))
	}
}

func TestSym_Diff_Self(t *testing.T) {
	result := gosymbol.S("x").Diff("x")
	if gosymbol.String(result) != "1" {
		t.Errorf("d/dx(x) should be 1, got %s", gosymbol.String(result))
	}
}

func TestSym_Diff_Other(t *testing.T) {
	result := gosymbol.S("y").Diff("x")
	if gosymbol.String(result) != "0" {
		t.Errorf("d/dx(y) should be 0, got %s", gosymbol.String(result))
	}
}

// ============================================================
// Add tests
// ============================================================

func TestAdd_Simple(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("x"), gosymbol.N(3))
	if gosymbol.String(expr) != "x + 3" {
		t.Errorf("want 'x + 3', got %s", gosymbol.String(expr))
	}
}

func TestAdd_CollapseToZero(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.N(1), gosymbol.N(-1))
	if gosymbol.String(expr) != "0" {
		t.Errorf("want 0, got %s", gosymbol.String(expr))
	}
}

func TestAdd_LikeTerms(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("x"), gosymbol.S("x"))
	if gosymbol.String(expr) != "2*x" {
		t.Errorf("want '2*x', got %s", gosymbol.String(expr))
	}
}

func TestAdd_Diff(t *testing.T) {
	// d/dx(x^2 + 3x + 1) = 2x + 3
	x := gosymbol.S("x")
	expr := gosymbol.AddOf(gosymbol.PowOf(x, gosymbol.N(2)), gosymbol.MulOf(gosymbol.N(3), x), gosymbol.N(1))
	d := gosymbol.Diff(expr, "x")
	// Result should contain x and constant terms
	str := gosymbol.String(d)
	if !strings.Contains(str, "x") {
		t.Errorf("d/dx(x^2+3x+1) should contain x, got %s", str)
	}
}

func TestAdd_SingleTerm(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.N(5))
	if gosymbol.String(expr) != "5" {
		t.Errorf("single-term Add should unwrap, got %s", gosymbol.String(expr))
	}
}

// ============================================================
// Mul tests
// ============================================================

func TestMul_Simple(t *testing.T) {
	expr := gosymbol.MulOf(gosymbol.N(3), gosymbol.S("x"))
	if gosymbol.String(expr) != "3*x" {
		t.Errorf("want '3*x', got %s", gosymbol.String(expr))
	}
}

func TestMul_ZeroCollapse(t *testing.T) {
	expr := gosymbol.MulOf(gosymbol.N(0), gosymbol.S("x"))
	if gosymbol.String(expr) != "0" {
		t.Errorf("0*x should be 0, got %s", gosymbol.String(expr))
	}
}

func TestMul_OneElide(t *testing.T) {
	expr := gosymbol.MulOf(gosymbol.N(1), gosymbol.S("x"))
	if gosymbol.String(expr) != "x" {
		t.Errorf("1*x should be x, got %s", gosymbol.String(expr))
	}
}

func TestMul_ProductRule(t *testing.T) {
	// d/dx(x * x) = 2x
	x := gosymbol.S("x")
	expr := gosymbol.MulOf(x, x)
	d := gosymbol.Diff(expr, "x")
	str := gosymbol.String(d)
	if !strings.Contains(str, "x") && str != "2" {
		t.Errorf("d/dx(x*x) should be 2x or similar, got %s", str)
	}
}

// ============================================================
// Pow tests
// ============================================================

func TestPow_Simple(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(2))
	if gosymbol.String(expr) != "x^2" {
		t.Errorf("want x^2, got %s", gosymbol.String(expr))
	}
}

func TestPow_ZeroExp(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(0))
	if gosymbol.String(expr) != "1" {
		t.Errorf("x^0 should be 1, got %s", gosymbol.String(expr))
	}
}

func TestPow_OneExp(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(1))
	if gosymbol.String(expr) != "x" {
		t.Errorf("x^1 should be x, got %s", gosymbol.String(expr))
	}
}

func TestPow_NumericEval(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.N(2), gosymbol.N(3))
	if gosymbol.String(expr) != "8" {
		t.Errorf("2^3 should be 8, got %s", gosymbol.String(expr))
	}
}

func TestPow_Diff_PowerRule(t *testing.T) {
	// d/dx(x^3) = 3*x^2
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(3))
	d := gosymbol.Diff(expr, "x")
	str := gosymbol.String(d)
	if !strings.Contains(str, "3") {
		t.Errorf("d/dx(x^3) should contain 3, got %s", str)
	}
}

func TestPow_LaTeX(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(2))
	if expr.LaTeX() != "x^{2}" {
		t.Errorf("want x^{2}, got %s", expr.LaTeX())
	}
}

// ============================================================
// Func tests
// ============================================================

func TestFunc_Sin_String(t *testing.T) {
	expr := gosymbol.SinOf(gosymbol.S("x"))
	if gosymbol.String(expr) != "sin(x)" {
		t.Errorf("want sin(x), got %s", gosymbol.String(expr))
	}
}

func TestFunc_Sin_Diff(t *testing.T) {
	// d/dx(sin(x)) = cos(x)
	d := gosymbol.Diff(gosymbol.SinOf(gosymbol.S("x")), "x")
	if gosymbol.String(d) != "cos(x)" {
		t.Errorf("d/dx(sin(x)) should be cos(x), got %s", gosymbol.String(d))
	}
}

func TestFunc_Cos_Diff(t *testing.T) {
	// d/dx(cos(x)) = -sin(x)
	d := gosymbol.Diff(gosymbol.CosOf(gosymbol.S("x")), "x")
	str := gosymbol.String(d)
	if !strings.Contains(str, "sin") {
		t.Errorf("d/dx(cos(x)) should contain sin, got %s", str)
	}
}

func TestFunc_Exp_Diff(t *testing.T) {
	// d/dx(exp(x)) = exp(x)
	d := gosymbol.Diff(gosymbol.ExpOf(gosymbol.S("x")), "x")
	if gosymbol.String(d) != "exp(x)" {
		t.Errorf("d/dx(exp(x)) should be exp(x), got %s", gosymbol.String(d))
	}
}

func TestFunc_Ln_Diff(t *testing.T) {
	// d/dx(ln(x)) = x^(-1)
	d := gosymbol.Diff(gosymbol.LnOf(gosymbol.S("x")), "x")
	str := gosymbol.String(d)
	if !strings.Contains(str, "x") {
		t.Errorf("d/dx(ln(x)) should contain x, got %s", str)
	}
}

func TestFunc_Numeric_Eval(t *testing.T) {
	expr := gosymbol.SinOf(gosymbol.N(0))
	if gosymbol.String(expr) != "0" {
		t.Errorf("sin(0) should evaluate to 0, got %s", gosymbol.String(expr))
	}
}

func TestFunc_LaTeX_Sin(t *testing.T) {
	l := gosymbol.SinOf(gosymbol.S("x")).LaTeX()
	if !strings.Contains(l, `\sin`) {
		t.Errorf("LaTeX for sin should contain \\sin, got %s", l)
	}
}

// ============================================================
// Expand tests
// ============================================================

func TestExpand_Distribution(t *testing.T) {
	// (x+1)*(x+2) => x^2 + 3x + 2
	x := gosymbol.S("x")
	expr := gosymbol.MulOf(
		gosymbol.AddOf(x, gosymbol.N(1)),
		gosymbol.AddOf(x, gosymbol.N(2)),
	)
	expanded := gosymbol.Expand(expr)
	str := gosymbol.String(expanded)
	// Should contain x^2, x terms, and constant
	if !strings.Contains(str, "x") {
		t.Errorf("expanded (x+1)(x+2) should contain x, got %s", str)
	}
}

// ============================================================
// FreeSymbols tests
// ============================================================

func TestFreeSymbols(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("x"), gosymbol.MulOf(gosymbol.S("y"), gosymbol.N(2)))
	syms := gosymbol.FreeSymbols(expr)
	if _, ok := syms["x"]; !ok {
		t.Error("expected x in free symbols")
	}
	if _, ok := syms["y"]; !ok {
		t.Error("expected y in free symbols")
	}
	if len(syms) != 2 {
		t.Errorf("expected 2 free symbols, got %d", len(syms))
	}
}

func TestFreeSymbols_Constant(t *testing.T) {
	syms := gosymbol.FreeSymbols(gosymbol.N(5))
	if len(syms) != 0 {
		t.Errorf("constant should have no free symbols, got %d", len(syms))
	}
}

// ============================================================
// Degree tests
// ============================================================

func TestDegree_Linear(t *testing.T) {
	x := gosymbol.S("x")
	if gosymbol.Degree(x, "x") != 1 {
		t.Error("degree of x should be 1")
	}
}

func TestDegree_Quadratic(t *testing.T) {
	x := gosymbol.S("x")
	expr := gosymbol.PowOf(x, gosymbol.N(2))
	if gosymbol.Degree(expr, "x") != 2 {
		t.Error("degree of x^2 should be 2")
	}
}

func TestDegree_Constant(t *testing.T) {
	if gosymbol.Degree(gosymbol.N(5), "x") != 0 {
		t.Error("degree of constant should be 0")
	}
}

// ============================================================
// PolyCoeffs tests
// ============================================================

func TestPolyCoeffs(t *testing.T) {
	x := gosymbol.S("x")
	// 3x^2 + 2x + 1
	expr := gosymbol.AddOf(
		gosymbol.MulOf(gosymbol.N(3), gosymbol.PowOf(x, gosymbol.N(2))),
		gosymbol.MulOf(gosymbol.N(2), x),
		gosymbol.N(1),
	)
	coeffs := gosymbol.PolyCoeffs(expr, "x")
	if _, ok := coeffs[2]; !ok {
		t.Error("expected coefficient for degree 2")
	}
	if _, ok := coeffs[1]; !ok {
		t.Error("expected coefficient for degree 1")
	}
	if _, ok := coeffs[0]; !ok {
		t.Error("expected coefficient for degree 0")
	}
}

// ============================================================
// Solver tests
// ============================================================

func TestSolveLinear_Exact(t *testing.T) {
	// 2x + 4 = 0 => x = -2
	res := gosymbol.SolveLinear(gosymbol.N(2), gosymbol.N(4))
	if res.Error != "" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
	if len(res.Solutions) != 1 {
		t.Fatalf("expected 1 solution, got %d", len(res.Solutions))
	}
	if gosymbol.String(res.Solutions[0]) != "-2" {
		t.Errorf("expected -2, got %s", gosymbol.String(res.Solutions[0]))
	}
}

func TestSolveLinear_Rational(t *testing.T) {
	// 3x + 1 = 0 => x = -1/3
	res := gosymbol.SolveLinear(gosymbol.N(3), gosymbol.N(1))
	if res.Error != "" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
	if gosymbol.String(res.Solutions[0]) != "-1/3" {
		t.Errorf("expected -1/3, got %s", gosymbol.String(res.Solutions[0]))
	}
}

func TestSolveLinear_ZeroA_ZeroB(t *testing.T) {
	res := gosymbol.SolveLinear(gosymbol.N(0), gosymbol.N(0))
	if res.Error == "" {
		t.Error("expected error for 0x+0=0 (infinite solutions)")
	}
}

func TestSolveQuadratic_TwoRoots(t *testing.T) {
	// x^2 - 5x + 6 = 0 => x = 2, 3
	res := gosymbol.SolveQuadratic(gosymbol.N(1), gosymbol.N(-5), gosymbol.N(6))
	if res.Error != "" {
		t.Fatalf("unexpected error: %s", res.Error)
	}
	if len(res.Solutions) != 2 {
		t.Fatalf("expected 2 solutions, got %d", len(res.Solutions))
	}
}

func TestSolveQuadratic_Complex(t *testing.T) {
	// x^2 + 1 = 0 => complex
	res := gosymbol.SolveQuadratic(gosymbol.N(1), gosymbol.N(0), gosymbol.N(1))
	if res.Error == "" {
		t.Error("expected error for complex roots")
	}
	if !strings.Contains(res.Error, "complex") {
		t.Errorf("expected 'complex' in error, got %s", res.Error)
	}
}

func TestSolveLinearSystem2x2(t *testing.T) {
	// x + y = 3, x - y = 1 => x=2, y=1
	x, y, err := gosymbol.SolveLinearSystem2x2(
		gosymbol.N(1), gosymbol.N(1), gosymbol.N(3),
		gosymbol.N(1), gosymbol.N(-1), gosymbol.N(1),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if gosymbol.String(x) != "2" {
		t.Errorf("expected x=2, got %s", gosymbol.String(x))
	}
	if gosymbol.String(y) != "1" {
		t.Errorf("expected y=1, got %s", gosymbol.String(y))
	}
}

// ============================================================
// Integration tests
// ============================================================

func TestIntegrate_Constant(t *testing.T) {
	result, ok := gosymbol.Integrate(gosymbol.N(5), "x")
	if !ok {
		t.Fatal("integration of constant should succeed")
	}
	if gosymbol.String(result) != "5*x" {
		t.Errorf("∫5 dx should be 5*x, got %s", gosymbol.String(result))
	}
}

func TestIntegrate_Variable(t *testing.T) {
	result, ok := gosymbol.Integrate(gosymbol.S("x"), "x")
	if !ok {
		t.Fatal("integration of x should succeed")
	}
	str := gosymbol.String(result)
	if !strings.Contains(str, "x^2") {
		t.Errorf("∫x dx should contain x^2, got %s", str)
	}
}

func TestIntegrate_Power(t *testing.T) {
	// ∫x^3 dx = (1/4)x^4
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(3))
	result, ok := gosymbol.Integrate(expr, "x")
	if !ok {
		t.Fatal("integration of x^3 should succeed")
	}
	str := gosymbol.String(result)
	if !strings.Contains(str, "x^4") {
		t.Errorf("∫x^3 dx should contain x^4, got %s", str)
	}
}

func TestIntegrate_InverseX(t *testing.T) {
	// ∫x^(-1) dx = ln|x|
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(-1))
	result, ok := gosymbol.Integrate(expr, "x")
	if !ok {
		t.Fatal("integration of x^-1 should succeed")
	}
	str := gosymbol.String(result)
	if !strings.Contains(str, "ln") {
		t.Errorf("∫x^-1 dx should contain ln, got %s", str)
	}
}

func TestIntegrate_Sin(t *testing.T) {
	result, ok := gosymbol.Integrate(gosymbol.SinOf(gosymbol.S("x")), "x")
	if !ok {
		t.Fatal("integration of sin(x) should succeed")
	}
	str := gosymbol.String(result)
	if !strings.Contains(str, "cos") {
		t.Errorf("∫sin(x) dx should contain cos, got %s", str)
	}
}

func TestIntegrate_Cos(t *testing.T) {
	result, ok := gosymbol.Integrate(gosymbol.CosOf(gosymbol.S("x")), "x")
	if !ok {
		t.Fatal("integration of cos(x) should succeed")
	}
	if gosymbol.String(result) != "sin(x)" {
		t.Errorf("∫cos(x) dx should be sin(x), got %s", gosymbol.String(result))
	}
}

func TestIntegrate_Exp(t *testing.T) {
	result, ok := gosymbol.Integrate(gosymbol.ExpOf(gosymbol.S("x")), "x")
	if !ok {
		t.Fatal("integration of exp(x) should succeed")
	}
	if gosymbol.String(result) != "exp(x)" {
		t.Errorf("∫exp(x) dx should be exp(x), got %s", gosymbol.String(result))
	}
}

func TestDefiniteIntegrate(t *testing.T) {
	// ∫_0^1 x dx = 0.5
	result := gosymbol.DefiniteIntegrate(gosymbol.S("x"), "x", 0, 1)
	if result < 0.49 || result > 0.51 {
		t.Errorf("∫_0^1 x dx should be ~0.5, got %f", result)
	}
}

// ============================================================
// Taylor series tests
// ============================================================

func TestTaylorSeries_Constant(t *testing.T) {
	result := gosymbol.TaylorSeries(gosymbol.N(5), "x", gosymbol.N(0), 3)
	if gosymbol.String(result) != "5" {
		t.Errorf("Taylor of 5 should be 5, got %s", gosymbol.String(result))
	}
}

func TestTaylorSeries_Linear(t *testing.T) {
	// Taylor of x around 0 to order 3 = x
	x := gosymbol.S("x")
	result := gosymbol.TaylorSeries(x, "x", gosymbol.N(0), 3)
	str := gosymbol.String(result)
	if !strings.Contains(str, "x") {
		t.Errorf("Taylor of x should contain x, got %s", str)
	}
}

// ============================================================
// JSON Serialization tests
// ============================================================

func TestToJSON_Num(t *testing.T) {
	j, err := gosymbol.ToJSON(gosymbol.N(3))
	if err != nil {
		t.Fatalf("ToJSON error: %v", err)
	}
	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)
	if m["type"] != "num" {
		t.Errorf("expected type=num, got %v", m["type"])
	}
}

func TestToJSON_Add(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("x"), gosymbol.N(1))
	j, err := gosymbol.ToJSON(expr)
	if err != nil {
		t.Fatalf("ToJSON error: %v", err)
	}
	if !strings.Contains(j, "add") && !strings.Contains(j, "sym") {
		t.Errorf("JSON for Add should contain 'add' or 'sym', got: %s", j)
	}
}

func TestFromJSON_RoundTrip(t *testing.T) {
	original := gosymbol.AddOf(gosymbol.MulOf(gosymbol.N(2), gosymbol.S("x")), gosymbol.N(1))
	j, _ := gosymbol.ToJSON(original)
	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)
	rebuilt, err := gosymbol.FromJSON(m)
	if err != nil {
		t.Fatalf("FromJSON error: %v", err)
	}
	if gosymbol.String(rebuilt) != gosymbol.String(original) {
		t.Errorf("round-trip mismatch: %s != %s", gosymbol.String(rebuilt), gosymbol.String(original))
	}
}

// ============================================================
// MCP tool call tests
// ============================================================

func TestHandleToolCall_Simplify(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("x"), gosymbol.S("x"))
	j, _ := gosymbol.ToJSON(expr)
	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)

	resp := gosymbol.HandleToolCall(gosymbol.ToolRequest{
		Tool:   "simplify",
		Params: map[string]interface{}{"expr": m},
	})
	if resp.Error != "" {
		t.Fatalf("unexpected error: %s", resp.Error)
	}
	if !strings.Contains(resp.String, "x") {
		t.Errorf("simplified x+x should contain x, got %s", resp.String)
	}
}

func TestHandleToolCall_Diff(t *testing.T) {
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(2))
	j, _ := gosymbol.ToJSON(expr)
	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)

	resp := gosymbol.HandleToolCall(gosymbol.ToolRequest{
		Tool:   "diff",
		Params: map[string]interface{}{"expr": m, "var": "x"},
	})
	if resp.Error != "" {
		t.Fatalf("unexpected error: %s", resp.Error)
	}
	if !strings.Contains(resp.String, "x") {
		t.Errorf("d/dx(x^2) should contain x, got %s", resp.String)
	}
}

func TestHandleToolCall_FreeSymbols(t *testing.T) {
	expr := gosymbol.AddOf(gosymbol.S("a"), gosymbol.S("b"))
	j, _ := gosymbol.ToJSON(expr)
	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)

	resp := gosymbol.HandleToolCall(gosymbol.ToolRequest{
		Tool:   "free_symbols",
		Params: map[string]interface{}{"expr": m},
	})
	if resp.Error != "" {
		t.Fatalf("unexpected error: %s", resp.Error)
	}
	syms, ok := resp.Result.([]string)
	if !ok {
		t.Fatalf("expected []string result, got %T", resp.Result)
	}
	if len(syms) != 2 {
		t.Errorf("expected 2 symbols, got %d", len(syms))
	}
}

func TestHandleToolCall_UnknownTool(t *testing.T) {
	resp := gosymbol.HandleToolCall(gosymbol.ToolRequest{Tool: "nonexistent", Params: map[string]interface{}{}})
	if resp.Error == "" {
		t.Error("expected error for unknown tool")
	}
}

func TestMCPToolSpec(t *testing.T) {
	spec := gosymbol.MCPToolSpec()
	if !strings.Contains(spec, "simplify") {
		t.Error("MCP spec should contain 'simplify'")
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(spec), &m); err != nil {
		t.Errorf("MCP spec should be valid JSON: %v", err)
	}
}

// ============================================================
// Equation tests
// ============================================================

func TestEquation_String(t *testing.T) {
	eq := gosymbol.Eq(gosymbol.S("x"), gosymbol.N(5))
	if eq.String() != "x = 5" {
		t.Errorf("want 'x = 5', got %s", eq.String())
	}
}

func TestEquation_Residual(t *testing.T) {
	// x = 5 => x - 5 = 0
	eq := gosymbol.Eq(gosymbol.S("x"), gosymbol.N(5))
	res := eq.Residual()
	str := gosymbol.String(res)
	if !strings.Contains(str, "x") {
		t.Errorf("residual should contain x, got %s", str)
	}
}

// ============================================================
// Higher-order derivative tests
// ============================================================

func TestDiff2(t *testing.T) {
	// d^2/dx^2(x^3) = 6x
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(3))
	d2 := gosymbol.Diff2(expr, "x")
	str := gosymbol.String(d2)
	if !strings.Contains(str, "x") {
		t.Errorf("second derivative of x^3 should contain x, got %s", str)
	}
}

func TestDiffN(t *testing.T) {
	// d^4/dx^4(x^4) = 24
	expr := gosymbol.PowOf(gosymbol.S("x"), gosymbol.N(4))
	d4 := gosymbol.DiffN(expr, "x", 4)
	str := gosymbol.String(d4)
	if str != "24" {
		t.Errorf("d^4/dx^4(x^4) should be 24, got %s", str)
	}
}

// ============================================================
// Equal tests
// ============================================================

func TestEqual_NumTrue(t *testing.T) {
	if !gosymbol.N(3).Equal(gosymbol.N(3)) {
		t.Error("N(3) should equal N(3)")
	}
}

func TestEqual_NumFalse(t *testing.T) {
	if gosymbol.N(3).Equal(gosymbol.N(4)) {
		t.Error("N(3) should not equal N(4)")
	}
}

func TestEqual_SymTrue(t *testing.T) {
	if !gosymbol.S("x").Equal(gosymbol.S("x")) {
		t.Error("S(x) should equal S(x)")
	}
}

func TestEqual_CrossType(t *testing.T) {
	if gosymbol.N(1).Equal(gosymbol.S("x")) {
		t.Error("N(1) should not equal S(x)")
	}
}

// ============================================================
// Determinism test
// ============================================================

func TestDeterminism(t *testing.T) {
	for i := 0; i < 10; i++ {
		expr := gosymbol.AddOf(gosymbol.S("z"), gosymbol.S("a"), gosymbol.S("m"), gosymbol.N(1))
		result := gosymbol.String(expr)
		expected := gosymbol.String(gosymbol.AddOf(gosymbol.S("z"), gosymbol.S("a"), gosymbol.S("m"), gosymbol.N(1)))
		if result != expected {
			t.Errorf("non-deterministic output on iteration %d: %s != %s", i, result, expected)
		}
	}
}