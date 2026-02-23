// examples/main.go — go-sympy usage examples
//
// Run: go run examples/main.go
package main

import (
	"encoding/json"
	"fmt"

	gosympy "github.com/njchilds90/gosymbol"
)

func section(title string) {
	fmt.Printf("\n═══ %s ═══\n", title)
}

func main() {
	x := gosympy.S("x")
	y := gosympy.S("y")

	// ── Basic expressions ──────────────────────────────────────
	section("Basic expressions")

	linear := gosympy.AddOf(gosympy.MulOf(gosympy.N(2), x), gosympy.N(3))
	fmt.Println("2x + 3              =", gosympy.String(linear))
	fmt.Println("2x + 3 (LaTeX)      =", gosympy.LaTeX(linear))

	quad := gosympy.AddOf(
		gosympy.PowOf(x, gosympy.N(2)),
		gosympy.MulOf(gosympy.N(-4), x),
		gosympy.N(4),
	)
	fmt.Println("x^2 - 4x + 4       =", gosympy.String(quad))

	// ── Like-term combination ──────────────────────────────────
	section("Like-term combination")
	expr := gosympy.AddOf(x, x, x, gosympy.N(2))
	fmt.Println("x + x + x + 2      =", gosympy.String(expr)) // 3*x + 2

	// ── Rational arithmetic ────────────────────────────────────
	section("Exact rational arithmetic")
	a := gosympy.F(1, 3)
	b := gosympy.F(5, 6)
	fmt.Println("1/3 =", gosympy.String(a))
	fmt.Println("5/6 =", gosympy.String(b))
	fmt.Println("(1/3)*x + (5/6)*x =",
		gosympy.String(gosympy.AddOf(gosympy.MulOf(a, x), gosympy.MulOf(b, x))))

	// ── Substitution ──────────────────────────────────────────
	section("Substitution")
	fmt.Println("2x+3 at x=5        =", gosympy.String(gosympy.Sub(linear, "x", gosympy.N(5)))) // 13

	// ── Differentiation ───────────────────────────────────────
	section("Differentiation")
	fmt.Println("d/dx(x^2-4x+4)     =", gosympy.String(gosympy.Diff(quad, "x")))    // 2*x + -4
	fmt.Println("d^2/dx^2(x^4)      =", gosympy.String(gosympy.Diff2(gosympy.PowOf(x, gosympy.N(4)), "x")))
	fmt.Println("d^4/dx^4(x^4)      =", gosympy.String(gosympy.DiffN(gosympy.PowOf(x, gosympy.N(4)), "x", 4))) // 24
	fmt.Println("d/dx(sin(x))       =", gosympy.String(gosympy.Diff(gosympy.SinOf(x), "x")))    // cos(x)
	fmt.Println("d/dx(cos(x))       =", gosympy.String(gosympy.Diff(gosympy.CosOf(x), "x")))    // -1*sin(x)
	fmt.Println("d/dx(exp(x))       =", gosympy.String(gosympy.Diff(gosympy.ExpOf(x), "x")))    // exp(x)
	fmt.Println("d/dx(ln(x))        =", gosympy.String(gosympy.Diff(gosympy.LnOf(x), "x")))     // x^-1
	// Chain rule: d/dx sin(x^2) = cos(x^2) * 2x
	chainExpr := gosympy.SinOf(gosympy.PowOf(x, gosympy.N(2)))
	fmt.Println("d/dx(sin(x^2))     =", gosympy.String(gosympy.Diff(chainExpr, "x")))

	// ── Expand ────────────────────────────────────────────────
	section("Expand")
	factoredQuad := gosympy.MulOf(
		gosympy.AddOf(x, gosympy.N(-2)),
		gosympy.AddOf(x, gosympy.N(-2)),
	)
	fmt.Println("(x-2)^2 expanded   =", gosympy.String(gosympy.Expand(factoredQuad)))

	// ── Integration ───────────────────────────────────────────
	section("Integration")
	if r, ok := gosympy.Integrate(gosympy.PowOf(x, gosympy.N(3)), "x"); ok {
		fmt.Println("∫x^3 dx            =", gosympy.String(r))
	}
	if r, ok := gosympy.Integrate(gosympy.SinOf(x), "x"); ok {
		fmt.Println("∫sin(x) dx         =", gosympy.String(r))
	}
	if r, ok := gosympy.Integrate(gosympy.ExpOf(x), "x"); ok {
		fmt.Println("∫exp(x) dx         =", gosympy.String(r))
	}
	fmt.Printf("∫_0^1 x dx (num)   ≈ %.6f\n", gosympy.DefiniteIntegrate(x, "x", 0, 1))
	fmt.Printf("∫_0^π sin(x) dx    ≈ %.6f\n", gosympy.DefiniteIntegrate(gosympy.SinOf(x), "x", 0, 3.14159265))

	// ── Taylor Series ─────────────────────────────────────────
	section("Taylor Series")
	sinTaylor := gosympy.TaylorSeries(gosympy.SinOf(x), "x", gosympy.N(0), 5)
	fmt.Println("sin(x) ≈ (order 5) =", gosympy.String(sinTaylor))

	// ── Free symbols ──────────────────────────────────────────
	section("Free symbols")
	multiVar := gosympy.AddOf(gosympy.MulOf(x, y), gosympy.S("z"))
	syms := gosympy.FreeSymbols(multiVar)
	names := []string{}
	for k := range syms {
		names = append(names, k)
	}
	fmt.Println("xy + z free symbols =", names)

	// ── Polynomial utilities ──────────────────────────────────
	section("Polynomial utilities")
	poly := gosympy.AddOf(
		gosympy.MulOf(gosympy.N(5), gosympy.PowOf(x, gosympy.N(3))),
		gosympy.MulOf(gosympy.N(2), gosympy.PowOf(x, gosympy.N(2))),
		gosympy.MulOf(gosympy.N(-1), x),
		gosympy.N(7),
	)
	fmt.Println("Poly:               =", gosympy.String(poly))
	fmt.Println("Degree in x        =", gosympy.Degree(poly, "x"))
	coeffs := gosympy.PolyCoeffs(poly, "x")
	for deg := 3; deg >= 0; deg-- {
		if c, ok := coeffs[deg]; ok {
			fmt.Printf("  coeff[x^%d]       = %s\n", deg, gosympy.String(c))
		}
	}

	// ── Solvers ───────────────────────────────────────────────
	section("Solvers")
	res := gosympy.SolveLinear(gosympy.N(3), gosympy.N(-9))
	fmt.Println("3x - 9 = 0, x     =", gosympy.String(res.Solutions[0])) // 3

	qres := gosympy.SolveQuadratic(gosympy.N(1), gosympy.N(-5), gosympy.N(6))
	fmt.Printf("x^2-5x+6=0, x     = %s, %s\n",
		gosympy.String(qres.Solutions[0]),
		gosympy.String(qres.Solutions[1]))

	xSol, ySol, _ := gosympy.SolveLinearSystem2x2(
		gosympy.N(1), gosympy.N(1), gosympy.N(5),
		gosympy.N(1), gosympy.N(-1), gosympy.N(1),
	)
	fmt.Println("x+y=5, x-y=1     x=", gosympy.String(xSol), " y=", gosympy.String(ySol))

	// ── Equations ─────────────────────────────────────────────
	section("Equations")
	eq := gosympy.Eq(gosympy.PowOf(x, gosympy.N(2)), gosympy.N(4))
	fmt.Println("Equation           =", eq.String())
	fmt.Println("Equation (LaTeX)   =", eq.LaTeX())
	fmt.Println("Residual           =", gosympy.String(eq.Residual()))

	// ── JSON serialization ────────────────────────────────────
	section("JSON serialization")
	j, _ := gosympy.ToJSON(linear)
	fmt.Println("JSON(2x+3)         =", j)

	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)
	rebuilt, _ := gosympy.FromJSON(m)
	fmt.Println("Rebuilt from JSON  =", gosympy.String(rebuilt))

	// ── MCP tool calls ────────────────────────────────────────
	section("MCP tool calls")
	exprJSON := map[string]interface{}{
		"type": "pow",
		"base": map[string]interface{}{"type": "sym", "name": "x"},
		"exp":  map[string]interface{}{"type": "num", "value": "3"},
	}
	resp := gosympy.HandleToolCall(gosympy.ToolRequest{
		Tool:   "diff",
		Params: map[string]interface{}{"expr": exprJSON, "var": "x"},
	})
	fmt.Println("diff(x^3, x)       =", resp.String)
	fmt.Println("  LaTeX            =", resp.LaTeX)

	intResp := gosympy.HandleToolCall(gosympy.ToolRequest{
		Tool:   "integrate",
		Params: map[string]interface{}{"expr": exprJSON, "var": "x"},
	})
	fmt.Println("integrate(x^3, x)  =", intResp.String)

	fmt.Println()
	fmt.Println("MCP tool schema (truncated):")
	schema := gosympy.MCPToolSpec()
	if len(schema) > 200 {
		fmt.Println(schema[:200], "...")
	}
}
