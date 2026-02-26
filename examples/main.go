// examples/main.go — gosymbol usage examples
//
// Run: go run examples/main.go
package main

import (
	"encoding/json"
	"fmt"

	gosymbol "github.com/njchilds90/gosymbol"
)

func section(title string) {
	fmt.Printf("\n═══ %s ═══\n", title)
}

func main() {
	x := gosymbol.S("x")
	y := gosymbol.S("y")

	// ── Basic expressions ──────────────────────────────────────
	section("Basic expressions")

	linear := gosymbol.AddOf(gosymbol.MulOf(gosymbol.N(2), x), gosymbol.N(3))
	fmt.Println("2x + 3              =", gosymbol.String(linear))
	fmt.Println("2x + 3 (LaTeX)      =", gosymbol.LaTeX(linear))

	quad := gosymbol.AddOf(
		gosymbol.PowOf(x, gosymbol.N(2)),
		gosymbol.MulOf(gosymbol.N(-4), x),
		gosymbol.N(4),
	)
	fmt.Println("x^2 - 4x + 4       =", gosymbol.String(quad))

	// ── Like-term combination ──────────────────────────────────
	section("Like-term combination")
	expr := gosymbol.AddOf(x, x, x, gosymbol.N(2))
	fmt.Println("x + x + x + 2      =", gosymbol.String(expr))

	// ── Rational arithmetic ────────────────────────────────────
	section("Exact rational arithmetic")
	a := gosymbol.F(1, 3)
	b := gosymbol.F(5, 6)
	fmt.Println("1/3 =", gosymbol.String(a))
	fmt.Println("5/6 =", gosymbol.String(b))
	fmt.Println("(1/3)*x + (5/6)*x =",
		gosymbol.String(gosymbol.AddOf(gosymbol.MulOf(a, x), gosymbol.MulOf(b, x))))

	// ── Substitution ──────────────────────────────────────────
	section("Substitution")
	fmt.Println("2x+3 at x=5        =", gosymbol.String(gosymbol.Sub(linear, "x", gosymbol.N(5))))

	// ── Differentiation ───────────────────────────────────────
	section("Differentiation")
	fmt.Println("d/dx(x^2-4x+4)     =", gosymbol.String(gosymbol.Diff(quad, "x")))
	fmt.Println("d^2/dx^2(x^4)      =", gosymbol.String(gosymbol.Diff2(gosymbol.PowOf(x, gosymbol.N(4)), "x")))
	fmt.Println("d^4/dx^4(x^4)      =", gosymbol.String(gosymbol.DiffN(gosymbol.PowOf(x, gosymbol.N(4)), "x", 4)))
	fmt.Println("d/dx(sin(x))       =", gosymbol.String(gosymbol.Diff(gosymbol.SinOf(x), "x")))
	fmt.Println("d/dx(cos(x))       =", gosymbol.String(gosymbol.Diff(gosymbol.CosOf(x), "x")))
	fmt.Println("d/dx(exp(x))       =", gosymbol.String(gosymbol.Diff(gosymbol.ExpOf(x), "x")))
	fmt.Println("d/dx(ln(x))        =", gosymbol.String(gosymbol.Diff(gosymbol.LnOf(x), "x")))
	chainExpr := gosymbol.SinOf(gosymbol.PowOf(x, gosymbol.N(2)))
	fmt.Println("d/dx(sin(x^2))     =", gosymbol.String(gosymbol.Diff(chainExpr, "x")))

	// ── Expand ────────────────────────────────────────────────
	section("Expand")
	factoredQuad := gosymbol.MulOf(
		gosymbol.AddOf(x, gosymbol.N(-2)),
		gosymbol.AddOf(x, gosymbol.N(-2)),
	)
	fmt.Println("(x-2)^2 expanded   =", gosymbol.String(gosymbol.Expand(factoredQuad)))

	// ── Integration ───────────────────────────────────────────
	section("Integration")
	if r, ok := gosymbol.Integrate(gosymbol.PowOf(x, gosymbol.N(3)), "x"); ok {
		fmt.Println("∫x^3 dx            =", gosymbol.String(r))
	}
	if r, ok := gosymbol.Integrate(gosymbol.SinOf(x), "x"); ok {
		fmt.Println("∫sin(x) dx         =", gosymbol.String(r))
	}
	if r, ok := gosymbol.Integrate(gosymbol.ExpOf(x), "x"); ok {
		fmt.Println("∫exp(x) dx         =", gosymbol.String(r))
	}
	fmt.Printf("∫_0^1 x dx (num)   ≈ %.6f\n", gosymbol.DefiniteIntegrate(x, "x", 0, 1))
	fmt.Printf("∫_0^π sin(x) dx    ≈ %.6f\n", gosymbol.DefiniteIntegrate(gosymbol.SinOf(x), "x", 0, 3.14159265))

	// ── Taylor Series ─────────────────────────────────────────
	section("Taylor Series")
	sinTaylor := gosymbol.TaylorSeries(gosymbol.SinOf(x), "x", gosymbol.N(0), 5)
	fmt.Println("sin(x) ≈ (order 5) =", gosymbol.String(sinTaylor))

	// ── Free symbols ──────────────────────────────────────────
	section("Free symbols")
	multiVar := gosymbol.AddOf(gosymbol.MulOf(x, y), gosymbol.S("z"))
	syms := gosymbol.FreeSymbols(multiVar)
	names := []string{}
	for k := range syms {
		names = append(names, k)
	}
	fmt.Println("xy + z free symbols =", names)

	// ── Polynomial utilities ──────────────────────────────────
	section("Polynomial utilities")
	poly := gosymbol.AddOf(
		gosymbol.MulOf(gosymbol.N(5), gosymbol.PowOf(x, gosymbol.N(3))),
		gosymbol.MulOf(gosymbol.N(2), gosymbol.PowOf(x, gosymbol.N(2))),
		gosymbol.MulOf(gosymbol.N(-1), x),
		gosymbol.N(7),
	)
	fmt.Println("Poly:               =", gosymbol.String(poly))
	fmt.Println("Degree in x        =", gosymbol.Degree(poly, "x"))
	coeffs := gosymbol.PolyCoeffs(poly, "x")
	for deg := 3; deg >= 0; deg-- {
		if c, ok := coeffs[deg]; ok {
			fmt.Printf("  coeff[x^%d]       = %s\n", deg, gosymbol.String(c))
		}
	}

	// ── Solvers ───────────────────────────────────────────────
	section("Solvers")
	res := gosymbol.SolveLinear(gosymbol.N(3), gosymbol.N(-9))
	fmt.Println("3x - 9 = 0, x     =", gosymbol.String(res.Solutions[0]))

	qres := gosymbol.SolveQuadratic(gosymbol.N(1), gosymbol.N(-5), gosymbol.N(6))
	fmt.Printf("x^2-5x+6=0, x     = %s, %s\n",
		gosymbol.String(qres.Solutions[0]),
		gosymbol.String(qres.Solutions[1]))

	xSol, ySol, _ := gosymbol.SolveLinearSystem2x2(
		gosymbol.N(1), gosymbol.N(1), gosymbol.N(5),
		gosymbol.N(1), gosymbol.N(-1), gosymbol.N(1),
	)
	fmt.Println("x+y=5, x-y=1     x=", gosymbol.String(xSol), " y=", gosymbol.String(ySol))

	// ── Equations ─────────────────────────────────────────────
	section("Equations")
	eq := gosymbol.Eq(gosymbol.PowOf(x, gosymbol.N(2)), gosymbol.N(4))
	fmt.Println("Equation           =", eq.String())
	fmt.Println("Equation (LaTeX)   =", eq.LaTeX())
	fmt.Println("Residual           =", gosymbol.String(eq.Residual()))

	// ── JSON serialization ────────────────────────────────────
	section("JSON serialization")
	j, _ := gosymbol.ToJSON(linear)
	fmt.Println("JSON(2x+3)         =", j)

	var m map[string]interface{}
	json.Unmarshal([]byte(j), &m)
	rebuilt, _ := gosymbol.FromJSON(m)
	fmt.Println("Rebuilt from JSON  =", gosymbol.String(rebuilt))

	// ── MCP tool calls ────────────────────────────────────────
	section("MCP tool calls")
	exprJSON := map[string]interface{}{
		"type": "pow",
		"base": map[string]interface{}{"type": "sym", "name": "x"},
		"exp":  map[string]interface{}{"type": "num", "value": "3"},
	}
	resp := gosymbol.HandleToolCall(gosymbol.ToolRequest{
		Tool:   "diff",
		Params: map[string]interface{}{"expr": exprJSON, "var": "x"},
	})
	fmt.Println("diff(x^3, x)       =", resp.String)
	fmt.Println("  LaTeX            =", resp.LaTeX)

	intResp := gosymbol.HandleToolCall(gosymbol.ToolRequest{
		Tool:   "integrate",
		Params: map[string]interface{}{"expr": exprJSON, "var": "x"},
	})
	fmt.Println("integrate(x^3, x)  =", intResp.String)

	fmt.Println()
	fmt.Println("MCP tool schema (truncated):")
	schema := gosymbol.MCPToolSpec()
	if len(schema) > 200 {
		fmt.Println(schema[:200], "...")
	}
}