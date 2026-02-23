# gosymbol

[![Go Reference](https://pkg.go.dev/badge/github.com/njchilds90/gosymbol.svg)](https://pkg.go.dev/github.com/njchilds90/gosymbol)
[![Tests](https://github.com/njchilds90/gosymbol/actions/workflows/ci.yml/badge.svg)](https://github.com/njchilds90/gosymbol/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Minimal deterministic symbolic math kernel in pure Go.**

Single file. Zero dependencies. Exact rational arithmetic. AI-agent ready.

---

## Why gosymbol?

Python has SymPy. Go has... mostly numeric math.

`gosymbol` fills the gap with a compact symbolic core purpose-built for:

- AI agents and LLM tool backends that need exact symbolic reasoning
- Go microservices performing algebraic transforms
- Educational tooling
- Deterministic, reproducible symbolic computation
- MCP (Model Context Protocol) math tool servers

This is **not** a full SymPy port. It is a small, predictable, embeddable symbolic engine.

---

## Installation

```bash
go get github.com/njchilds90/gosymbol
```

---

## Quick Start

```go
import gosympy "github.com/njchilds90/gosymbol"

x := gosympy.S("x")

// Build expressions
expr := gosympy.AddOf(gosympy.MulOf(gosympy.N(3), gosympy.PowOf(x, gosympy.N(2))), gosympy.N(1))

fmt.Println(gosympy.String(expr))  // 3*x^2 + 1
fmt.Println(gosympy.LaTeX(expr))   // 3 x^{2} + 1

// Differentiate
d := gosympy.Diff(expr, "x")
fmt.Println(gosympy.String(d))     // 6*x

// Integrate
integral, ok := gosympy.Integrate(x, "x")
fmt.Println(gosympy.String(integral)) // 1/2*x^2

// Solve
res := gosympy.SolveLinear(gosympy.N(2), gosympy.N(-6))
fmt.Println(gosympy.String(res.Solutions[0])) // 3

// Substitute
v := gosympy.Sub(expr, "x", gosympy.N(2))
fmt.Println(gosympy.String(v))     // 13
```

---

## Design Goals

| Goal | Status |
|------|--------|
| Single file (`sympy.go`) | ✅ |
| Zero external dependencies | ✅ |
| Deterministic simplification | ✅ |
| Exact rational arithmetic | ✅ (`math/big.Rat`) |
| AI / MCP embeddable | ✅ |
| JSON serialization round-trip | ✅ |
| LaTeX output | ✅ |

---

## Expression Types

Every expression implements the `Expr` interface:

```go
type Expr interface {
    Simplify() Expr
    String() string
    LaTeX() string
    Sub(varName string, value Expr) Expr
    Diff(varName string) Expr
    Eval() (*Num, bool)
    Equal(other Expr) bool
}
```

### `Num` — Exact rational numbers

```go
gosympy.N(42)       // integer 42
gosympy.F(1, 3)     // exact fraction 1/3
gosympy.NFloat(3.14) // float approximation (use sparingly)
```

### `Sym` — Symbolic variables

```go
x := gosympy.S("x")
y := gosympy.S("y")
alpha := gosympy.S("alpha")  // any string name
```

### `Add` — Sums

```go
gosympy.AddOf(x, gosympy.N(1))      // x + 1
gosympy.AddOf(x, x, gosympy.N(2))   // 2*x + 2 (like terms combined)
```

### `Mul` — Products

```go
gosympy.MulOf(gosympy.N(3), x)      // 3*x
gosympy.MulOf(x, y)                 // x*y
```

### `Pow` — Powers

```go
gosympy.PowOf(x, gosympy.N(2))      // x^2
gosympy.PowOf(x, gosympy.F(1, 2))   // x^(1/2)  (sqrt)
gosympy.SqrtOf(x)                   // x^(1/2)
```

### `Func` — Named functions

```go
gosympy.SinOf(x)    // sin(x)
gosympy.CosOf(x)    // cos(x)
gosympy.TanOf(x)    // tan(x)
gosympy.ExpOf(x)    // exp(x)
gosympy.LnOf(x)     // ln(x)
gosympy.AbsOf(x)    // |x|
gosympy.SqrtOf(x)   // sqrt(x)
```

---

## Calculus

### Differentiation

```go
// First derivative
d := gosympy.Diff(expr, "x")

// Second derivative
d2 := gosympy.Diff2(expr, "x")

// nth derivative
dn := gosympy.DiffN(expr, "x", 4)
```

Supported rules:
- **Power rule**: d/dx(xⁿ) = n·xⁿ⁻¹
- **Sum rule**: d/dx(f+g) = f' + g'
- **Product rule**: d/dx(f·g) = f'g + fg'
- **Chain rule**: d/dx(f(g(x))) = f'(g(x))·g'(x)
- **Trig**: sin, cos, tan
- **Exponential/log**: exp, ln

### Integration (rule-based)

```go
result, ok := gosympy.Integrate(expr, "x")
```

Supported patterns:
- Constants: ∫c dx = cx
- Power rule: ∫xⁿ dx = xⁿ⁺¹/(n+1)
- Inverse: ∫x⁻¹ dx = ln|x|
- Sum rule: ∫(f+g) dx = ∫f dx + ∫g dx
- Constant multiple: ∫cf dx = c∫f dx
- Basic trig: ∫sin(x) dx = -cos(x), ∫cos(x) dx = sin(x)
- Exponential: ∫eˣ dx = eˣ

### Numerical definite integration

Uses 10-point Gaussian quadrature:

```go
result := gosympy.DefiniteIntegrate(expr, "x", 0.0, 1.0)
```

### Taylor Series

```go
// Taylor expansion of sin(x) around 0, up to order 5
series := gosympy.TaylorSeries(gosympy.SinOf(x), "x", gosympy.N(0), 5)
```

---

## Algebra

### Expand

Distributes multiplication over addition:

```go
expr := gosympy.MulOf(
    gosympy.AddOf(x, gosympy.N(1)),
    gosympy.AddOf(x, gosympy.N(2)),
)
expanded := gosympy.Expand(expr)  // x^2 + 3*x + 2
```

Also expands `(a+b)^n` for integer n ≤ 10.

### Polynomial utilities

```go
// Degree of expr as polynomial in x
deg := gosympy.Degree(expr, "x")

// Extract coefficients by degree
coeffs := gosympy.PolyCoeffs(expr, "x")
// coeffs[2] = coefficient of x^2
// coeffs[1] = coefficient of x
// coeffs[0] = constant term
```

### Free symbols

```go
syms := gosympy.FreeSymbols(expr)
// returns map[string]struct{}{} of symbol names
```

---

## Solvers

### Linear: ax + b = 0

```go
res := gosympy.SolveLinear(a, b)
// res.Solutions[0] = exact rational solution
// res.ExactForm = true if exact
// res.Error = non-empty if no unique solution
```

### Quadratic: ax² + bx + c = 0

```go
res := gosympy.SolveQuadratic(a, b, c)
// Returns float64 roots; res.Error contains complex root info if discriminant < 0
```

### 2×2 Linear System

```go
// a1*x + b1*y = c1
// a2*x + b2*y = c2
xSol, ySol, err := gosympy.SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2)
```

---

## Equations

```go
eq := gosympy.Eq(x, gosympy.N(5))
fmt.Println(eq.String())           // x = 5
fmt.Println(eq.LaTeX())            // x = 5
fmt.Println(eq.Residual())         // x + -5 (expression = 0)
```

---

## LaTeX Output

All expressions support LaTeX rendering:

```go
gosympy.LaTeX(gosympy.F(1, 3))                       // \frac{1}{3}
gosympy.LaTeX(gosympy.PowOf(x, gosympy.N(2)))        // x^{2}
gosympy.LaTeX(gosympy.SinOf(x))                      // \sin\left(x\right)
```

---

## JSON Serialization

Expressions serialize to/from a structured JSON tree — ideal for passing between services or AI tools.

```go
// Serialize
json, err := gosympy.ToJSON(expr)
// {"type":"add","terms":[{"type":"mul","factors":[{"type":"num","value":"2"},{"type":"sym","name":"x"}]},{"type":"num","value":"1"}]}

// Deserialize
var m map[string]interface{}
json.Unmarshal([]byte(jsonStr), &m)
expr, err := gosympy.FromJSON(m)
```

**Expression JSON format:**

| Type | JSON |
|------|------|
| `Num` | `{"type":"num","value":"3/4"}` |
| `Sym` | `{"type":"sym","name":"x"}` |
| `Add` | `{"type":"add","terms":[...]}` |
| `Mul` | `{"type":"mul","factors":[...]}` |
| `Pow` | `{"type":"pow","base":{...},"exp":{...}}` |
| `Func` | `{"type":"func","name":"sin","arg":{...}}` |

---

## AI Agent Integration

### MCP Tool Interface

`go-sympy` exposes a unified tool call interface compatible with AI agent frameworks including Model Context Protocol (MCP):

```go
req := gosympy.ToolRequest{
    Tool: "diff",
    Params: map[string]interface{}{
        "expr": exprJSON,  // JSON expression tree
        "var":  "x",
    },
}
resp := gosympy.HandleToolCall(req)
fmt.Println(resp.String) // human-readable result
fmt.Println(resp.LaTeX)  // LaTeX result
fmt.Println(resp.Error)  // error if failed
```

**Available tools:**

| Tool | Description | Required params |
|------|-------------|-----------------|
| `simplify` | Simplify expression | `expr` |
| `diff` | Differentiate | `expr`, `var` |
| `integrate` | Integrate (symbolic) | `expr`, `var` |
| `expand` | Algebraic expansion | `expr` |
| `substitute` | Substitute variable | `expr`, `var`, `value` |
| `to_latex` | Convert to LaTeX | `expr` |
| `free_symbols` | List free variables | `expr` |
| `degree` | Polynomial degree | `expr`, `var` |
| `solve_linear` | Solve ax+b=0 | `a`, `b` |
| `solve_quadratic` | Solve ax²+bx+c=0 | `a`, `b`, `c` |
| `taylor` | Taylor series | `expr`, `var`, `around`?, `order`? |

### Get the MCP Tool Schema

```go
schema := gosympy.MCPToolSpec()
// Returns full JSON schema for all tools, suitable for registering
// with any MCP-compatible agent framework.
```

### LLM System Prompt Recommendation

When using go-sympy as an LLM tool backend, include this in your system prompt:

```
You have access to a symbolic math engine. Build expression trees as JSON and call tools:
- Expressions are JSON objects with a "type" field.
- Types: "num" (with "value"), "sym" (with "name"), "add" (with "terms":[]), 
         "mul" (with "factors":[]), "pow" (with "base" and "exp"),
         "func" (with "name" and "arg").
- Available tools: simplify, diff, integrate, expand, substitute, solve_linear,
                   solve_quadratic, to_latex, free_symbols, degree, taylor.
- Always simplify results before presenting to the user.
- Use to_latex to present math in rendered form.
```

---

## Architecture

```
sympy.go
├── Expr interface (Simplify, String, LaTeX, Sub, Diff, Eval, Equal)
├── Core nodes
│   ├── Num    — exact rational (math/big.Rat)
│   ├── Sym    — symbolic variable
│   ├── Add    — sum (flattens, combines like terms)
│   ├── Mul    — product (flattens, collects numeric coefficient)
│   ├── Pow    — base^exp (numeric evaluation for small integer exponents)
│   └── Func   — sin, cos, tan, exp, ln, abs
├── Calculus
│   ├── Diff / Diff2 / DiffN
│   ├── Integrate (rule-based symbolic)
│   ├── DefiniteIntegrate (Gaussian quadrature)
│   └── TaylorSeries
├── Algebra
│   ├── Expand (distributive expansion)
│   ├── FreeSymbols
│   ├── Degree
│   └── PolyCoeffs
├── Solvers
│   ├── SolveLinear
│   ├── SolveQuadratic
│   └── SolveLinearSystem2x2
├── Equation
├── Serialization
│   ├── ToJSON / FromJSON
│   └── LaTeX
└── AI/MCP Interface
    ├── ToolRequest / ToolResponse
    ├── HandleToolCall
    └── MCPToolSpec
```

---

## Limitations

- No symbolic factoring (`factor(x^2-1)` → `(x-1)(x+1)`)
- No symbolic limits (`limit(sin(x)/x, x, 0)`)
- No matrix algebra
- No Risch integration algorithm (transcendental integrals)
- No expression parser (build ASTs programmatically or via JSON)
- No pattern matching engine
- No Gröbner bases
- No complex number arithmetic

This is a **minimal symbolic kernel**. See the [Future Directions](#future-directions) section for the roadmap.

---

## Future Directions

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

- [ ] Expression parser (`"2*x^2 + 3*x + 1"` → AST)
- [ ] Symbolic factoring
- [ ] `factor()`, `collect()`, `cancel()`, `apart()`
- [ ] `limit()` using substitution and L'Hôpital
- [ ] Symbolic matrix operations
- [ ] `pprint()` ASCII pretty-printer
- [ ] MCP server wrapper (standalone HTTP server)
- [ ] WASM build target
- [ ] Assumptions system (positive, integer, real, etc.)
- [ ] Piecewise expressions
- [ ] Trigonometric identities
- [ ] Expand via `expand_trig`, `expand_log`
- [ ] `Lambdify` → compiled Go function

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Philosophy

> Small. Predictable. Deterministic. Embeddable.
> Not big. Not magical. Not opaque.
> Built for humans and AI agents alike.
