# gosymbol

[![Go Reference](https://pkg.go.dev/badge/github.com/njchilds90/gosymbol.svg)](https://pkg.go.dev/github.com/njchilds90/gosymbol)
[![Tests](https://github.com/njchilds90/gosymbol/actions/workflows/ci.yml/badge.svg)](https://github.com/njchilds90/gosymbol/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](go.mod)
[![Coverage](https://img.shields.io/badge/Coverage-go%20test%20-cover-success)](#testing)

**Minimal deterministic symbolic math kernel in pure Go.**

Modular multi-file layout. Zero dependencies. Exact rational arithmetic. AI-agent ready.

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

## Implemented Features

| Capability | Status | Primary entry points |
|------|--------|--------|
| Deterministic simplification and canonicalization | ✅ | `Simplify`, `DeepSimplify`, `Canonicalize`, `Expr.Canonicalize()` |
| Differentiation, series, and definite integration | ✅ | `DifferentiateExpression`, `TaylorSeries`, `TaylorSeriesWithContext`, `DefiniteIntegrate` |
| Rule-based and transcendental integration | ✅ | `Integrate`, `IntegrateWithConstant`, `PerformRischTranscendentalIntegration`, `PerformRischTranscendentalIntegrationWithContext` |
| Pattern rewriting and polynomial tooling | ✅ | `RewriteExpressionByPatternRules`, `FactorExpression`, `ComputeGroebnerBasis`, `ComputeGroebnerBasisWithContext` |
| Trigonometric and logarithmic expansion | ✅ | `TrigSimplify`, `ExpandTrigonometric`, `ExpandLogarithmic` |
| Symbolic matrices | ✅ | `CreateSymbolicMatrix`, `MatAdd`, `MatMul`, `Det`, `Inverse` |
| Piecewise expressions | ✅ | `CreatePiecewiseExpression` |
| Complex numbers | ✅ | `CreateComplexNumber`, `CreateImaginaryUnit`, `AddComplexNumbers`, `MultiplyComplexNumbers` |
| Assumptions-aware symbols | ✅ | `CreateEnhancedAssumptions`, `ApplyAssumptionsToSymbolicVariable` |
| ASCII and boxed pretty printing | ✅ | `AsciiPrettyPrint`, `CreateBoxedAsciiPrettyPrint` |
| Lambdify and runnable Go generation | ✅ | `LambdifyToGoFunction`, `GenerateRunnableGoFunctionSource` |
| MCP and standalone HTTP wrappers | ✅ | `HandleToolCall`, `MCPToolSpec`, `cmd/mcp-server`, `cmd/standalone_model_context_protocol_http_server` |
| WebAssembly browser interop | ✅ | `RegisterWebAssemblyExample` |

---

## Quick Start

```go
import gosymbol "github.com/njchilds90/gosymbol"

variableX := gosymbol.CreateSymbolicVariable("x")
expression := gosymbol.Parse("sin(x)^2 + cos(x)^2 + 3*x^2 + 1")

fmt.Println(gosymbol.String(gosymbol.Canonicalize(expression))) // 3*x^2 + 2
fmt.Println(gosymbol.LaTeX(expression))

derivative := gosymbol.DifferentiateExpression(expression, "x")
fmt.Println(gosymbol.String(derivative)) // 6*x

integral, ok := gosymbol.PerformRischTranscendentalIntegration(gosymbol.Parse("ln(x)/x"), "x")
fmt.Println(ok, gosymbol.String(integral)) // true 1/2*ln(x)^2

matrix := gosymbol.CreateSymbolicMatrixFromEntries(2, 2, []gosymbol.Expr{
	gosymbol.N(1), variableX,
	gosymbol.N(0), gosymbol.N(1),
})
fmt.Println(matrix.Det().String()) // 1

complexValue := gosymbol.CreateComplexNumber(gosymbol.N(2), gosymbol.N(1))
fmt.Println(complexValue.String()) // 2 + i

generatedSource, _ := gosymbol.GenerateRunnableGoFunctionSource(gosymbol.Parse("x^2 + 1"), "EvaluateSquarePlusOne")
fmt.Println(generatedSource)

fmt.Println(gosymbol.CreateBoxedAsciiPrettyPrint(expression))
```

---


## Repository Layout

```text
core_expression_interface.go                 # Core expression interface, equations, Big-O, simplification
rational_number_node.go                     # Exact rational number node
symbolic_variable_node.go                   # Symbol node + assumptions
addition_node.go                            # Sum node + canonical addition
multiplication_node.go                      # Product node + factor combination
power_node.go                               # Power node
function_node.go                            # Function node family (trig/hyperbolic/inverses)
symbolic_calculus_operations.go             # Differentiate, integrate, Taylor, definite integration
symbolic_algebra_operations.go              # Expand, factor, pattern rewriting, Gröbner bases, polynomials
symbolic_matrix_operations.go               # Matrix helpers: addition, multiplication, determinant, inverse
equation_solving_operations.go              # Linear/quadratic/system/equation solvers
javascript_object_notation_serialization.go # Structured serialization
latex_string_output.go                      # LaTeX and boxed ASCII pretty-print helpers
model_context_protocol_integration.go       # Tool dispatch and schema output
infix_string_parser.go                      # String parser for infix expressions
piecewise_conditional_expression.go         # Piecewise symbolic expressions
compiled_function_generator.go              # Native closure generation and runnable Go source output
complex_number_node.go                      # Symbolic complex-number node
backward_compatibility_aliases.go           # Short-name compatibility layer
arbitrary_constant_term.go                  # Explicit indefinite integration constant
symbolic_mathematics_errors.go              # Structured symbolic errors
cmd/mcp-server/main.go                      # Optional standalone HTTP server
cmd/standalone_model_context_protocol_http_server/main.go # Simple net/http JSON MCP wrapper
wasm_build_example.go                       # js/wasm browser registration helper
examples/main.go                            # Runnable usage demo
comprehensive_test_suite_test.go            # Tests and benchmarks
.github/workflows/ci.yml                    # CI: test/race/gofmt/vet/coverage
```

---

## Design Goals

| Goal | Status |
|------|--------|
| Modular root-package layout | ✅ |
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
gosymbol.N(42)       // integer 42
gosymbol.F(1, 3)     // exact fraction 1/3
gosymbol.NFloat(3.14) // float approximation (use sparingly)
```

### `Sym` — Symbolic variables

```go
x := gosymbol.S("x")
y := gosymbol.S("y")
alpha := gosymbol.S("alpha")  // any string name
```

### `Add` — Sums

```go
gosymbol.AddOf(x, gosymbol.N(1))      // x + 1
gosymbol.AddOf(x, x, gosymbol.N(2))   // 2*x + 2 (like terms combined)
```

### `Mul` — Products

```go
gosymbol.MulOf(gosymbol.N(3), x)      // 3*x
gosymbol.MulOf(x, y)                 // x*y
```

### `Pow` — Powers

```go
gosymbol.PowOf(x, gosymbol.N(2))      // x^2
gosymbol.PowOf(x, gosymbol.F(1, 2))   // x^(1/2)  (sqrt)
gosymbol.SqrtOf(x)                   // x^(1/2)
```

### `Func` — Named functions

```go
gosymbol.SinOf(x)    // sin(x)
gosymbol.CosOf(x)    // cos(x)
gosymbol.TanOf(x)    // tan(x)
gosymbol.ExpOf(x)    // exp(x)
gosymbol.LnOf(x)     // ln(x)
gosymbol.AbsOf(x)    // |x|
gosymbol.SqrtOf(x)   // sqrt(x)
```

---
### Infix parsing and assumptions

```go
expr := gosymbol.Parse("sin(x)^2 + cos(x)^2")
fmt.Println(gosymbol.String(gosymbol.DeepSimplify(expr))) // 1

xPos := gosymbol.SAssume("x", gosymbol.Assumptions{Real: true, Positive: true})
fmt.Println(gosymbol.String(gosymbol.AbsOf(xPos))) // x

assumptions := gosymbol.CreateEnhancedAssumptions(true, true, false, true, true)
xNatural := gosymbol.ApplyAssumptionsToSymbolicVariable(gosymbol.S("n"), assumptions)
fmt.Println(gosymbol.SymbolicVariableIsKnownNatural(xNatural)) // true
```

### Matrix, complex, and piecewise helpers

```go
piecewise := gosymbol.CreatePiecewiseExpression(
	[]gosymbol.PiecewiseCase{
		{Condition: "x > 0", Expression: gosymbol.S("x")},
		{Condition: "otherwise", Expression: gosymbol.N(0)},
	},
	nil,
)
fmt.Println(piecewise.String())

complexProduct := gosymbol.MultiplyComplexNumbers(
	gosymbol.CreateImaginaryUnit(),
	gosymbol.CreateImaginaryUnit(),
)
fmt.Println(complexProduct.String()) // -1
```

### Generated Go source and context-aware helpers

```go
source, _ := gosymbol.GenerateRunnableGoFunctionSource(gosymbol.Parse("exp(x) + 1"), "EvaluateExpPlusOne")
fmt.Println(source)

basis := gosymbol.ComputeGroebnerBasisWithContext(context.Background(), []gosymbol.Expr{
	gosymbol.Parse("x*y - 1"),
	gosymbol.Parse("y - 1"),
}, []string{"x", "y"})
fmt.Println(len(basis) > 0) // true
```

## Rename Mapping

- `expr.go` → `core_expression_interface.go`
- `num.go` → `rational_number_node.go`
- `sym.go` → `symbolic_variable_node.go`
- `add.go` → `addition_node.go`
- `mul.go` → `multiplication_node.go`
- `pow.go` → `power_node.go`
- `func.go` → `function_node.go`
- `calculus.go` → `symbolic_calculus_operations.go`
- `algebra.go` → `symbolic_algebra_operations.go`
- `solvers.go` → `equation_solving_operations.go`
- `parsing.go` → `infix_string_parser.go`
- `serialize.go` → `javascript_object_notation_serialization.go`
- `latex.go` → `latex_string_output.go`
- `mcp.go` → `model_context_protocol_integration.go`
- `matrix.go` → `symbolic_matrix_operations.go`
- `piecewise.go` → `piecewise_conditional_expression.go`
- `lambdify.go` → `compiled_function_generator.go`
- `compatibility.go` → `backward_compatibility_aliases.go`
- `constant.go` → `arbitrary_constant_term.go`
- `errors.go` → `symbolic_mathematics_errors.go`
- `gosymbol_test.go` → `comprehensive_test_suite_test.go`

## Migration Notes

The package keeps the existing short compatibility surface (`N`, `S`, `Diff`, `Eq`, and related helpers) so existing callers continue to compile.
It now also exposes a descriptive full-name surface such as `CreateRationalNumber`, `CreateSymbolicVariable`, `DifferentiateExpression`, `SymbolicIntegration`, `FactorExpression`, `CreateBoxedAsciiPrettyPrint`, `GenerateRunnableGoFunctionSource`, and `CreateComplexNumber` for teams that prefer more explicit APIs.

```go
expression := gosymbol.CreateAddition(
    gosymbol.CreatePower(gosymbol.CreateSymbolicVariable("x"), gosymbol.CreateRationalNumber(2)),
    gosymbol.CreateRationalNumber(1),
)
result, _ := gosymbol.SymbolicIntegration(expression, "x")
fmt.Println(gosymbol.String(result))
```

---

## Calculus

### Differentiation

```go
// First derivative
d := gosymbol.Diff(expr, "x")

// Second derivative
d2 := gosymbol.Diff2(expr, "x")

// nth derivative
dn := gosymbol.DiffN(expr, "x", 4)
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
result, ok := gosymbol.Integrate(expr, "x")
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
result := gosymbol.DefiniteIntegrate(expr, "x", 0.0, 1.0)
```

### Taylor Series

```go
// Taylor expansion of sin(x) around 0, up to order 5
series := gosymbol.TaylorSeries(gosymbol.SinOf(x), "x", gosymbol.N(0), 5)
```

---
## Algebra

### Expand

Distributes multiplication over addition:

```go
expr := gosymbol.MulOf(
    gosymbol.AddOf(x, gosymbol.N(1)),
    gosymbol.AddOf(x, gosymbol.N(2)),
)
expanded := gosymbol.Expand(expr)  // x^2 + 3*x + 2
```

Also expands `(a+b)^n` for integer n ≤ 10.

### Polynomial utilities

```go
// Degree of expr as polynomial in x
deg := gosymbol.Degree(expr, "x")

// Extract coefficients by degree
coeffs := gosymbol.PolyCoeffs(expr, "x")
// coeffs[2] = coefficient of x^2
// coeffs[1] = coefficient of x
// coeffs[0] = constant term
```

### Free symbols

```go
syms := gosymbol.FreeSymbols(expr)
// returns map[string]struct{}{} of symbol names
```

---
## Solvers

### Linear: ax + b = 0

```go
res := gosymbol.SolveLinear(a, b)
// res.Solutions[0] = exact rational solution
// res.ExactForm = true if exact
// res.Error = non-empty if no unique solution
```

### Quadratic: ax² + bx + c = 0

```go
res := gosymbol.SolveQuadratic(a, b, c)
// Returns float64 roots; res.Error contains complex root info if discriminant < 0
```

### 2×2 Linear System

```go
// a1*x + b1*y = c1
// a2*x + b2*y = c2
xSol, ySol, err := gosymbol.SolveLinearSystem2x2(a1, b1, c1, a2, b2, c2)
```

---
## Equations

```go
eq := gosymbol.Eq(x, gosymbol.N(5))
fmt.Println(eq.String())           // x = 5
fmt.Println(eq.LaTeX())            // x = 5
fmt.Println(eq.Residual())         // x + -5 (expression = 0)
```

---
## LaTeX Output

All expressions support LaTeX rendering:

```go
gosymbol.LaTeX(gosymbol.F(1, 3))                       // \frac{1}{3}
gosymbol.LaTeX(gosymbol.PowOf(x, gosymbol.N(2)))        // x^{2}
gosymbol.LaTeX(gosymbol.SinOf(x))                      // \sin\left(x\right)
```

---
## JSON Serialization

Expressions serialize to/from a structured JSON tree — ideal for passing between services or AI tools.

```go
// Serialize
json, err := gosymbol.ToJSON(expr)
// {"type":"add","terms":[{"type":"mul","factors":[{"type":"num","value":"2"},{"type":"sym","name":"x"}]},{"type":"num","value":"1"}]}

// Deserialize
var m map[string]interface{}
json.Unmarshal([]byte(jsonStr), &m)
expr, err := gosymbol.FromJSON(m)
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

`gosymbol` exposes a unified tool call interface compatible with AI agent frameworks including Model Context Protocol (MCP):

```go
req := gosymbol.ToolRequest{
    Tool: "diff",
    Params: map[string]interface{}{
        "expr": exprJSON,  // JSON expression tree
        "var":  "x",
    },
}
resp := gosymbol.HandleToolCall(req)
fmt.Println(resp.String) // human-readable result
fmt.Println(resp.LaTeX)  // LaTeX result
fmt.Println(resp.Error)  // error if failed
```

**Available tools:**

| Tool | Description | Required params |
|------|-------------|-----------------|
| `simplify` | Simplify expression | `expr` |
| `deep_simplify` / `deeply_simplify_expression` | Repeated simplification with trig/hyperbolic identities | `expr` |
| `diff` | Differentiate | `expr`, `var` |
| `integrate` | Integrate (symbolic) | `expr`, `var` |
| `integrate_with_constant` / `perform_symbolic_integration` | Indefinite integration with explicit arbitrary constant | `expr`, `var` |
| `expand` | Algebraic expansion | `expr` |
| `factor` / `factor_expression` | Factor polynomial or rational expressions | `expr`, `var` or `expr` |
| `substitute` | Substitute variable | `expr`, `var`, `value` |
| `to_latex` | Convert to LaTeX | `expr` |
| `free_symbols` | List free variables | `expr` |
| `degree` | Polynomial degree | `expr`, `var` |
| `limit` / `limit_expression` | Symbolic limits with optional `direction` (`+`, `-`, `both`) | `expr`, `var`, `point` |
| `solve_linear` | Solve ax+b=0 | `a`, `b` |
| `solve_quadratic` | Solve ax²+bx+c=0 | `a`, `b`, `c` |
| `taylor` | Taylor series | `expr`, `var`, `around`?, `order`? |

Recent additions include exact-content polynomial/rational factoring, one-sided and infinity-aware limits with basic L'Hôpital support, and broader inverse/hyperbolic calculus support such as `asin`, `acos`, `atan`, `asinh`, `acosh`, and `atanh`.

### Standalone HTTP MCP wrappers

Two runnable entry points are provided:

- `cmd/mcp-server/main.go` for the compact server wrapper.
- `cmd/standalone_model_context_protocol_http_server/main.go` for a minimal `net/http` JSON server with:
  - `POST /tool` for tool execution,
  - `GET /tools` for the tool schema,
  - `GET /healthz` for health checks.

```bash
go run ./cmd/standalone_model_context_protocol_http_server
curl -s http://127.0.0.1:8081/healthz
curl -s http://127.0.0.1:8081/tools
curl -s http://127.0.0.1:8081/tool \
  -H 'content-type: application/json' \
  -d '{"tool":"simplify","params":{"expr":"sin(x)^2 + cos(x)^2"}}'
```

### WebAssembly build target

The browser interop example in `wasm_build_example.go` is guarded by `//go:build js && wasm` and registers `simplifyExpressionWithGoSymbol` in JavaScript.

```bash
GOOS=js GOARCH=wasm go build -o gosymbol.wasm
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" .
```

Then load `wasm_exec.js` in a page, instantiate `gosymbol.wasm`, and call `simplifyExpressionWithGoSymbol("sin(x)^2 + cos(x)^2")`.

### Get the MCP Tool Schema

```go
schema := gosymbol.MCPToolSpec()
// Returns full JSON schema for all tools, suitable for registering
// with any MCP-compatible agent framework.
```

### LLM System Prompt Recommendation

When using gosymbol as an LLM tool backend, include this in your system prompt:

```
You have access to a symbolic math engine. Build expression trees as JSON and call tools:
- Expressions are JSON objects with a "type" field.
- Types: "num" (with "value"), "sym" (with "name"), "add" (with "terms":[]), 
         "mul" (with "factors":[]), "pow" (with "base" and "exp"),
         "func" (with "name" and "arg").
- Available tools: simplify, deep_simplify, diff, integrate, integrate_with_constant,
                 factor_expression, limit_expression, expand, substitute, solve_linear,
                 solve_quadratic, to_latex, free_symbols, degree, taylor.
- Always simplify results before presenting to the user.
- Use to_latex to present math in rendered form.
```

---
## Architecture

```text
core_expression_interface.go
├── Expr interface (Simplify, Canonicalize, String, LaTeX, Sub, Diff, Eval, Equal)
├── canonicalization and simplification guardrails
├── trigonometric identity passes
└── Equation / BigO support

symbolic_calculus_operations.go
├── Diff / Diff2 / DiffN
├── Integrate / IntegrateWithConstant
├── PerformRischTranscendentalIntegration
├── TaylorSeries / TaylorSeriesWithContext
└── DefiniteIntegrate

symbolic_algebra_operations.go
├── Expand / Canonicalize / Collect / Cancel
├── ExpandTrigonometric / ExpandLogarithmic
├── RewriteExpressionByPatternRules
├── FactorExpression
└── ComputeGroebnerBasis / ComputeGroebnerBasisWithContext

symbolic_matrix_operations.go
├── MatAdd / MatMul
├── Det / Inverse
└── Jacobian / Hessian

compiled_function_generator.go
├── LambdifyToGoFunction
├── GenerateRunnableGoFunctionSource
└── AsciiPrettyPrint / boxed rendering

model_context_protocol_integration.go
├── ToolRequest / ToolResponse
├── HandleToolCall
└── MCPToolSpec
```

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
