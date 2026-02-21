# AGENTS.md — go-sympy for AI Agents

This document is written for AI agents, LLMs, and automated systems using go-sympy as a symbolic math backend.

---

## What go-sympy does

go-sympy is a deterministic symbolic math kernel. Given an expression tree (as JSON), it can:

- **Simplify** algebraic expressions
- **Differentiate** (symbolic derivatives, chain rule, product rule, trig, exp, ln)
- **Integrate** (rule-based: polynomials, trig, exp)
- **Expand** (distribute multiplication over addition)
- **Substitute** (replace variables with values or sub-expressions)
- **Solve** (linear, quadratic, 2×2 systems)
- **Taylor series** (around a point to arbitrary order)
- **Extract free symbols** and **polynomial degree/coefficients**
- **Render LaTeX** for display

---

## Expression JSON Format

All expressions are passed as JSON objects with a required `"type"` field.

### Atoms

```json
{"type": "num", "value": "3"}
{"type": "num", "value": "1/3"}
{"type": "sym", "name": "x"}
```

### Compound expressions

```json
// x + 1
{"type": "add", "terms": [
    {"type": "sym", "name": "x"},
    {"type": "num", "value": "1"}
]}

// 2 * x
{"type": "mul", "factors": [
    {"type": "num", "value": "2"},
    {"type": "sym", "name": "x"}
]}

// x^2
{"type": "pow",
    "base": {"type": "sym", "name": "x"},
    "exp":  {"type": "num", "value": "2"}
}

// sin(x)
{"type": "func", "name": "sin", "arg": {"type": "sym", "name": "x"}}
```

### Supported function names

`sin`, `cos`, `tan`, `exp`, `ln`, `abs`

---

## Available Tools

### `simplify`
Simplify an expression. Combines like terms, evaluates constants, applies algebraic identities.
```json
{"tool": "simplify", "params": {"expr": <EXPR>}}
```

### `diff`
Differentiate with respect to a variable.
```json
{"tool": "diff", "params": {"expr": <EXPR>, "var": "x"}}
```

### `integrate`
Rule-based symbolic integration.
```json
{"tool": "integrate", "params": {"expr": <EXPR>, "var": "x"}}
```
Returns `error` if the form is not supported.

### `expand`
Expand algebraically (distribute, expand powers).
```json
{"tool": "expand", "params": {"expr": <EXPR>}}
```

### `substitute`
Replace a variable with a value.
```json
{"tool": "substitute", "params": {"expr": <EXPR>, "var": "x", "value": <EXPR>}}
```

### `to_latex`
Get LaTeX rendering of an expression.
```json
{"tool": "to_latex", "params": {"expr": <EXPR>}}
```

### `free_symbols`
Get list of variable names in the expression.
```json
{"tool": "free_symbols", "params": {"expr": <EXPR>}}
```
Returns: `{"result": ["x", "y"]}` (sorted alphabetically)

### `degree`
Polynomial degree in a given variable.
```json
{"tool": "degree", "params": {"expr": <EXPR>, "var": "x"}}
```
Returns: `{"result": 2}`

### `solve_linear`
Solve `a*x + b = 0` for `x`.
```json
{"tool": "solve_linear", "params": {"a": <EXPR>, "b": <EXPR>}}
```
Returns exact rational solution.

### `solve_quadratic`
Solve `a*x^2 + b*x + c = 0`.
```json
{"tool": "solve_quadratic", "params": {"a": <EXPR>, "b": <EXPR>, "c": <EXPR>}}
```
Returns float solutions or error for complex roots.

### `taylor`
Taylor series around a point.
```json
{"tool": "taylor", "params": {
    "expr": <EXPR>,
    "var": "x",
    "around": {"type": "num", "value": "0"},
    "order": 5
}}
```
`around` defaults to 0, `order` defaults to 5 if omitted.

---

## Response Format

All tools return:

```json
{
    "result": <JSON expression tree or primitive>,
    "string": "human readable form",
    "latex": "LaTeX form",
    "error": "error message or empty string"
}
```

Always check `"error"` before using `"result"`.

---

## Worked Examples

### Example 1: Differentiate a polynomial

Goal: Find d/dx(3x² + 2x + 1)

Build the expression:
```json
{
    "type": "add",
    "terms": [
        {"type": "mul", "factors": [
            {"type": "num", "value": "3"},
            {"type": "pow", "base": {"type": "sym", "name": "x"}, "exp": {"type": "num", "value": "2"}}
        ]},
        {"type": "mul", "factors": [
            {"type": "num", "value": "2"},
            {"type": "sym", "name": "x"}
        ]},
        {"type": "num", "value": "1"}
    ]
}
```

Call:
```json
{"tool": "diff", "params": {"expr": <above>, "var": "x"}}
```

Expected response string: `6*x + 2`

---

### Example 2: Solve a linear equation

Goal: Solve 5x - 10 = 0

```json
{"tool": "solve_linear", "params": {
    "a": {"type": "num", "value": "5"},
    "b": {"type": "num", "value": "-10"}
}}
```

Expected: `{"result": [{"type":"num","value":"2"}], "string": "2"}`

---

### Example 3: Integrate and evaluate

Goal: Find ∫₀¹ x² dx

Step 1 — integrate symbolically:
```json
{"tool": "integrate", "params": {
    "expr": {"type": "pow", "base": {"type": "sym", "name": "x"}, "exp": {"type": "num", "value": "2"}},
    "var": "x"
}}
```
Result: `1/3*x^3`

Step 2 — substitute x=1:
```json
{"tool": "substitute", "params": {
    "expr": <result from step 1>,
    "var": "x",
    "value": {"type": "num", "value": "1"}
}}
```
Result: `1/3`

Step 3 — substitute x=0 and compute F(1) - F(0) = 1/3 - 0 = 1/3.

---

## Determinism Guarantee

go-sympy guarantees that:
- Identical input expressions always produce identical output
- Term ordering in Add and Mul is lexicographically stable
- No random or stateful elements affect output
- Safe to call from concurrent goroutines (expressions are immutable after simplification)

This makes go-sympy reliable for reasoning chains where the agent checks intermediate results.

---

## Error Handling

If a tool cannot complete its operation, it returns:
```json
{"error": "description of what went wrong"}
```

Common errors:
- `"integration failed: unsupported form"` — try a different approach or numerical integration
- `"complex roots: ..."` — quadratic has no real solutions
- `"system is singular"` — 2×2 system has no unique solution
- `"missing param: ..."` — required parameter not provided

---

## Limitations Agents Should Know

1. **No parser** — expressions must be built as JSON trees, not strings like `"2*x+1"`
2. **Integration is pattern-based** — it will fail on integrals like ∫sin(x²)dx
3. **Simplification is not always canonical** — two equivalent expressions may not compare equal
4. **No complex numbers** — computations are real-valued
5. **Float results** — quadratic solver and numerical integration return floats, not exact rationals

---

## MCP Server Setup (Coming Soon)

A standalone MCP server wrapper is planned. Until then, embed directly:

```go
import gosympy "github.com/njchilds90/go-sympy"

// Handle incoming tool call from agent
func handleMCPRequest(body []byte) []byte {
    var req gosympy.ToolRequest
    json.Unmarshal(body, &req)
    resp := gosympy.HandleToolCall(req)
    result, _ := json.Marshal(resp)
    return result
}

// Get tool schema to register with agent framework
schema := gosympy.MCPToolSpec()
```
