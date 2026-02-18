# go-sympy

Minimal deterministic symbolic math kernel in pure Go.

Single-file. No dependencies. Rational arithmetic. AI-friendly.

---

## Why

Python has SymPy.

Go has… mostly numeric math.

`go-sympy` is a compact symbolic core designed for:

- AI agents embedding math reasoning
- Lightweight symbolic manipulation
- Deterministic algebraic transforms
- Educational tooling
- LLM tool backends
- Minimal math kernels in Go services

This is **not** a full SymPy clone.

It is a small, predictable symbolic engine.

---

## Design Goals

- Single file (`sympy.go`)
- Zero external dependencies
- Deterministic simplification (stable term ordering)
- Exact rational arithmetic (`math/big.Rat`)
- AI-embeddable
- Rule-based and transparent
- Compact API surface

---

## Features

### Core Expression Types

- `Num` (exact rational numbers)
- `Sym` (symbols)
- `Add`
- `Mul`
- `Pow`

All implement:

```go
type Expr interface {
    Simplify() Expr
    String() string
    Sub(varName string, value Expr) Expr
}
```

---

### Deterministic Simplification

- Flattens nested operations
- Combines numeric terms
- Stable lexicographic ordering
- Reproducible output for AI systems

```go
sympy.String(
    sympy.AddOf(sympy.S("x"), sympy.N(2), sympy.S("x")),
)
```

---

### Exact Rational Arithmetic

```go
sympy.F(1, 3)   // 1/3
sympy.F(5, 7)   // 5/7
```

Backed by `math/big.Rat`.

No float rounding unless explicitly requested.

---

### Polynomial Utilities

#### Degree

```go
sympy.Degree(expr, "x")
```

#### Coefficient Extraction

```go
coeffs := sympy.PolyCoeffs(expr, "x")
// map[degree]Rational
```

Enables:

- AI reasoning over polynomials
- Custom solvers
- Lightweight CAS behaviors

---

### Solvers

#### Linear

```go
sympy.SolveLinear(a, b)  // solves ax + b = 0
```

Exact rational result.

#### Quadratic

```go
sympy.SolveQuadratic(a, b, c)
```

Float64 roots.

---

### Integration (Rule-Based)

Supports:

- Power rule
- Sum rule
- Constant multiple rule
- Polynomial integration

```go
sympy.Integrate(expr, "x")
```

Pattern-based only. No Risch algorithm.

---

## Public Helpers

```go
sympy.Simplify(expr)
sympy.String(expr)
```

Designed for AI pipelines where deterministic formatting matters.

---

## Limitations

- No advanced polynomial factoring
- No symbolic matrix algebra
- No canonical simplification engine
- No symbolic limits beyond substitution
- No transcendental integration
- No parser (AST is built programmatically)

This is a **minimal symbolic kernel**, not a full CAS.

---

## Intended Use Cases

- AI tool execution layer
- LLM symbolic reasoning backend
- Go-based math microservices
- Deterministic algebra transforms
- Educational symbolic engines
- Lightweight research tools

---

## Philosophy

Small.
Predictable.
Deterministic.
Embeddable.

Not big.
Not magical.
Not opaque.

---

## Future Directions

- Differentiation
- Expanded polynomial solving
- Structured pattern matching
- Optional parser layer
- Symbolic equation solving

---

## License

MIT

---

Minimal symbolic math for Go.
Built for humans and AI.
