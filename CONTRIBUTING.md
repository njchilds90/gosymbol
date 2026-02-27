# Contributing to go-symbol

Thank you for your interest in contributing. go-symbol is a minimal symbolic math kernel and contributions are welcome — especially ones that stay true to the design goals.

---

## Design Principles

Before contributing, please understand what gosymbol is and isn't:

**gosymbol IS:**
- A minimal, single-file symbolic kernel
- Zero-dependency
- Deterministic (same input → same output, always)
- AI/agent embeddable
- Educational and readable

**gosymbol IS NOT:**
- A full SymPy port
- A numeric computing library
- A CAS with every possible feature
- A replacement for Mathematica/Maple

If your contribution would turn `sympy.go` into a multi-thousand line file that requires external dependencies, it's probably better suited as a separate package that imports gosymbol.

---

## Getting Started

```bash
git clone https://github.com/njchilds90/gosymbol.git
cd gosymbol
go test ./...
```

All tests should pass before and after your change.

---

## Contribution Types

### Bug Fixes
Always welcome. Please include a test that fails before your fix and passes after.

### New Expression Types
New node types (e.g., `Floor`, `Ceiling`, `Piecewise`) must:
- Implement the full `Expr` interface (Simplify, String, LaTeX, Sub, Diff, Eval, Equal, exprType, toJSON)
- Be handled in `FromJSON`
- Have full test coverage
- Be handled in `HandleToolCall` if they should be MCP-accessible

### New Solver/Algorithm
- Prefer exact rational arithmetic over floats
- Include at least 3 test cases including edge cases
- Document limitations clearly

### New MCP Tool
- Add to `HandleToolCall`'s switch statement
- Add to the AGENTS.md tool table and a worked example
- Add tests for the tool call path

---

## Code Style

- Standard Go formatting (`gofmt`)
- Comments on all exported types and functions
- No external dependencies (stdlib only)
- Prefer immutable expression trees (don't mutate, return new nodes)
- Panic only for programmer errors (e.g., zero denominator), not for unexpected inputs

---

## Test Coverage

All new code must have tests in `gosymbol_test.go`. Tests should:
- Use `package gosymbol_test` (black-box testing)
- Cover the happy path
- Cover edge cases (zero, one, negative, rational)
- Cover error/failure cases
- Be deterministic (no randomness)

Run tests with race detector:
```bash
go test -race ./...
```

---

## Pull Request Checklist

- [ ] `go test ./...` passes
- [ ] `go test -race ./...` passes
- [ ] `gofmt -l .` produces no output
- [ ] New exported symbols have doc comments
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] AGENTS.md updated if any MCP tools were added/changed
- [ ] README.md updated if API surface changed

---

## Reporting Issues

Please include:
1. Go version (`go version`)
2. A minimal code example reproducing the issue
3. Expected vs actual output

---

## License

By contributing, you agree your contributions are licensed under MIT.