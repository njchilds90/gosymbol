# Changelog

All notable changes to gosympy are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added
- `Func` node type: `SinOf`, `CosOf`, `TanOf`, `ExpOf`, `LnOf`, `AbsOf`, `SqrtOf`
- Chain rule differentiation for all Func types
- `Expand()` — algebraic expansion with distribution and `(a+b)^n` support
- `FreeSymbols()` — returns the set of free variable names in an expression
- `Diff2()` and `DiffN()` — second and nth derivatives
- `TaylorSeries()` — Taylor expansion around a point
- `DefiniteIntegrate()` — numerical definite integration (10-point Gaussian quadrature)
- `SolveLinearSystem2x2()` — exact 2×2 linear system solver via Cramer's rule
- `Equation` type with `Eq()`, `String()`, `LaTeX()`, `Residual()`
- `Equal()` method on all Expr types — structural equality check
- `Eval()` method on all Expr types — full numeric evaluation
- `LaTeX()` method on all Expr types — LaTeX rendering
- `ToJSON()` / `FromJSON()` — full expression serialization round-trip
- `HandleToolCall()` — unified MCP-compatible tool dispatch
- `MCPToolSpec()` — JSON schema of available tools for agent registration
- `AGENTS.md` — dedicated AI agent integration guide with worked examples
- `CONTRIBUTING.md`
- `CHANGELOG.md`
- GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- Comprehensive test suite covering all new functionality
  
### Changed
- Package renamed from `sympy` to `gosympy` for import clarity
- `Mul.Simplify()` now sorts factors lexicographically for deterministic output
- `Add.Simplify()` now handles `Sym` like-term collection directly
 
---

## [0.1.0] — 2026-02-18

### Added
- Initial prototype release
- `Num`, `Sym`, `Add`, `Mul`, `Pow` expression types
- `Simplify()`, `String()`, `Sub()` methods
- `SolveLinear()`, `SolveQuadratic()`
- `Integrate()` (polynomial and constant rules)
- `Degree()`, `PolyCoeffs()`
- Exact rational arithmetic via `math/big.Rat`
- `N()`, `F()` constructors
- MIT license