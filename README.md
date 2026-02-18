# go-sympy

A compact symbolic algebra engine written in pure Go.

`go-sympy` provides basic symbolic mathematics capabilities in a single-file, dependency-free package. It is designed to be small, embeddable, and usable by both humans and AI agents.

This is **not** a full SymPy port. It is a lightweight symbolic system for Go projects that need algebraic manipulation without Python.

---

## ✨ Features

- Symbolic expressions
- Differentiation
- Simplification
- Substitution
- Basic integration
- Taylor series expansion
- Linear and quadratic equation solving
- Matrix operations
- LaTeX output
- Expression parsing from strings
- Built-in constants: `pi`, `e`
- Trig + exp + log + sqrt + abs

All implemented in a **single file** with no external dependencies.

---

## 📦 Installation

```bash
go get github.com/njchilds90/go-sympy

Then:
import "github.com/njchilds90/go-sympy"


🚀 Quick Example
package main

import (
	"fmt"
	"github.com/njchilds90/go-sympy"
)

func main() {
	x := sympy.Symbol("x")

	expr := sympy.Add(
		sympy.Pow(x, sympy.Number(2)),
		sympy.Number(3),
	)

	fmt.Println(expr)                 // x^2+3
	fmt.Println(expr.Diff(x))         // 2*x
	fmt.Println(expr.Eval(map[string]float64{"x": 2})) // 7
}

🧠 Expression Construction
Expressions are built using constructors:
x := Symbol("x")

Add(a, b)
Sub(a, b)
Mul(a, b)
Div(a, b)
Pow(a, b)

Sin(x)
Cos(x)
Tan(x)
Exp(x)
Ln(x)
Sqrt(x)
Abs(x)

🔍 Parsing Expressions

You can parse strings:
expr := sympy.Parse("sin(x)^2 + cos(x)^2")
fmt.Println(expr.Simplify())   // 1
Supported operators:

+

-

*

/

^

Supported functions:

sin

cos

tan

exp

ln

sqrt

abs

Constants:

pi

e


📉 Differentiation:
x := sympy.Symbol("x")
f := sympy.Parse("x^3 + 2*x")

df := f.Diff(x).Simplify()

fmt.Println(df)   // 3*x^2+2

📈 Integration:
x := sympy.Symbol("x")
f := sympy.Sin(x)

F := sympy.Integrate(f, x)

fmt.Println(F)    // -cos(x)
val := sympy.IntegrateDefinite(sympy.Sin(x), x, 0, math.Pi)

📊 Taylor Series:
x := sympy.Symbol("x")
f := sympy.Exp(x)

t := sympy.Taylor(f, x, sympy.Number(0), 5)
fmt.Println(t)


🧮 Solving Equations
Supports linear and quadratic equations:
x := sympy.Symbol("x")
eq := sympy.Parse("x^2 - 4")

roots := sympy.Solve(eq, x)

🧩 Matrix Support:
A := sympy.Matrix{
	{sympy.Number(1), sympy.Number(2)},
	{sympy.Number(3), sympy.Number(4)},
}

B := sympy.Matrix{
	{sympy.Number(5), sympy.Number(6)},
	{sympy.Number(7), sympy.Number(8)},
}

C := sympy.MatrixMul(A, B)
fmt.Println(C)
