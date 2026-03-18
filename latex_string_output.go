package gosymbol

func latexFuncName(name string) string {
	switch name {
	case "sin", "cos", "tan", "exp", "ln", "sinh", "cosh", "tanh":
		return `\` + name
	case "asin":
		return `\arcsin`
	case "acos":
		return `\arccos`
	case "atan":
		return `\arctan`
	case "asinh":
		return `\operatorname{asinh}`
	case "acosh":
		return `\operatorname{acosh}`
	case "atanh":
		return `\operatorname{atanh}`
	default:
		return `\operatorname{` + name + `}`
	}
}

// PrettyPrint returns a compact, human-friendly single-line rendering.
func PrettyPrint(e Expr) string { return "  " + e.String() + "\n" }
