package gosymbol

import "strings"

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

// CreateBoxedAsciiPrettyPrint returns a boxed single-expression rendering.
func CreateBoxedAsciiPrettyPrint(expression Expr) string {
	if expression == nil {
		return "+-----+\n| nil |\n+-----+\n"
	}
	contentLines := strings.Split(strings.TrimRight(AsciiPrettyPrint(expression), "\n"), "\n")
	maximumWidth := 0
	for _, contentLine := range contentLines {
		if len(contentLine) > maximumWidth {
			maximumWidth = len(contentLine)
		}
	}
	var builder strings.Builder
	builder.WriteString("+")
	builder.WriteString(strings.Repeat("-", maximumWidth+2))
	builder.WriteString("+\n")
	for _, contentLine := range contentLines {
		builder.WriteString("| ")
		builder.WriteString(contentLine)
		if paddingCount := maximumWidth - len(contentLine); paddingCount > 0 {
			builder.WriteString(strings.Repeat(" ", paddingCount))
		}
		builder.WriteString(" |\n")
	}
	builder.WriteString("+")
	builder.WriteString(strings.Repeat("-", maximumWidth+2))
	builder.WriteString("+\n")
	return builder.String()
}
