//go:build js && wasm

package gosymbol

import "syscall/js"

// RegisterWebAssemblyExample installs a small JavaScript-callable simplification entry point.
func RegisterWebAssemblyExample() {
	js.Global().Set("simplifyExpressionWithGoSymbol", js.FuncOf(func(this js.Value, arguments []js.Value) interface{} {
		if len(arguments) == 0 {
			return ""
		}
		expression, parseError := ParseWithError(arguments[0].String())
		if parseError != nil {
			return parseError.Error()
		}
		return Simplify(expression).String()
	}))
}
