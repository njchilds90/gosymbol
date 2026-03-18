package gosymbol

import "fmt"

// SymbolicMathematicsError describes a structured symbolic mathematics failure.
type SymbolicMathematicsError struct {
	OperationName string
	Message       string
	Cause         error
}

func (symbolicMathematicsError *SymbolicMathematicsError) Error() string {
	if symbolicMathematicsError == nil {
		return ""
	}
	if symbolicMathematicsError.Cause == nil {
		return fmt.Sprintf("%s: %s", symbolicMathematicsError.OperationName, symbolicMathematicsError.Message)
	}
	return fmt.Sprintf(
		"%s: %s: %v",
		symbolicMathematicsError.OperationName,
		symbolicMathematicsError.Message,
		symbolicMathematicsError.Cause,
	)
}

func newSymbolicMathematicsError(operationName, message string, cause error) error {
	return &SymbolicMathematicsError{
		OperationName: operationName,
		Message:       message,
		Cause:         cause,
	}
}
