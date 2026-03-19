package gosymbol

import (
	"errors"
	"fmt"
)

// SymbolicMathematicsSentinelError marks errors originating from symbolic operations.
var SymbolicMathematicsSentinelError = errors.New("symbolic mathematics error")

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

// Unwrap returns the underlying sentinel or nested cause for errors.Is and errors.As support.
func (symbolicMathematicsError *SymbolicMathematicsError) Unwrap() error {
	if symbolicMathematicsError == nil {
		return nil
	}
	if symbolicMathematicsError.Cause == nil {
		return SymbolicMathematicsSentinelError
	}
	return errors.Join(SymbolicMathematicsSentinelError, symbolicMathematicsError.Cause)
}

func newSymbolicMathematicsError(operationName, message string, cause error) error {
	return &SymbolicMathematicsError{
		OperationName: operationName,
		Message:       message,
		Cause:         cause,
	}
}

// IsSymbolic reports whether the error belongs to the symbolic mathematics error domain.
func IsSymbolic(err error) bool {
	return errors.Is(err, SymbolicMathematicsSentinelError)
}
