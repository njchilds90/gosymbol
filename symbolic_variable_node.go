package gosymbol

// Assumptions captures optional symbol properties used by simplification.
type Assumptions struct {
	Real     bool `json:"real,omitempty"`
	Positive bool `json:"positive,omitempty"`
	Negative bool `json:"negative,omitempty"`
	Integer  bool `json:"integer,omitempty"`
	NonZero  bool `json:"non_zero,omitempty"`
}

// Sym is a symbolic variable.
type Sym struct {
	name        string
	assumptions Assumptions
}

// S constructs a symbolic variable with no assumptions.
func S(name string) *Sym { return &Sym{name: name} }

// SAssume constructs a symbolic variable with assumptions.
func SAssume(name string, assumptions Assumptions) *Sym {
	return &Sym{name: name, assumptions: assumptions}
}

func (s *Sym) Simplify() Expr { return s }
func (s *Sym) String() string { return s.name }
func (s *Sym) LaTeX() string  { return s.name }
func (s *Sym) Eval() (*Num, bool) {
	return nil, false
}
func (s *Sym) Equal(other Expr) bool {
	o, ok := other.(*Sym)
	return ok && s.name == o.name && s.assumptions == o.assumptions
}
func (s *Sym) exprType() string         { return "sym" }
func (s *Sym) Name() string             { return s.name }
func (s *Sym) Assumptions() Assumptions { return s.assumptions }
func (s *Sym) toJSON() map[string]interface{} {
	result := map[string]interface{}{"type": "sym", "name": s.name}
	if s.assumptions.Real || s.assumptions.Positive || s.assumptions.Negative || s.assumptions.Integer || s.assumptions.NonZero {
		result["assumptions"] = map[string]interface{}{
			"real":     s.assumptions.Real,
			"positive": s.assumptions.Positive,
			"negative": s.assumptions.Negative,
			"integer":  s.assumptions.Integer,
			"non_zero": s.assumptions.NonZero,
		}
	}
	return result
}
func (s *Sym) Sub(varName string, value Expr) Expr {
	if s.name == varName {
		return value
	}
	return s
}
func (s *Sym) Diff(varName string) Expr {
	if s.name == varName {
		return N(1)
	}
	return N(0)
}

// CreateEnhancedAssumptions constructs an assumptions bundle with the provided properties.
func CreateEnhancedAssumptions(real, positive, negative, integer, nonZero bool) Assumptions {
	return Assumptions{
		Real:     real,
		Positive: positive,
		Negative: negative,
		Integer:  integer,
		NonZero:  nonZero,
	}
}

// ApplyAssumptionsToSymbolicVariable returns a copy of the symbolic variable with merged assumptions.
func ApplyAssumptionsToSymbolicVariable(symbolicVariable *Sym, assumptions Assumptions) *Sym {
	if symbolicVariable == nil {
		return nil
	}
	mergedAssumptions := symbolicVariable.assumptions
	mergedAssumptions.Real = mergedAssumptions.Real || assumptions.Real
	mergedAssumptions.Positive = mergedAssumptions.Positive || assumptions.Positive
	mergedAssumptions.Negative = mergedAssumptions.Negative || assumptions.Negative
	mergedAssumptions.Integer = mergedAssumptions.Integer || assumptions.Integer
	mergedAssumptions.NonZero = mergedAssumptions.NonZero || assumptions.NonZero || assumptions.Positive || assumptions.Negative
	return &Sym{name: symbolicVariable.name, assumptions: mergedAssumptions}
}

// SymbolicVariableIsKnownReal reports whether the symbolic variable has a real assumption.
func SymbolicVariableIsKnownReal(symbolicVariable *Sym) bool {
	return symbolicVariable != nil && symbolicVariable.assumptions.Real
}

// SymbolicVariableIsKnownInteger reports whether the symbolic variable has an integer assumption.
func SymbolicVariableIsKnownInteger(symbolicVariable *Sym) bool {
	return symbolicVariable != nil && symbolicVariable.assumptions.Integer
}

// SymbolicVariableIsKnownNonZero reports whether the symbolic variable is assumed to be nonzero.
func SymbolicVariableIsKnownNonZero(symbolicVariable *Sym) bool {
	return symbolicVariable != nil && (symbolicVariable.assumptions.NonZero || symbolicVariable.assumptions.Positive || symbolicVariable.assumptions.Negative)
}
