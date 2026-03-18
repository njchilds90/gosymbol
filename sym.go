package gosymbol

// Assumptions captures optional symbol properties used by simplification.
type Assumptions struct {
	Real     bool `json:"real,omitempty"`
	Positive bool `json:"positive,omitempty"`
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
	if s.assumptions.Real || s.assumptions.Positive {
		result["assumptions"] = map[string]interface{}{
			"real":     s.assumptions.Real,
			"positive": s.assumptions.Positive,
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
