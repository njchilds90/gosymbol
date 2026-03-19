package gosymbol

// ConstantNode represents an arbitrary symbolic constant that does not vary with
// respect to differentiation variables.
type ConstantNode struct {
	name string
}

// CreateConstantNode constructs a symbolic constant node.
func CreateConstantNode(name string) *ConstantNode {
	return &ConstantNode{name: name}
}

// CreateArbitraryIntegrationConstant constructs the standard indefinite
// integration constant.
func CreateArbitraryIntegrationConstant() *ConstantNode {
	return CreateConstantNode("C")
}

func (constantNode *ConstantNode) Simplify() Expr        { return constantNode }
func (constantNode *ConstantNode) Canonicalize() Expr    { return Canonicalize(constantNode) }
func (constantNode *ConstantNode) String() string        { return constantNode.name }
func (constantNode *ConstantNode) LaTeX() string         { return constantNode.name }
func (constantNode *ConstantNode) Sub(string, Expr) Expr { return constantNode }
func (constantNode *ConstantNode) Diff(string) Expr      { return N(0) }
func (constantNode *ConstantNode) Eval() (*Num, bool)    { return nil, false }
func (constantNode *ConstantNode) Equal(other Expr) bool {
	otherConstantNode, isConstantNode := other.(*ConstantNode)
	return isConstantNode && constantNode.name == otherConstantNode.name
}
func (constantNode *ConstantNode) exprType() string { return "constant" }
func (constantNode *ConstantNode) toJSON() map[string]interface{} {
	return map[string]interface{}{"type": "constant", "name": constantNode.name}
}
