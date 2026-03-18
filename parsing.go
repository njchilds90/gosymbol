package gosymbol

import (
	"fmt"
	"strings"
	"unicode"
)

type parseError struct {
	pos int
	msg string
}

func (e *parseError) Error() string {
	return fmt.Sprintf("parse error at %d: %s", e.pos, e.msg)
}

type tokenType int

const (
	tokenEOF tokenType = iota
	tokenNumber
	tokenIdent
	tokenPlus
	tokenMinus
	tokenStar
	tokenSlash
	tokenCaret
	tokenLParen
	tokenRParen
	tokenComma
)

type token struct {
	typ tokenType
	lit string
	pos int
}

type parser struct {
	input  string
	tokens []token
	pos    int
}

// Parse converts a simple infix expression like "3*x^2 + 1" into an Expr.
// It panics on invalid input; use ParseWithError for recoverable parsing.
func Parse(input string) Expr {
	expr, err := ParseWithError(input)
	if err != nil {
		panic(err)
	}
	return expr
}

// ParseWithError converts a simple infix expression into an Expr.
func ParseWithError(input string) (Expr, error) {
	tokens, err := tokenize(input)
	if err != nil {
		return nil, err
	}
	p := &parser{input: input, tokens: tokens}
	expr, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if p.peek().typ != tokenEOF {
		return nil, &parseError{pos: p.peek().pos, msg: "unexpected trailing input"}
	}
	return expr.Simplify(), nil
}

func tokenize(input string) ([]token, error) {
	var tokens []token
	for i := 0; i < len(input); {
		switch ch := rune(input[i]); {
		case unicode.IsSpace(ch):
			i++
		case unicode.IsDigit(ch):
			start := i
			i++
			for i < len(input) && unicode.IsDigit(rune(input[i])) {
				i++
			}
			if i < len(input) && input[i] == '/' {
				j := i + 1
				for j < len(input) && unicode.IsDigit(rune(input[j])) {
					j++
				}
				if j > i+1 {
					i = j
				}
			}
			if i < len(input) && input[i] == '.' {
				i++
				for i < len(input) && unicode.IsDigit(rune(input[i])) {
					i++
				}
			}
			tokens = append(tokens, token{typ: tokenNumber, lit: input[start:i], pos: start})
		case unicode.IsLetter(ch):
			start := i
			i++
			for i < len(input) {
				r := rune(input[i])
				if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
					break
				}
				i++
			}
			tokens = append(tokens, token{typ: tokenIdent, lit: input[start:i], pos: start})
		default:
			typ := tokenEOF
			switch input[i] {
			case '+':
				typ = tokenPlus
			case '-':
				typ = tokenMinus
			case '*':
				typ = tokenStar
			case '/':
				typ = tokenSlash
			case '^':
				typ = tokenCaret
			case '(':
				typ = tokenLParen
			case ')':
				typ = tokenRParen
			case ',':
				typ = tokenComma
			default:
				return nil, &parseError{pos: i, msg: fmt.Sprintf("unexpected character %q", input[i])}
			}
			tokens = append(tokens, token{typ: typ, lit: input[i : i+1], pos: i})
			i++
		}
	}
	tokens = append(tokens, token{typ: tokenEOF, pos: len(input)})
	return tokens, nil
}

func (p *parser) peek() token {
	if p.pos >= len(p.tokens) {
		return token{typ: tokenEOF, pos: len(p.input)}
	}
	return p.tokens[p.pos]
}

func (p *parser) next() token {
	tok := p.peek()
	p.pos++
	return tok
}

func (p *parser) match(tt tokenType) bool {
	if p.peek().typ == tt {
		p.pos++
		return true
	}
	return false
}

func (p *parser) expect(tt tokenType, msg string) error {
	if p.peek().typ != tt {
		return &parseError{pos: p.peek().pos, msg: msg}
	}
	p.pos++
	return nil
}

func (p *parser) parseExpr() (Expr, error) {
	left, err := p.parseTerm()
	if err != nil {
		return nil, err
	}
	for {
		switch p.peek().typ {
		case tokenPlus:
			p.next()
			right, err := p.parseTerm()
			if err != nil {
				return nil, err
			}
			left = AddOf(left, right)
		case tokenMinus:
			p.next()
			right, err := p.parseTerm()
			if err != nil {
				return nil, err
			}
			left = AddOf(left, MulOf(N(-1), right))
		default:
			return left, nil
		}
	}
}

func (p *parser) parseTerm() (Expr, error) {
	left, err := p.parsePower()
	if err != nil {
		return nil, err
	}
	for {
		switch p.peek().typ {
		case tokenStar:
			p.next()
			right, err := p.parsePower()
			if err != nil {
				return nil, err
			}
			left = MulOf(left, right)
		case tokenSlash:
			p.next()
			right, err := p.parsePower()
			if err != nil {
				return nil, err
			}
			left = MulOf(left, PowOf(right, N(-1)))
		default:
			return left, nil
		}
	}
}

func (p *parser) parsePower() (Expr, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	if p.match(tokenCaret) {
		right, err := p.parsePower()
		if err != nil {
			return nil, err
		}
		return PowOf(left, right), nil
	}
	return left, nil
}

func (p *parser) parseUnary() (Expr, error) {
	if p.match(tokenMinus) {
		expr, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return MulOf(N(-1), expr), nil
	}
	if p.match(tokenPlus) {
		return p.parseUnary()
	}
	return p.parsePrimary()
}

func (p *parser) parsePrimary() (Expr, error) {
	tok := p.peek()
	switch tok.typ {
	case tokenNumber:
		p.next()
		return parseNumberLiteral(tok.lit, tok.pos)
	case tokenIdent:
		p.next()
		if p.match(tokenLParen) {
			arg, err := p.parseExpr()
			if err != nil {
				return nil, err
			}
			if err := p.expect(tokenRParen, "expected ')'"); err != nil {
				return nil, err
			}
			return makeFunc(tok.lit, arg)
		}
		return S(tok.lit), nil
	case tokenLParen:
		p.next()
		expr, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if err := p.expect(tokenRParen, "expected ')'"); err != nil {
			return nil, err
		}
		return expr, nil
	default:
		return nil, &parseError{pos: tok.pos, msg: "expected expression"}
	}
}

func parseNumberLiteral(lit string, pos int) (Expr, error) {
	if strings.Contains(lit, "/") {
		var p, q int64
		if _, err := fmt.Sscanf(lit, "%d/%d", &p, &q); err != nil {
			return nil, &parseError{pos: pos, msg: "invalid rational literal"}
		}
		return F(p, q), nil
	}
	if strings.Contains(lit, ".") {
		var f float64
		if _, err := fmt.Sscanf(lit, "%f", &f); err != nil {
			return nil, &parseError{pos: pos, msg: "invalid float literal"}
		}
		return NFloat(f), nil
	}
	var n int64
	if _, err := fmt.Sscanf(lit, "%d", &n); err != nil {
		return nil, &parseError{pos: pos, msg: "invalid integer literal"}
	}
	return N(n), nil
}

func makeFunc(name string, arg Expr) (Expr, error) {
	switch name {
	case "sin":
		return SinOf(arg), nil
	case "cos":
		return CosOf(arg), nil
	case "tan":
		return TanOf(arg), nil
	case "exp":
		return ExpOf(arg), nil
	case "ln":
		return LnOf(arg), nil
	case "abs":
		return AbsOf(arg), nil
	case "sqrt":
		return SqrtOf(arg), nil
	case "asin":
		return AsinOf(arg), nil
	case "acos":
		return AcosOf(arg), nil
	case "atan":
		return AtanOf(arg), nil
	case "sinh":
		return SinhOf(arg), nil
	case "cosh":
		return CoshOf(arg), nil
	case "tanh":
		return TanhOf(arg), nil
	case "asinh":
		return AsinhOf(arg), nil
	case "acosh":
		return AcoshOf(arg), nil
	case "atanh":
		return AtanhOf(arg), nil
	default:
		return nil, fmt.Errorf("unsupported function %q", name)
	}
}
