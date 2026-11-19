"""Безопасный калькулятор: только арифметика над десятичными числами.

Никакого eval/exec/ast-компиляции: собственный токенизатор и рекурсивный спуск.
Грамматика:
    expr    := term (('+' | '-') term)*
    term    := unary (('*' | '/') unary)*
    unary   := ('+' | '-') unary | primary
    primary := NUMBER | '(' expr ')' | FUNC '(' expr (',' expr)? ')'
    NUMBER  := [0-9]+ ('.' [0-9]*)? | '.' [0-9]+        (только ASCII-цифры, без экспоненты)
    FUNC    := abs | round_half_up | round_half_even

Числа — Decimal (точность 34 знака, как decimal128), float не используется нигде.
Ограничения: длина 256 символов, 128 токенов, вложенность 32, число до 40 цифр,
|значение| < 1e30 на каждом шаге, '**' и прочие операторы запрещены.
"""

from __future__ import annotations

import re
from decimal import (
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

MAX_CHARS = 256
MAX_TOKENS = 128
MAX_DEPTH = 32
MAX_DIGITS = 40
LIMIT = Decimal("1e30")
FUNCTIONS = {"abs": (1, 1), "round_half_up": (1, 2), "round_half_even": (1, 2)}

_TOKEN_RE = re.compile(r"\s*(?:(?P<num>[0-9]+(?:\.[0-9]*)?|\.[0-9]+)|(?P<name>[a-z_]+)|(?P<op>[-+*/(),]))")
_CTX = Context(prec=34, Emax=999, Emin=-999, traps=[InvalidOperation, DivisionByZero, Overflow])


class CalculatorError(ValueError):
    """detail — ключ из env.constants.CALC_DETAILS."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise CalculatorError("bad_char")
        pos = m.end()
        if m.group("num") is not None:
            digits = sum(ch.isdigit() for ch in m.group("num"))
            if digits > MAX_DIGITS:
                raise CalculatorError("number_too_long")
            tokens.append(("num", m.group("num")))
        elif m.group("name") is not None:
            tokens.append(("name", m.group("name")))
        else:
            tokens.append(("op", m.group("op")))
        if len(tokens) > MAX_TOKENS:
            raise CalculatorError("too_many_tokens")
    return tokens


def _check(value: Decimal) -> Decimal:
    if not value.is_finite() or abs(value) >= LIMIT:
        raise CalculatorError("too_large")
    return value


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.depth = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, kind: str | None = None, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (kind and tok[0] != kind) or (value and tok[1] != value):
            raise CalculatorError("syntax")
        self.pos += 1
        return tok

    def enter(self) -> None:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise CalculatorError("too_deep")

    def expr(self) -> Decimal:
        self.enter()
        value = self.term()
        while (tok := self.peek()) and tok in (("op", "+"), ("op", "-")):
            self.pos += 1
            rhs = self.term()
            value = _check(_CTX.add(value, rhs) if tok[1] == "+" else _CTX.subtract(value, rhs))
        self.depth -= 1
        return value

    def term(self) -> Decimal:
        value = self.unary()
        while (tok := self.peek()) and tok in (("op", "*"), ("op", "/")):
            self.pos += 1
            rhs = self.unary()
            if tok[1] == "*":
                value = _check(_CTX.multiply(value, rhs))
            else:
                if rhs.is_zero():
                    raise CalculatorError("division_by_zero")
                value = _check(_CTX.divide(value, rhs))
        return value

    def unary(self) -> Decimal:
        tok = self.peek()
        if tok in (("op", "+"), ("op", "-")):
            self.pos += 1
            self.enter()
            value = self.unary()
            self.depth -= 1
            return value if tok[1] == "+" else _CTX.minus(value)
        return self.primary()

    def primary(self) -> Decimal:
        tok = self.peek()
        if tok is None:
            raise CalculatorError("syntax")
        if tok[0] == "num":
            self.pos += 1
            text = tok[1]
            if text.endswith("."):
                text = text[:-1]
            try:
                return _check(Decimal(text))
            except InvalidOperation as exc:  # pragma: no cover — регулярка уже отсекла
                raise CalculatorError("bad_number") from exc
        if tok == ("op", "("):
            self.pos += 1
            value = self.expr()
            self.take("op", ")")
            return value
        if tok[0] == "name":
            name = tok[1]
            if name not in FUNCTIONS:
                raise CalculatorError("unknown_function")
            self.pos += 1
            self.take("op", "(")
            args = [self.expr()]
            while self.peek() == ("op", ","):
                self.pos += 1
                args.append(self.expr())
            self.take("op", ")")
            lo, hi = FUNCTIONS[name]
            if not lo <= len(args) <= hi:
                raise CalculatorError("bad_arity")
            return self.call(name, args)
        raise CalculatorError("syntax")

    @staticmethod
    def call(name: str, args: list[Decimal]) -> Decimal:
        if name == "abs":
            return _CTX.abs(args[0])
        ndigits = 0
        if len(args) == 2:
            nd = args[1]
            if nd != nd.to_integral_value() or not -12 <= nd <= 12:
                raise CalculatorError("bad_ndigits")
            ndigits = int(nd)
        quantum = Decimal(1).scaleb(-ndigits)
        mode = ROUND_HALF_UP if name == "round_half_up" else ROUND_HALF_EVEN
        try:
            return _check(args[0].quantize(quantum, rounding=mode, context=_CTX))
        except InvalidOperation as exc:
            raise CalculatorError("invalid_operation") from exc


def evaluate(expression: object) -> str:
    """Вычислить выражение; результат — десятичная строка без экспоненты."""
    if not isinstance(expression, str):
        raise CalculatorError("not_a_string")
    if len(expression) > MAX_CHARS:
        raise CalculatorError("too_long")
    tokens = _tokenize(expression)
    if not tokens:
        raise CalculatorError("empty")
    parser = _Parser(tokens)
    try:
        value = parser.expr()
    except (InvalidOperation, Overflow) as exc:
        raise CalculatorError("invalid_operation") from exc
    except DivisionByZero as exc:
        raise CalculatorError("division_by_zero") from exc
    except RecursionError as exc:  # pragma: no cover — глубина ограничена раньше
        raise CalculatorError("too_deep") from exc
    if parser.pos != len(tokens):
        raise CalculatorError("syntax")
    if value.is_zero():
        value = abs(value)
    return format(value, "f")
