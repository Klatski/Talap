from __future__ import annotations

import ast
import operator
from decimal import Decimal
from typing import Any


_BIN_OPS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_decimal(node: ast.AST, depth: int = 0) -> Decimal:
    if depth > 25:
        raise ValueError("Expression is too nested")
    if isinstance(node, ast.Expression):
        return _eval_decimal(node.body, depth + 1)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, str)):
            raise ValueError("Only numbers and decimal strings are allowed")
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_decimal(node.operand, depth + 1))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_decimal(node.left, depth + 1)
        right = _eval_decimal(node.right, depth + 1)
        if isinstance(node.op, ast.Pow) and abs(right) > Decimal("100"):
            raise ValueError("Exponent magnitude exceeds limit")
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError("Only arithmetic operations on decimals are permitted")


def safe_calculate(expression: Any) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    if len(expression) > 200:
        raise ValueError("Expression exceeds 200 characters")
    tree = ast.parse(expression, mode="eval")
    value = _eval_decimal(tree)
    if not value.is_finite():
        raise ValueError("Result is not finite")
    return format(value.normalize(), "f")