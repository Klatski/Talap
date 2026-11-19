"""Калькулятор: только арифметика и десятичные числа, никакого кода и файлов."""

import time
import unittest
from decimal import Decimal

import env_helpers  # noqa: F401
from env.calculator import CalculatorError, evaluate
from env.constants import CALC_DETAILS


class CalculatorValues(unittest.TestCase):
    def test_arithmetic(self):
        cases = {
            "1+2*3": "7", "(1+2)*3": "9", "0.1+0.2": "0.3", "-5 + 2": "-3", "10/4": "2.5",
            "--1": "1", "+-+1": "-1", "  7  ": "7", "2.": "2", ".5*2": "1.0", "-0": "0", "0*-1": "0",
            "abs(-3.5)": "3.5", "round_half_up(2.5)": "3", "round_half_even(2.5)": "2",
            "round_half_up(-2.5)": "-3", "round_half_even(-2.5)": "-2", "round_half_up(1.005, 2)": "1.01",
            "round_half_even(1.005, 2)": "1.00", "round_half_up(1234.5, -2)": "1200",
        }
        for expr, want in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(evaluate(expr), want)

    def test_conversion_is_exact_decimal(self):
        got = evaluate("23208 * 516.3571")
        self.assertEqual(Decimal(got), Decimal(23208) * Decimal("516.3571"))
        self.assertEqual(evaluate("round_half_up(23208 * 516.3571 / 100)"), "119836")

    def test_no_exponent_in_output(self):
        for expr in ("1/3", "1/7*1000000", "0.000001*0.000001", "99999999999999999999*99999999"):
            with self.subTest(expr=expr):
                out = evaluate(expr)
                self.assertNotIn("E", out.upper())
                Decimal(out)

    def test_division_precision(self):
        self.assertTrue(evaluate("1/3").startswith("0.3333333333"))


class CalculatorRejects(unittest.TestCase):
    REJECT = {
        "": "empty", "   ": "empty", "2**3": "syntax", "2^3": "bad_char", "1e5": "syntax",
        "__import__('os')": "bad_char", "open('x')": "bad_char", "a": "unknown_function", "abs": "syntax",
        "1;2": "bad_char", "x=1": "bad_char", "[1]": "bad_char", "1,2": "syntax", "abs()": "syntax",
        "abs(1,2)": "bad_arity", "round_half_up(1, 0.5)": "bad_ndigits", "round_half_up(1, 13)": "bad_ndigits",
        "1/0": "division_by_zero", "1/(2-2)": "division_by_zero", "9" * 41: "number_too_long",
        "1" + "0" * 30: "too_large", "10000000000000000*100000000000000": "too_large",
        "(" * 33 + "1" + ")" * 33: "too_deep", "-" * 40 + "1": "too_deep",
        "1+" * 70 + "1": "too_many_tokens", "1" * 300: "too_long",
        "١+١": "bad_char", "１": "bad_char", "1_000": "syntax", "0x10": "syntax",
        "1.2.3": "syntax", "max(1,2)": "unknown_function", "ABS(1)": "bad_char", "1 2": "syntax",
        "lambda: 1": "bad_char", "(1": "syntax", "1)": "syntax", "()": "syntax", "1 if 1 else 2": "syntax",
        "round_half_up(1e3)": "syntax", "abs(-1)(2)": "syntax", "\x00": "bad_char",
    }

    def test_rejected_with_catalogue_detail(self):
        for expr, detail in self.REJECT.items():
            with self.subTest(expr=expr[:40]):
                with self.assertRaises(CalculatorError) as ctx:
                    evaluate(expr)
                self.assertEqual(ctx.exception.detail, detail)
                self.assertIn(ctx.exception.detail, CALC_DETAILS)

    def test_non_strings(self):
        for value in (123, None, ["1"], {"e": "1"}, b"1+1", 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(CalculatorError) as ctx:
                    evaluate(value)
                self.assertEqual(ctx.exception.detail, "not_a_string")

    def test_worst_cases_are_fast(self):
        worst = ["(" * 32 + "1" + ")" * 32, "9" * 40 + "*" + "9" * 40, "1/3*" * 30 + "1", "9" * 30 + "/0.0000000001"]
        t0 = time.perf_counter()
        for expr in worst:
            try:
                evaluate(expr)
            except CalculatorError:
                pass
        self.assertLess(time.perf_counter() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
