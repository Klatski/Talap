import unittest
from datetime import date, datetime, timezone
from decimal import Decimal

import gen_helpers  # noqa: F401  (sys.path)
from generator.money import convert_minor, format_rate
from generator.timeutil import format_offset, local_date, parse_iso, parse_offset, to_iso


class MoneyTests(unittest.TestCase):
    def test_same_currency_is_identity(self):
        self.assertEqual(convert_minor(12345, "KZT", "KZT", "999", "half_up"), 12345)

    def test_exponents_are_applied(self):
        # 10.00 USD * 500 = 5000.00 KZT
        self.assertEqual(convert_minor(1000, "USD", "KZT", "500", "half_up"), 500000)
        # 1000 JPY (exp 0) * 3.3 = 3300.00 KZT
        self.assertEqual(convert_minor(1000, "JPY", "KZT", "3.3", "half_up"), 330000)
        # 1.000 KWD (exp 3) * 1600 = 1600.00 KZT
        self.assertEqual(convert_minor(1000, "KWD", "KZT", "1600", "half_up"), 160000)
        # 100.00 USD -> JPY (exp 0) at 150.5 = 15050 JPY
        self.assertEqual(convert_minor(10000, "USD", "JPY", "150.5", "half_up"), 15050)

    def test_rounding_modes_differ_only_on_ties(self):
        # 1 * 0.5 = 0.5 -> half_up 1, half_even 0 ; 3 * 0.5 = 1.5 -> both 2
        self.assertEqual(convert_minor(1, "USD", "EUR", "0.5", "half_up"), 1)
        self.assertEqual(convert_minor(1, "USD", "EUR", "0.5", "half_even"), 0)
        self.assertEqual(convert_minor(3, "USD", "EUR", "0.5", "half_up"), 2)
        self.assertEqual(convert_minor(3, "USD", "EUR", "0.5", "half_even"), 2)

    def test_float_amount_rejected(self):
        with self.assertRaises(TypeError):
            convert_minor(10.0, "USD", "KZT", "500", "half_up")
        with self.assertRaises(TypeError):
            convert_minor(True, "USD", "KZT", "500", "half_up")

    def test_rate_strings_are_plain_decimals(self):
        for value in (Decimal("498.123456"), Decimal("0.00205"), Decimal("0.0000123")):
            text = format_rate(value)
            self.assertNotIn("E", text)
            self.assertRegex(text, r"^\d+\.\d+$")


class TimeTests(unittest.TestCase):
    def test_offsets_roundtrip(self):
        for minutes in (-660, -300, 0, 330, 345, 780):
            self.assertEqual(parse_offset(format_offset(minutes)), minutes)

    def test_iso_has_explicit_offset(self):
        instant = datetime(2026, 3, 31, 19, 30, tzinfo=timezone.utc)
        self.assertEqual(to_iso(instant, 0), "2026-03-31T19:30:00+00:00")
        self.assertEqual(to_iso(instant, 300), "2026-04-01T00:30:00+05:00")
        self.assertEqual(parse_iso(to_iso(instant, 300)), instant)

    def test_local_date_depends_on_timezone(self):
        instant = datetime(2026, 3, 31, 19, 30, tzinfo=timezone.utc)
        self.assertEqual(local_date(instant, 0), date(2026, 3, 31))
        self.assertEqual(local_date(instant, 300), date(2026, 4, 1))

    def test_naive_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            parse_iso("2026-03-01T10:00:00")


if __name__ == "__main__":
    unittest.main()
