import math

import numpy as np
import pytest

from actipy._reader_support import java_arithmetic, java_long, parse_double, parse_long
from actipy.actigraph_reader import _java_round


# Expected values are Java's Double.parseDouble results (checked with JDK 17).
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0x1p1024", math.inf),
        ("-0x1p1024", -math.inf),
        ("0x1.fffffffffffff8p1023", math.inf),  # overflows only after rounding
        ("0x1p1024f", math.inf),
        ("0x1.fffffffffffffp1023", 1.7976931348623157e308),  # largest finite double
        ("0x1p-1080", 0.0),  # underflows to zero
        ("-0x1p-1080", -0.0),
        ("1e400", math.inf),
    ],
)
def test_parse_double_matches_java_at_overflow_and_underflow(text, expected):
    value = parse_double(text)
    assert value == expected
    assert math.copysign(1.0, value) == math.copysign(1.0, expected)


# Java's Double.parseDouble accepts ASCII digits only (checked with JDK 17), unlike float().
@pytest.mark.parametrize("text", ["１２", "١٢.5", "1e５"])
def test_parse_double_rejects_non_ascii_digits_like_java(text):
    with pytest.raises(ValueError, match="Invalid number"):
        parse_double(text)


def test_parse_long_accepts_ascii_digits_only():
    # Intentional difference: Java's Long.parseLong also accepts other Unicode decimal digits.
    assert parse_long("-12") == -12
    with pytest.raises(ValueError, match="Invalid integer"):
        parse_long("１２")


_LONG_MAX, _LONG_MIN = 2**63 - 1, -(2**63)

# (value, Java's (long) value, Java's Math.round(value)), checked with JDK 17.
_JAVA_LONG_AND_ROUND = [
    (math.nan, 0, 0),
    (math.inf, _LONG_MAX, _LONG_MAX),
    (-math.inf, _LONG_MIN, _LONG_MIN),
    (2.0**63, _LONG_MAX, _LONG_MAX),
    (-(2.0**63), _LONG_MIN, _LONG_MIN),
    (9.3e18, _LONG_MAX, _LONG_MAX),
    (-9.3e18, _LONG_MIN, _LONG_MIN),
    (9.2e18, 9_200_000_000_000_000_000, 9_200_000_000_000_000_000),
    (-9.2e18, -9_200_000_000_000_000_000, -9_200_000_000_000_000_000),
    (1.9999, 1, 2),
    (-1.9999, -1, -2),
    (-0.0, 0, 0),
    (0.49999999999999994, 0, 0),  # floor(x + 0.5) in doubles would give 1
    (0.5, 0, 1),
    (-0.5, 0, 0),
    (2.5, 2, 3),
    (-2.5, -2, -2),
    (-1.5, -1, -1),
    (4503599627370497.0, 4503599627370497, 4503599627370497),
    (1e-300, 0, 0),
    (-1e-300, 0, 0),
]


def test_java_long_matches_java_casts():
    values = np.array([value for value, _, _ in _JAVA_LONG_AND_ROUND])
    assert java_long(values).tolist() == [expected for _, expected, _ in _JAVA_LONG_AND_ROUND]


def test_java_round_then_long_matches_math_round():
    values = np.array([value for value, _, _ in _JAVA_LONG_AND_ROUND])
    with java_arithmetic():
        rounded = java_long(_java_round(values))
    assert rounded.tolist() == [expected for _, _, expected in _JAVA_LONG_AND_ROUND]
