"""
Shared helpers for the device readers.

The readers replace the former Java parsers and aim to return the same values: the real-device
test fixtures match the Java output byte for byte, and the tests check edge cases against Java's
results. These helpers mirror Java semantics where Python differs: number parsing, saturating
float-to-long and millisecond-to-nanosecond conversion, silent IEEE floating-point arithmetic, and
proleptic calendar validation.

Known, intentional differences from the Java readers:

- Integer fields accept ASCII digits only; Java's ``Long.parseLong`` also accepts other Unicode
  decimal digits (e.g. fullwidth digits).
- A .gt3x archive member whose CRC does not match is rejected; the Java reader read it anyway.
- When several packets are malformed, the error reported may come from a later packet than the
  one the Java reader reported.
- Unreadable files raise ``FormatError`` (a ``ValueError``), where the Java reader's process
  failed and ``subprocess.CalledProcessError`` was raised.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, ContextManager, Dict, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]
Columns = Dict[str, Array]
ReadInfo = Dict[str, Union[int, float]]
ReadResult = Tuple[Columns, ReadInfo]

_INT64_MAX = 2**63 - 1
_TWO_TO_63 = 2.0**63
_MAX_MILLIS = _INT64_MAX // 1_000_000
_DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
_DAYS_BEFORE_MONTH = np.concatenate(([0], np.cumsum(_DAYS_IN_MONTH)[:-1]))

# re.ASCII: Java's Double.parseDouble accepts ASCII digits only, unlike float() and str-pattern \d
_DECIMAL = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFdD]?', re.ASCII)
_HEXADECIMAL = re.compile(r'[+-]?0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?\d+[fFdD]?', re.ASCII)
_SPECIAL = re.compile(r'[+-]?(?:NaN|Infinity)')
_INTEGER = re.compile(r'[+-]?[0-9]+')

_LOGGER = logging.getLogger('actipy')


class FormatError(ValueError):
    """A device file is malformed or unsupported."""


def warn(message: str) -> None:
    """Report a recoverable problem (formerly printed to stderr by the Java readers).

    Logged rather than raised with ``warnings.warn``: a warnings filter set to "error" would turn
    the message into an exception and discard the samples read so far, whereas the reader is
    meant to recover. Problems with the recorded data (skipped blocks, packets or pages,
    truncation, a page count that differs from the header) are also counted in ``ReadErrors``;
    metadata fallbacks, such as ignoring an invalid acceleration scale, are not. Without logging
    configuration, the message goes to stderr, as the Java readers' did.
    """

    _LOGGER.warning(message)


def java_arithmetic() -> ContextManager[Any]:
    """Floating-point errors as in Java: IEEE results (inf, nan) without warnings or exceptions.

    NumPy warns, or raises under ``np.errstate(all='raise')``, on overflow and invalid operations,
    including in values that are never used (e.g. calibration table entries for codes absent from
    a file). The readers run inside this context so that the caller's settings cannot abort a read
    that Java completed.
    """

    return np.errstate(all='ignore')


def java_long(values: Array) -> NDArray[np.int64]:
    """``(long) value`` for doubles: truncate toward zero, saturate at the int64 limits, NaN to 0."""

    values = np.asarray(values, dtype=np.float64)
    if values.size and -_TWO_TO_63 < values.min() and values.max() < _TWO_TO_63:  # False if any NaN
        return values.astype(np.int64)  # the usual case: every value in range, a plain truncation
    in_range = np.abs(values) < _TWO_TO_63  # False for NaN and +-Infinity
    result = np.where(in_range, values, 0.0).astype(np.int64)
    result[values >= _TWO_TO_63] = _INT64_MAX
    result[values <= -_TWO_TO_63] = -_INT64_MAX - 1
    return cast(NDArray[np.int64], result)


def java_trim(text: str) -> str:
    """``String.trim``: strip characters up to U+0020 from both ends."""

    start, stop = 0, len(text)
    while start < stop and text[start] <= ' ':
        start += 1
    while stop > start and text[stop - 1] <= ' ':
        stop -= 1
    return text[start:stop]


def parse_double(text: str) -> float:
    """``Double.parseDouble``, which accepts a different syntax than ``float`` (ASCII digits only)."""

    text = java_trim(text)
    if _DECIMAL.fullmatch(text):
        return float(text.rstrip('fFdD'))
    if _SPECIAL.fullmatch(text):
        return float(text.replace('Infinity', 'inf'))
    if _HEXADECIMAL.fullmatch(text):
        body = text.rstrip('fFdD')
        sign = -1.0 if body.startswith('-') else 1.0
        try:
            return sign * float.fromhex(body.lstrip('+-'))
        except OverflowError:  # Java returns +-Infinity, as float() already does for decimals
            return sign * math.inf
    raise ValueError(f"Invalid number: {text!r}")


def _parse_integer(text: str, bits: int) -> int:
    if not _INTEGER.fullmatch(text):
        raise ValueError(f"Invalid integer: {text!r}")
    value = int(text)
    if not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
        raise ValueError(f"Integer out of range: {text!r}")
    return value


def parse_int(text: str) -> int:
    """``Integer.parseInt`` for ASCII digits (Java also accepts other Unicode decimal digits)."""

    return _parse_integer(text, 32)


def parse_long(text: str) -> int:
    """``Long.parseLong`` for ASCII digits (Java also accepts other Unicode decimal digits)."""

    return _parse_integer(text, 64)


def millis_to_nanos(millis: NDArray[np.int64]) -> NDArray[np.int64]:
    """``TimeUnit.MILLISECONDS.toNanos`` in place, saturating like Java."""

    if millis.size and (millis.max() > _MAX_MILLIS or millis.min() < -_MAX_MILLIS):
        high = millis > _MAX_MILLIS
        low = millis < -_MAX_MILLIS
        millis *= 1_000_000
        millis[high] = _INT64_MAX
        millis[low] = -_INT64_MAX - 1
    else:
        millis *= 1_000_000
    return millis


def epoch_seconds(
    year: NDArray[np.int64],
    month: NDArray[np.int64],
    day: NDArray[np.int64],
    hour: NDArray[np.int64],
    minute: NDArray[np.int64],
    second: NDArray[np.int64],
) -> Tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """UTC epoch seconds and validity, as ``LocalDateTime.of(...)`` checks them.

    Valid for any non-negative proleptic Gregorian year.
    """

    month_index = np.clip(month - 1, 0, 11)
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    valid = (
        (month >= 1) & (month <= 12) & (day >= 1)
        & (day <= _DAYS_IN_MONTH[month_index] + ((month == 2) & leap))
        & (hour >= 0) & (hour < 24) & (minute >= 0) & (minute < 60)
        & (second >= 0) & (second < 60)
    )
    previous = year - 1
    days = (
        365 * (year - 1970)
        + (previous // 4 - 492) - (previous // 100 - 19) + (previous // 400 - 4)
        + _DAYS_BEFORE_MONTH[month_index] + ((month > 2) & leap) + day - 1
    )
    seconds = days * 86400 + hour * 3600 + minute * 60 + second
    return cast(NDArray[np.int64], seconds), cast(NDArray[np.bool_], valid)


def big_endian_words(buffer: Any, offset: int, stride: int, count: int) -> NDArray[np.uint16]:
    """Big-endian 16-bit words at ``offset + k * stride`` without copying."""

    if count == 0:
        return np.empty(0, dtype='>u2')
    return np.ndarray(
        (count,), dtype='>u2', buffer=buffer, offset=offset, strides=(stride,)
    )


def high_12_bits(buffer: Any, offset: int, stride: int, count: int) -> Array:
    """``(b[offset] << 4) | (b[offset + 1] >> 4)`` for each record."""

    return cast(Array, big_endian_words(buffer, offset, stride, count) >> 4)


def low_12_bits(buffer: Any, offset: int, stride: int, count: int) -> Array:
    """``((b[offset] & 0xF) << 8) | b[offset + 1]`` for each record."""

    return cast(Array, big_endian_words(buffer, offset, stride, count) & 0xFFF)


def signed_12_bit_table() -> NDArray[np.float64]:
    """Signed value of every 12-bit two's-complement code."""

    codes = np.arange(4096, dtype=np.int64)
    return cast(NDArray[np.float64], np.where(codes >= 2048, codes - 4096, codes).astype(np.float64))


def read_info(read_errors: int, sample_rate: float) -> ReadInfo:
    """The metadata the Java readers wrote to ``info.txt``."""

    return {'ReadOK': 1, 'ReadErrors': read_errors, 'SampleRate': float(sample_rate)}
