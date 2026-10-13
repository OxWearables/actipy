"""
GENEActiv (.bin) reader.

Decodes the text pages of a GENEActiv file with vectorized NumPy operations
and returns the values of the former Java reader (``GENEActivReader.java``).
The tests compare the results with the Java reader's on real-device fixtures
and edge cases.

The file is streamed in windows of whole lines. Pages in the canonical layout
are parsed in bulk; any other page goes through a line-by-line port of the
Java page parser.
"""

from __future__ import annotations

import binascii
import os
from typing import List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from actipy._reader_support import (
    Array,
    Columns,
    FormatError,
    ReadResult,
    epoch_seconds,
    high_12_bits,
    java_arithmetic,
    java_long,
    java_trim,
    low_12_bits,
    millis_to_nanos,
    parse_double,
    parse_int,
    read_info,
    signed_12_bit_table,
    warn,
)

__all__ = ['read']

HEADER_LINES = 59
PAGE_LINES = 10
SAMPLES_PER_PAGE = 300
PAYLOAD_LENGTH = SAMPLES_PER_PAGE * 12
WINDOW_BYTES = 1 << 26
PAGES_PER_BATCH = 4096

_HEX_DIGITS = np.full(256, 255, dtype=np.uint8)
for _value, _digit in enumerate(b'0123456789ABCDEF'):
    _HEX_DIGITS[_digit] = _value
for _value, _digit in enumerate(b'abcdef'):
    _HEX_DIGITS[_digit] = 10 + _value

_HEADER_FIELDS = {47: 'x gain', 48: 'x offset', 49: 'y gain', 50: 'y offset', 51: 'z gain',
                  52: 'z offset', 53: 'voltage', 54: 'illuminance', 55: 'header separator',
                  56: 'memory status header', 57: 'page count'}


def read(path: str) -> ReadResult:
    """Read an uncompressed GENEActiv .bin file into columns and reader metadata."""

    with java_arithmetic():
        return _Reader(os.path.getsize(path)).read(path)


class _Reader:
    def __init__(self, file_size: int) -> None:
        capacity = (file_size // (PAYLOAD_LENGTH + PAGE_LINES) + 1) * SAMPLES_PER_PAGE
        self.columns: Columns = {'time': np.empty(capacity, dtype='datetime64[ns]')}
        self.columns.update(
            (name, np.empty(capacity, dtype=np.float32)) for name in ('x', 'y', 'z', 'temperature')
        )
        self.rows = 0
        self.pages = 0
        self.valid_pages = 0
        self.read_errors = 0
        self.sample_rate = -1.0
        self.lookup: List[NDArray[np.float32]] = []

    def read(self, path: str) -> ReadResult:
        expected_pages: Optional[int] = None
        data = b''
        with open(path, 'rb') as stream:
            while True:
                chunk = stream.read(WINDOW_BYTES)
                final = not chunk
                data += chunk
                buffer = np.frombuffer(data, dtype=np.uint8)
                starts, ends, consumed = _line_bounds(buffer, final)
                first = 0
                if expected_pages is None:
                    if len(starts) < HEADER_LINES:
                        if final:
                            missing = _HEADER_FIELDS.get(len(starts), 'GENEActiv file header')
                            raise FormatError(f"Unexpected end of {missing}")
                        continue
                    expected_pages = self._read_calibration(buffer, starts, ends)
                    first = HEADER_LINES
                whole = (len(starts) - first) // PAGE_LINES * PAGE_LINES
                for batch in range(first, first + whole, PAGES_PER_BATCH * PAGE_LINES):
                    stop = min(batch + PAGES_PER_BATCH * PAGE_LINES, first + whole)
                    self._decode_pages(
                        buffer,
                        starts[batch:stop].reshape(-1, PAGE_LINES),
                        ends[batch:stop].reshape(-1, PAGE_LINES),
                    )
                leftover = first + whole
                if final:
                    return self._finish(expected_pages, truncated=leftover < len(starts))
                data = data[int(starts[leftover]) if leftover < len(starts) else consumed:]

    def _read_calibration(self, buffer: NDArray[np.uint8], starts: Array, ends: Array) -> int:
        lines = [_text(buffer, starts[i], ends[i]) for i in range(HEADER_LINES)]
        gains = [_parse_double_field(lines[i], key) for i, key in ((47, 'x gain'), (49, 'y gain'), (51, 'z gain'))]
        offsets = [_parse_int_field(lines[i], key) for i, key in ((48, 'x offset'), (50, 'y offset'), (52, 'z offset'))]
        _parse_int_field(lines[53], 'Volts')
        _parse_int_field(lines[54], 'Lux')
        expected_pages = _parse_int_field(lines[57], 'Number of Pages')
        for axis, gain in enumerate(gains):
            if not np.isfinite(gain) or gain == 0:
                raise FormatError(f"Calibration gain must be finite and non-zero for axis {axis}")
        if expected_pages < 0:
            raise FormatError("Page count must not be negative")
        # (raw * 100.0 - offset) / gain for every 12-bit code, cast like Java's (float)
        signed = signed_12_bit_table()
        self.lookup = [((signed * 100.0 - offsets[a]) / gains[a]).astype(np.float32) for a in range(3)]
        return expected_pages

    def _decode_pages(self, buffer: NDArray[np.uint8], starts: Array, ends: Array) -> None:
        count = len(starts)
        start_millis, temperature, sample_rate, header_ok = _parse_page_headers(buffer, starts, ends)
        for page in np.flatnonzero(~header_ok):        # exact Java port for other layouts
            lines = [_text(buffer, starts[page, j], ends[page, j]) for j in range(PAGE_LINES)]
            try:
                start_millis[page], temperature[page], sample_rate[page] = _parse_page_header(lines)
                header_ok[page] = True
            except FormatError as error:
                warn(f"Skipping malformed GENEActiv page {self.pages + page + 1}: {error}")
        self.pages += count
        self.read_errors += int((~header_ok).sum())

        good = header_ok & (ends[:, 9] - starts[:, 9] == PAYLOAD_LENGTH)
        selected = np.flatnonzero(good)
        hex_rows = np.lib.stride_tricks.sliding_window_view(buffer, PAYLOAD_LENGTH)[starts[selected, 9]] \
            if len(selected) else np.empty((0, PAYLOAD_LENGTH), dtype=np.uint8)
        try:
            raw = binascii.a2b_hex(hex_rows.tobytes())
        except binascii.Error:                          # locate the pages that are not pure hex
            pure = (_HEX_DIGITS[hex_rows] != 255).all(axis=1)
            good[selected[~pure]] = False
            selected, hex_rows = selected[pure], hex_rows[pure]
            raw = binascii.a2b_hex(hex_rows.tobytes())
        invalid_payloads = int((header_ok & ~good).sum())
        if invalid_payloads:
            self.read_errors += invalid_payloads
            warn(f"Skipping {invalid_payloads} invalid GENEActiv page payload(s): "
                 f"expected {PAYLOAD_LENGTH} hexadecimal characters")
        if len(selected) == 0:
            return

        # Each sample is 6 bytes: 12-bit x, y, z and an auxiliary field.
        samples = len(selected) * SAMPLES_PER_PAGE
        first, stop = self.rows, self.rows + samples
        self.columns['x'][first:stop] = self.lookup[0][high_12_bits(raw, 0, 6, samples)]
        self.columns['y'][first:stop] = self.lookup[1][low_12_bits(raw, 1, 6, samples)]
        self.columns['z'][first:stop] = self.lookup[2][high_12_bits(raw, 3, 6, samples)]
        # (long) (pageStart + sampleIndex * 1000d / sampleRate)
        offsets = np.arange(SAMPLES_PER_PAGE) * 1000.0
        sample_millis = offsets[None, :] / sample_rate[selected, None] + start_millis[selected, None].astype(np.float64)
        times = self.columns['time'][first:stop].view(np.int64).reshape(-1, SAMPLES_PER_PAGE)
        times[...] = java_long(sample_millis)
        millis_to_nanos(times)
        self.columns['temperature'][first:stop] = np.repeat(
            temperature[selected].astype(np.float32), SAMPLES_PER_PAGE
        )
        self.rows = stop
        self.valid_pages += len(selected)
        self.sample_rate = float(sample_rate[selected[-1]])

    def _finish(self, expected_pages: Optional[int], truncated: bool) -> ReadResult:
        if truncated:
            if self.rows == 0:
                raise FormatError("Unexpected end of GENEActiv page")
            self.read_errors += 1
            warn("Stopping at truncated GENEActiv data: Unexpected end of GENEActiv page")
        if (self.pages > 0 or (expected_pages or 0) > 0) and self.valid_pages == 0:
            raise FormatError("No valid GENEActiv pages were decoded")
        if self.pages != expected_pages and not truncated:
            self.read_errors += 1
            warn(f"GENEActiv page count differs from header: expected {expected_pages} but found {self.pages}")
        for values in self.columns.values():
            values.resize(self.rows, refcheck=False)
        return self.columns, read_info(self.read_errors, self.sample_rate)


def _line_bounds(buffer: NDArray[np.uint8], final: bool) -> Tuple[Array, Array, int]:
    """Line [start, end) offsets using ``BufferedReader.readLine`` terminators.

    Unless ``final``, only complete lines are returned; ``consumed`` is where
    the first incomplete line starts. A trailing carriage return is held back
    because it may be the first half of a CRLF in the next window.
    """

    size = buffer.size
    newlines = np.flatnonzero(buffer == 10)
    carriage_returns = int(np.count_nonzero(buffer == 13))
    if carriage_returns == 0:
        ends, nexts = newlines, newlines + 1
    elif (carriage_returns == len(newlines) and len(newlines) and newlines[0] > 0
          and (buffer[newlines - 1] == 13).all()):
        ends, nexts = newlines - 1, newlines + 1
    else:  # any mix of \n, \r and \r\n
        is_return = buffer == 13
        after_return = np.zeros(size, dtype=bool)
        after_return[1:] = is_return[:-1]
        ends = np.flatnonzero(is_return | ((buffer == 10) & ~after_return))
        if not final and len(ends) and ends[-1] == size - 1 and buffer[-1] == 13:
            ends = ends[:-1]
        crlf = (buffer[ends] == 13) & (ends + 1 < size) & (buffer[np.minimum(ends + 1, size - 1)] == 10)
        nexts = ends + 1 + crlf
    starts = np.concatenate(([0], nexts[:-1])).astype(np.int64) if len(nexts) else np.zeros(0, dtype=np.int64)
    consumed = int(nexts[-1]) if len(nexts) else 0
    if final and consumed < size:  # last line without a terminator
        starts = np.append(starts, consumed)
        ends = np.append(ends, size)
        consumed = size
    return starts, ends.astype(np.int64), consumed


def _text(buffer: NDArray[np.uint8], start: int, end: int) -> str:
    return bytes(buffer[int(start):int(end)]).decode('ascii', errors='replace')


def _gather(buffer: NDArray[np.uint8], starts: Array, width: int) -> NDArray[np.uint8]:
    """``width`` bytes from each start (clipped at the end of the buffer)."""

    return cast(NDArray[np.uint8], buffer[np.minimum(starts[:, None] + np.arange(width), buffer.size - 1)])


def _matches(buffer: NDArray[np.uint8], starts: Array, text: bytes) -> NDArray[np.bool_]:
    return cast(NDArray[np.bool_], (_gather(buffer, starts, len(text)) == np.frombuffer(text, dtype=np.uint8)).all(axis=1))


def _parse_numbers(buffer: NDArray[np.uint8], starts: Array, ends: Array) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """``Double.parseDouble`` of each value, parsing each distinct string once."""

    width = 24
    lengths = ends - starts
    values = _gather(buffer, starts, width)
    values[np.arange(width)[None, :] >= lengths[:, None]] = 0
    keys = np.ascontiguousarray(values).view(f'S{width}')[:, 0]
    unique, inverse = np.unique(keys, return_inverse=True)
    parsed = np.empty(len(unique))
    ok = np.ones(len(unique), dtype=bool)
    for k, key in enumerate(unique):
        try:
            parsed[k] = parse_double(key.decode('ascii', errors='replace'))
        except ValueError:
            ok[k] = False
    inverse = inverse.reshape(-1)
    return parsed[inverse], (lengths <= width) & ok[inverse]


def _parse_page_headers(
    buffer: NDArray[np.uint8], starts: Array, ends: Array
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Parse pages in the canonical layout in bulk; flag the others for the exact port."""

    ok = (ends[:, 0] - starts[:, 0] == 13) & _matches(buffer, starts[:, 0], b'Recorded Data')
    # Page Time:YYYY-MM-DD hh:mm:ss:mmm
    ok &= (ends[:, 3] - starts[:, 3] == 33) & _matches(buffer, starts[:, 3], b'Page Time:')
    stamp = _gather(buffer, starts[:, 3] + 10, 23).astype(np.int64)
    separators = [4, 7, 10, 13, 16, 19]
    ok &= (stamp[:, separators] == np.frombuffer(b'-- :::', dtype=np.uint8)).all(axis=1)
    digits = np.delete(stamp, separators, axis=1) - ord('0')
    ok &= ((digits >= 0) & (digits <= 9)).all(axis=1)
    digits = np.clip(digits, 0, 9)

    def number(first: int, stop: int) -> NDArray[np.int64]:
        return cast(NDArray[np.int64], (digits[:, first:stop] * 10 ** np.arange(stop - first - 1, -1, -1)).sum(axis=1))

    seconds, valid_date = epoch_seconds(number(0, 4), number(4, 6), number(6, 8),
                                        number(8, 10), number(10, 12), number(12, 14))
    ok &= valid_date
    start_millis = seconds * 1000 + number(14, 17)
    ok &= _matches(buffer, starts[:, 5], b'Temperature:')
    ok &= _matches(buffer, starts[:, 8], b'Measurement Frequency:')
    temperature, ok_temperature = _parse_numbers(buffer, starts[:, 5] + 12, ends[:, 5])
    sample_rate, ok_rate = _parse_numbers(buffer, starts[:, 8] + 22, ends[:, 8])
    ok &= ok_temperature & ok_rate & np.isfinite(temperature) & np.isfinite(sample_rate) & (sample_rate > 0)
    return start_millis, temperature, sample_rate, ok


def _value_after_colon(line: str, key: str) -> str:
    separator = line.find(':')
    if separator < 0 or java_trim(line[:separator]) != key:
        raise FormatError(f"Expected {key} header field")
    return java_trim(line[separator + 1:])


def _parse_double_field(line: str, key: str) -> float:
    try:
        return parse_double(_value_after_colon(line, key))
    except ValueError as error:
        if isinstance(error, FormatError):
            raise
        raise FormatError(f"Invalid numeric value for {key}") from error


def _parse_int_field(line: str, key: str) -> int:
    try:
        return parse_int(_value_after_colon(line, key))
    except ValueError as error:
        if isinstance(error, FormatError):
            raise
        raise FormatError(f"Invalid numeric value for {key}") from error


def _page_time_millis(stamp: str) -> int:
    """``parsePageTime``: a strict ``yyyy-MM-dd HH:mm:ss:SSS`` timestamp."""

    if (len(stamp) != 23 or stamp[4] != '-' or stamp[7] != '-' or stamp[10] != ' '
            or stamp[13] != ':' or stamp[16] != ':' or stamp[19] != ':'):
        raise FormatError("Invalid page timestamp or numeric metadata")
    fields = (stamp[0:4], stamp[5:7], stamp[8:10], stamp[11:13], stamp[14:16], stamp[17:19], stamp[20:23])
    if not all('0' <= character <= '9' for field in fields for character in field):
        raise FormatError("Invalid page timestamp or numeric metadata")
    year, month, day, hour, minute, second, millis = (int(field) for field in fields)
    seconds, valid = epoch_seconds(*(np.array([value], dtype=np.int64) for value in (year, month, day, hour, minute, second)))
    if not valid[0]:
        raise FormatError("Invalid page timestamp or numeric metadata")
    return int(seconds[0]) * 1000 + millis


def _parse_page_header(lines: List[str]) -> Tuple[int, float, float]:
    """``parsePageHeader`` for one page: start time (ms), temperature and rate."""

    if lines[0] != 'Recorded Data':
        raise FormatError("Missing Recorded Data marker")
    start_millis = _page_time_millis(_value_after_colon(lines[3], 'Page Time'))
    try:
        temperature = parse_double(_value_after_colon(lines[5], 'Temperature'))
        sample_rate = parse_double(_value_after_colon(lines[8], 'Measurement Frequency'))
    except FormatError:
        raise
    except ValueError as error:
        raise FormatError("Invalid page timestamp or numeric metadata") from error
    if not np.isfinite(temperature):
        raise FormatError("Temperature must be finite")
    if not (np.isfinite(sample_rate) and sample_rate > 0):
        raise FormatError("Measurement Frequency must be positive and finite")
    return start_millis, temperature, sample_rate
