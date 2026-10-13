"""
ActiGraph (.gt3x) reader.

Decodes V1 (``activity.bin``) and V2 (``log.bin``) archives with vectorized
NumPy operations and returns the values of the former Java reader
(``ActigraphReader.java``), using Java's ``Math.round`` rounding and its
saturating conversion to ``long``. The tests compare the results with the
Java reader's on real-device fixtures and edge cases.
"""

from __future__ import annotations

import math
import re
import struct
import zipfile
import zlib
from typing import IO, Dict, List, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from actipy._reader_support import (
    Array,
    Columns,
    FormatError,
    ReadResult,
    high_12_bits,
    java_arithmetic,
    java_long,
    java_trim,
    low_12_bits,
    millis_to_nanos,
    parse_double,
    parse_long,
    read_info,
    signed_12_bit_table,
    warn,
)

__all__ = ['read']

SYNC_BYTE = 0x1E
PARAMETER_RECORD = 21
ACTIVITY_RECORD = 0
ACTIVITY2_RECORD = 26
READ_BYTES = 9 << 20  # decompressed bytes per read (~2M packed samples)
_PACKET_HEADER = struct.Struct('<BxxxxxH')
_SIGNED_16 = np.arange(65536, dtype=np.int64).astype(np.uint16).view(np.int16).astype(np.float64)


class _Metadata:
    def __init__(self) -> None:
        self.sample_rate = -1.0
        self.scale = -1.0
        self.scale_min = math.nan
        self.scale_max = math.nan
        self.scale_present = False
        self.first_millis = -1
        self.serial = ''


def read(path: str) -> ReadResult:
    """Read a .gt3x archive into columns and reader metadata."""

    try:
        with java_arithmetic(), zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if 'info.txt' in names and 'activity.bin' in names and 'lux.bin' in names:
                version, member = 1, 'activity.bin'
            elif 'info.txt' in names and 'log.bin' in names:
                version, member = 2, 'log.bin'
            else:
                raise FormatError("File is not a supported V1 or V2 GT3X archive")
            metadata = _read_metadata(archive.read('info.txt').decode('utf-8', errors='replace'), path)
            _validate(metadata, version)
            capacity = 2 * (archive.getinfo(member).file_size // 9) + 1
            with archive.open(member) as stream:
                if version == 1:
                    columns, errors = _read_v1(stream, metadata, capacity)
                else:
                    columns, errors = _read_v2(stream, metadata, capacity)
    except (zipfile.BadZipFile, zlib.error, EOFError) as error:
        raise FormatError(f"Invalid GT3X archive: {error}") from error
    return columns, read_info(errors, metadata.sample_rate)


def _read_metadata(text: str, path: str) -> _Metadata:  # noqa: C901
    metadata = _Metadata()
    try:
        for line in re.split(r'\r\n|\r|\n', text):
            tokens = line.split(': ', 1)
            if len(tokens) != 2:
                continue
            key = java_trim(tokens[0])
            if key.startswith('﻿'):
                key = java_trim(key[1:])
            value = java_trim(tokens[1])
            if key == 'Sample Rate':
                metadata.sample_rate = parse_double(value)
            elif key == 'Start Date':
                metadata.first_millis = _ticks_to_millis(parse_long(value))
            elif key == 'Acceleration Scale':
                metadata.scale = parse_double(value)
                metadata.scale_present = True
            elif key == 'Acceleration Min':
                metadata.scale_min = parse_double(value)
            elif key == 'Acceleration Max':
                metadata.scale_max = parse_double(value)
            elif key == 'Serial Number':
                metadata.serial = value
    except ValueError as error:
        raise FormatError(str(error)) from error
    if not _valid_scale(metadata.scale) or not _scale_matches_range(metadata):
        if metadata.scale_present:
            warn(f"Ignoring invalid Acceleration Scale in {path}")
        serial = metadata.serial
        metadata.scale = 341.0 if serial.startswith(('NEO', 'CLE')) else 256.0 if serial.startswith('MOS') else -1.0
    return metadata


def _validate(metadata: _Metadata, version: int) -> None:
    if not (math.isfinite(metadata.sample_rate) and metadata.sample_rate > 0):
        raise FormatError("GT3X metadata must contain a positive finite Sample Rate")
    if version == 1:
        if metadata.first_millis < 0:
            raise FormatError("V1 GT3X metadata must contain Start Date")
        if not _valid_scale(metadata.scale):
            raise FormatError("V1 GT3X metadata must contain a valid acceleration scale")


def _ticks_to_millis(ticks: int) -> int:
    """(ticks - 621355968000000000L) / 10000 with Java's long overflow and truncation."""

    difference = ((ticks - 621355968000000000 + 2**63) % 2**64) - 2**63
    millis = abs(difference) // 10000
    return millis if difference >= 0 else -millis


def _valid_scale(scale: float) -> bool:
    return math.isfinite(scale) and 16.0 <= scale <= 32768.0


def _scale_matches_range(metadata: _Metadata) -> bool:
    low, high = metadata.scale_min, metadata.scale_max
    if math.isnan(low) or math.isnan(high):
        return True
    if not math.isfinite(low) or not math.isfinite(high) or low >= 0 or high <= 0:
        return False
    full_scale = metadata.scale * max(abs(low), abs(high))
    return 1024.0 <= full_scale <= 32768.0


def _decode_parameter(value: int) -> float:
    if value == 0x007FFFFF:
        return float(2**31 - 1)
    if value == 0x00800000:
        return -float(2**31 - 1)
    exponent = value >> 24
    if exponent & 0x80:
        exponent -= 256
    significand = value & 0xFFFFFF
    if significand & 0x800000:
        significand -= 1 << 24
    return math.ldexp(significand, exponent - 23)


def _java_round(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``Math.round`` before its conversion to ``long``: exact ``floor(x + 0.5)``.

    NaN and infinities pass through; ``java_long`` then gives Java's ``long`` result
    (NaN to 0, saturating at the int64 limits).
    """

    rounded = np.floor(values)
    rounded += (values - rounded) >= 0.5
    return cast(NDArray[np.float64], rounded)


def _round_thousandth(values: NDArray[np.float64]) -> NDArray[np.float32]:
    """``(float) (Math.round(x * 1000d) / 1000d)``."""

    return (_java_round(values * 1000.0) / 1000.0).astype(np.float32)


def _packed_fields(rows: Union[bytes, Array], count: int) -> Tuple[Tuple[Array, Array, Array], Tuple[Array, Array, Array]]:
    """x, y, z codes of the two samples packed in each 9-byte record (y1 x1 z1 y2 x2 z2)."""

    first = (low_12_bits(rows, 1, 9, count), high_12_bits(rows, 0, 9, count), high_12_bits(rows, 3, 9, count))
    second = (high_12_bits(rows, 6, 9, count), low_12_bits(rows, 4, 9, count), low_12_bits(rows, 7, 9, count))
    return first, second


class _Output:
    """Output columns with a capacity that only grows if the archive size was wrong."""

    def __init__(self, capacity: int) -> None:
        self.time = np.empty(capacity, dtype=np.int64)
        self.xyz = [np.empty(capacity, dtype=np.float32) for _ in range(3)]
        self.rows = 0

    def reserve(self, count: int) -> None:
        if self.rows + count > len(self.time):
            capacity = max(self.rows + count, 2 * len(self.time))
            for values in [self.time] + self.xyz:
                values.resize(capacity, refcheck=False)

    def columns(self) -> Columns:
        for values in [self.time] + self.xyz:
            values.resize(self.rows, refcheck=False)
        return {'time': self.time.view('datetime64[ns]'), 'x': self.xyz[0], 'y': self.xyz[1], 'z': self.xyz[2]}


def _read_v1(stream: IO[bytes], metadata: _Metadata, capacity: int) -> Tuple[Columns, int]:
    lookup = (signed_12_bit_table() / metadata.scale).astype(np.float32)
    output = _Output(capacity)
    data = b''
    while True:
        chunk = stream.read(READ_BYTES)
        if not chunk:
            break
        data += chunk
        whole = len(data) - len(data) % 9
        _write_v1(output, data[:whole], whole // 9, lookup, metadata)
        data = data[whole:]
    errors = 0
    if len(data) == 5:  # a final unpaired sample
        _write_v1(output, data + bytes(4), 1, lookup, metadata, single=True)
    elif data:
        if output.rows == 0:
            raise FormatError("Unexpected end of V1 packed activity")
        errors = 1
        warn("Stopping at truncated GT3X data: Unexpected end of V1 packed activity")
    return output.columns(), errors


def _write_v1(
    output: _Output, data: bytes, pairs: int, lookup: NDArray[np.float32],
    metadata: _Metadata, single: bool = False,
) -> None:
    count = 1 if single else 2 * pairs
    if count == 0:
        return
    output.reserve(count)
    first = output.rows
    # firstSampleTime + Math.round(1000d * sampleIndex / sampleRate)
    offsets = np.arange(first, first + count, dtype=np.float64)
    offsets *= 1000.0
    offsets /= metadata.sample_rate
    times = output.time[first:first + count]
    times[...] = java_long(_java_round(offsets))
    times += metadata.first_millis
    millis_to_nanos(times)
    samples = _packed_fields(data, pairs)
    for half, fields in enumerate(samples[:1] if single else samples):
        for axis, codes in enumerate(fields):
            output.xyz[axis][first + half:first + count:2] = lookup[codes[:count]]
    output.rows += count


def _read_v2(stream: IO[bytes], metadata: _Metadata, capacity: int) -> Tuple[Columns, int]:
    state = _V2State(metadata)
    output = _Output(capacity)
    data = b''
    while True:
        chunk = stream.read(READ_BYTES)
        data += chunk
        starts, position = _scan_packets(data)
        if starts:
            _write_v2(output, data, np.array(starts, dtype=np.int64), state)
        data = data[position:]
        if not chunk:
            break
    errors = 0
    if data:  # a truncated final packet
        if len(data) < 8:
            message = "Unexpected end of V2 packet header"
        else:
            payload_size = _PACKET_HEADER.unpack_from(data, 0)[1]
            if len(data) < 8 + payload_size:
                message = "Unexpected end of V2 packet payload"
            else:
                message = "Unexpected end of V2 packet checksum"
        if not state.wrote_samples:
            raise FormatError(message)
        errors = 1
        warn(f"Stopping at truncated GT3X data: {message}")
    return output.columns(), errors


class _V2State:
    def __init__(self, metadata: _Metadata) -> None:
        self.sample_rate = metadata.sample_rate
        self.scale = metadata.scale
        self.wrote_samples = False
        self.lookups: Dict[Tuple[int, float], NDArray[np.float32]] = {}

    def lookup(self, record_type: int, scale: float) -> NDArray[np.float32]:
        key = (record_type, scale)
        if key not in self.lookups:
            codes = _SIGNED_16 if record_type == ACTIVITY2_RECORD else signed_12_bit_table()
            self.lookups[key] = _round_thousandth(codes / scale)
        return self.lookups[key]


def _scan_packets(data: bytes) -> Tuple[List[int], int]:
    """Offsets of the complete packets; stops at the first incomplete one."""

    starts: List[int] = []
    append = starts.append
    unpack = _PACKET_HEADER.unpack_from
    position, end = 0, len(data)
    while end - position >= 8:
        sync, size = unpack(data, position)
        if sync != SYNC_BYTE:
            raise FormatError("Invalid GT3X packet synchronization byte")
        following = position + 9 + size
        if following > end:
            break
        append(position)
        position = following
    return starts, position


def _write_v2(output: _Output, data: bytes, starts: NDArray[np.int64], state: _V2State) -> None:  # noqa: C901
    buffer = np.frombuffer(data, dtype=np.uint8)
    headers = np.lib.stride_tricks.sliding_window_view(buffer, 8)[starts]
    record_type = headers[:, 1].astype(np.int64)
    timestamp = np.ascontiguousarray(headers[:, 2:6]).view('<u4')[:, 0].astype(np.int64)
    size = np.ascontiguousarray(headers[:, 6:8]).view('<u2')[:, 0].astype(np.int64)
    checksum_at = starts + 8 + size
    bounds = np.empty(2 * len(starts), dtype=np.int64)
    bounds[0::2], bounds[1::2] = starts, checksum_at
    xor = np.bitwise_xor.reduceat(buffer[:checksum_at[-1] + 1], bounds)[0::2]
    if ((~xor) != buffer[checksum_at]).any():
        raise FormatError("GT3X packet checksum mismatch")

    # The acceleration scale is updated by parameter records, in stream order.
    scale = np.empty(len(starts))
    current, previous = state.scale, 0
    for packet in np.flatnonzero(record_type == PARAMETER_RECORD):
        scale[previous:packet] = current
        payload_size = int(size[packet])
        if payload_size % 8:
            raise FormatError("GT3X parameter payload length is not a multiple of eight")
        payload = data[starts[packet] + 8:starts[packet] + 8 + payload_size]
        for offset in range(0, payload_size, 8):
            if payload[offset] == 0 and payload[offset + 2] == 55:
                decoded = _decode_parameter(struct.unpack_from('<I', payload, offset + 4)[0])
                if _valid_scale(decoded):
                    current = decoded
        previous = packet
    scale[previous:] = current
    state.scale = current

    activity = np.flatnonzero(
        ((record_type == ACTIVITY_RECORD) | (record_type == ACTIVITY2_RECORD)) & (size > 1)
    )
    if len(activity) == 0:
        return
    if not all(_valid_scale(value) for value in np.unique(scale[activity])):
        raise FormatError("No valid acceleration scale found before activity data")
    kinds, sizes = record_type[activity], size[activity]
    if ((kinds == ACTIVITY2_RECORD) & (sizes % 6 != 0)).any():
        raise FormatError("GT3X activity payload length is not a multiple of six")
    if ((kinds == ACTIVITY_RECORD) & ~np.isin(sizes % 9, (0, 5))).any():
        raise FormatError("Packed GT3X activity payload has an incomplete sample")

    counts = np.where(kinds == ACTIVITY2_RECORD, sizes // 6, (sizes // 9) * 2 + (sizes % 9 == 5))
    total = int(counts.sum())
    output.reserve(total)
    first_row = output.rows + np.concatenate(([0], np.cumsum(counts)[:-1]))
    longest = int(counts.max())
    # timestamp * 1000 + Math.round(1000d * sampleIndex / sampleRate)
    offset_millis = java_long(_java_round(1000.0 * np.arange(longest, dtype=np.float64) / state.sample_rate))

    # Packets with the same type, size and scale decode together as one 2-D block.
    keys = np.stack([kinds.astype(np.float64), sizes.astype(np.float64), scale[activity]], axis=1)
    if (keys == keys[0]).all():
        groups = [np.arange(len(activity))]
    else:
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        groups = [np.flatnonzero(inverse == group) for group in range(int(inverse.max()) + 1)]
    contiguous = len(groups) == 1
    for members in groups:
        packets = activity[members]
        kind, payload_size, count = int(kinds[members[0]]), int(sizes[members[0]]), int(counts[members[0]])
        lookup = state.lookup(kind, float(scale[packets[0]]))
        payloads = np.ascontiguousarray(
            np.lib.stride_tricks.sliding_window_view(buffer, payload_size)[starts[packets] + 8]
        )
        if kind == ACTIVITY2_RECORD:
            codes = payloads.view('<u2').reshape(len(packets), count, 3)
            xyz = [codes[:, :, axis] for axis in range(3)]
        else:
            whole = payload_size // 9
            records = np.ascontiguousarray(payloads[:, :whole * 9])
            fields = _packed_fields(records, len(packets) * whole)
            xyz = []
            for axis in range(3):
                pair = np.stack([fields[0][axis], fields[1][axis]], axis=1).reshape(len(packets), 2 * whole)
                if payload_size % 9 == 5:  # one unpaired sample at the end
                    tail = np.zeros((len(packets), 9), dtype=np.uint8)
                    tail[:, :5] = payloads[:, whole * 9:]
                    last = _packed_fields(tail, len(packets))[0][axis]
                    pair = np.concatenate([pair, last[:, None]], axis=1)
                xyz.append(pair)
        millis = timestamp[packets, None] * 1000 + offset_millis[None, :count]
        if contiguous:
            rows = slice(output.rows, output.rows + total)
            output.time[rows] = millis_to_nanos(millis).reshape(-1)
            for axis in range(3):
                output.xyz[axis][rows] = lookup[xyz[axis]].reshape(-1)
        else:
            rows_index = first_row[members][:, None] + np.arange(count)[None, :]
            output.time[rows_index] = millis_to_nanos(millis)
            for axis in range(3):
                output.xyz[axis][rows_index] = lookup[xyz[axis]]
    output.rows += total
    state.wrote_samples = state.wrote_samples or total > 0
