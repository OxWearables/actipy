"""
Axivity AX3/AX6 (.cwa) reader.

Decodes the 512-byte CWA blocks with vectorized NumPy operations and returns
the values of the former Java reader (``AxivityReader.java``): the float
arithmetic follows the same single/double precision steps. The tests compare
the results with the Java reader's on real-device fixtures and edge cases.

The file is read twice in fixed-size chunks: first for block headers and
checksums (small per-block arrays), then for the samples, which are written
straight into exactly sized output arrays.
"""

from __future__ import annotations

import os
from io import BufferedReader
from typing import Dict, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from actipy._reader_support import (
    Array,
    Columns,
    FormatError,
    ReadResult,
    epoch_seconds,
    java_arithmetic,
    java_long,
    millis_to_nanos,
    read_info,
    warn,
)

__all__ = ['read']

BLOCK_SIZE = 512
HEADER_SIZE = 30
PAYLOAD_SIZE = 480
CHUNK_BLOCKS = 16384  # 8 MB of input per chunk

# Java: (float) Math.pow(10, raw / 341.0)
_LIGHT = np.power(10.0, np.arange(1024) / 341.0).astype(np.float32)
# Java: (float) (((raw & 0x3FF) * 150.0 - 20500) / 1000)
_TEMPERATURE = ((np.arange(1024) * 150.0 - 20500) / 1000).astype(np.float32)


def read(path: str) -> ReadResult:
    """Read an uncompressed .cwa file into columns and reader metadata."""

    size = os.path.getsize(path)
    num_blocks, partial = divmod(size, BLOCK_SIZE)
    with java_arithmetic(), open(path, 'rb') as stream:
        headers, checksums = _read_headers(stream, num_blocks)
        blocks = _BlockTable.parse(headers, checksums)
        if blocks.total_samples == 0 and partial:
            raise FormatError("Unexpected partial CWA block")
        read_errors = blocks.read_errors
        if partial:
            read_errors += 1
            warn("Stopping at truncated CWA data: Unexpected partial CWA block")
        stream.seek(0)
        columns = _read_samples(stream, num_blocks, blocks)
    return columns, read_info(read_errors, blocks.last_sample_rate)


def _read_exact(stream: BufferedReader, buffer: NDArray[np.uint8]) -> None:
    """Fill ``buffer`` completely, reading as often as needed; raise if the file ends first."""

    view = memoryview(cast(bytearray, buffer)).cast('B')  # flat bytes: len() and slices count bytes, not rows
    filled = 0
    while filled < len(view):
        count = stream.readinto(view[filled:])
        if not count:
            raise FormatError("Unexpected end of CWA file")
        filled += count


def _read_headers(stream: BufferedReader, num_blocks: int) -> Tuple[NDArray[np.uint8], NDArray[np.uint16]]:
    headers = np.empty((num_blocks, HEADER_SIZE), dtype=np.uint8)
    checksums = np.empty(num_blocks, dtype=np.uint16)
    buffer = np.empty((CHUNK_BLOCKS, BLOCK_SIZE), dtype=np.uint8)
    for first in range(0, num_blocks, CHUNK_BLOCKS):
        stop = min(first + CHUNK_BLOCKS, num_blocks)
        chunk = buffer[:stop - first]
        _read_exact(stream, chunk)
        headers[first:stop] = chunk[:, :HEADER_SIZE]
        # Sum of all 256 little-endian words, modulo 2^16 (Java's short sum).
        checksums[first:stop] = chunk.view('<u2').sum(axis=1, dtype=np.uint16)
    return headers, checksums


class _BlockTable:
    """Per-block metadata of the valid data blocks, in file order."""

    def __init__(self) -> None:
        self.index: NDArray[np.intp] = np.empty(0, dtype=np.intp)
        self.has_gyroscope = False
        self.packing: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.num_axes: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.count: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.start: NDArray[np.float64] = np.empty(0)
        self.end: NDArray[np.float64] = np.empty(0)
        self.inverse_unit: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.gyroscope_unit: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.temperature: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.light: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self.offsets: NDArray[np.int64] = np.zeros(1, dtype=np.int64)
        self.read_errors = 0
        self.last_sample_rate = -1.0

    @property
    def total_samples(self) -> int:
        return int(self.offsets[-1])

    @classmethod
    def parse(cls, headers: NDArray[np.uint8], checksums: NDArray[np.uint16]) -> _BlockTable:
        """Validate block headers and derive sample timing (DataBlock.parse + BlockDecoder)."""

        table = cls()
        words = headers.view('<u2')
        is_data = (headers[:, 0] == ord('A')) & (headers[:, 1] == ord('X'))
        rate_code = headers[:, 24].astype(np.int64)
        num_axes = (headers[:, 25] >> 4).astype(np.int64)
        packing = (headers[:, 25] & 0x0F).astype(np.int64)
        bytes_per_sample = np.where(
            packing == 2, 2 * num_axes, np.where((packing == 0) & (num_axes == 3), 4, 0)
        )
        timestamp_offset = words[:, 13].view('<i2')  # little-endian like ``words``, on any host
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            sample_rate = np.where(
                rate_code == 0,
                timestamp_offset.astype(np.float32),
                np.float32(3200.0) / (1 << (15 - (rate_code & 15))).astype(np.float32),
            ).astype(np.float32)
            offset = np.where(
                rate_code == 0, np.float32(0), (-timestamp_offset.astype(np.float32)) / sample_rate
            ).astype(np.float32)
        packed_time = np.ascontiguousarray(headers[:, 14:18]).view('<u4')[:, 0].astype(np.int64)
        seconds, valid_time = epoch_seconds(
            ((packed_time >> 26) & 0x3F) + 2000, (packed_time >> 22) & 0x0F,
            (packed_time >> 17) & 0x1F, (packed_time >> 12) & 0x1F,
            (packed_time >> 6) & 0x3F, packed_time & 0x3F,
        )

        # Checks in the Java order; each malformed block is counted once.
        remaining = is_data.copy()
        for problem, message in (
            ((rate_code != 0) & (checksums != 0), "CWA block checksum mismatch"),
            (num_axes < 3, "CWA block has fewer than three axes"),
            (bytes_per_sample == 0, "Unsupported CWA packing and axis combination"),
            (~(np.isfinite(sample_rate) & (sample_rate > 0)), "CWA block has no positive finite sample rate"),
            (~valid_time, "CWA block has an invalid timestamp"),
        ):
            rejected = remaining & problem
            if rejected.any():
                rejected_count = int(rejected.sum())
                table.read_errors += rejected_count
                warn(f"Skipping {rejected_count} malformed CWA block(s): {message}")
                remaining &= ~rejected

        index = np.flatnonzero(remaining)
        if len(index) == 0:
            raise FormatError("No valid CWA data block found")
        six_axes = num_axes[index] >= 6
        if (six_axes != six_axes[0]).any():
            first, other = ('AX6', 'AX3') if six_axes[0] else ('AX3', 'AX6')
            raise FormatError(f"CWA axis layout changes from {first} to {other}")

        declared = words[index, 14].astype(np.int64)
        max_samples = PAYLOAD_SIZE // bytes_per_sample[index]
        count = np.minimum(declared, max_samples)
        clamped = int((declared > max_samples).sum())
        if clamped:
            table.read_errors += clamped
            warn(f"Capping malformed CWA sample count at payload capacity ({clamped} block(s))")

        # Block timing, in Java's float/double steps.
        rate = sample_rate[index]
        offset = offset[index]
        whole = np.floor(offset.astype(np.float64))
        offset = (offset - whole.astype(np.float32)).astype(np.float32)
        start = (seconds[index] + whole.astype(np.int64)).astype(np.float64) + offset.astype(np.float64)
        end = start + (count.astype(np.float32) / rate).astype(np.float64)
        # A block starting less than 1 s after the previous block's end continues it.
        joined = np.zeros(len(start), dtype=bool)
        joined[1:] = (start[1:] - end[:-1]) < 1.0
        start[joined] = end[np.flatnonzero(joined) - 1]

        raw_light = words[index, 9].astype(np.int64)
        gyroscope_code = (raw_light >> 10) & 7
        table.index = index
        table.has_gyroscope = bool(six_axes[0])
        table.packing = packing[index]
        table.num_axes = num_axes[index]
        table.count = count
        table.start = start
        table.end = end
        # 1 / 2^(8 + code), exact in float32 (no ldexp: it lacks int64 loops on Windows)
        table.inverse_unit = np.float32(1) / (1 << (8 + ((raw_light >> 13) & 7))).astype(np.float32)
        table.gyroscope_unit = (
            np.float32(32768.0)
            / np.where(gyroscope_code != 0, 8000 // (1 << gyroscope_code), 2000).astype(np.float32)
        )
        table.temperature = _TEMPERATURE[words[index, 10] & 0x3FF]
        table.light = _LIGHT[raw_light & 0x3FF]
        table.offsets = np.concatenate(([0], np.cumsum(count)))
        table.last_sample_rate = float(rate[-1])
        return table


def _read_samples(stream: BufferedReader, num_blocks: int, blocks: _BlockTable) -> Columns:  # noqa: C901
    names = ['x', 'y', 'z']
    if blocks.has_gyroscope:
        names += ['gyro_x', 'gyro_y', 'gyro_z']
    total = blocks.total_samples
    columns: Columns = {'time': np.empty(total, dtype='datetime64[ns]')}
    columns.update((name, np.empty(total, dtype=np.float32)) for name in names + ['temperature', 'light'])
    times = columns['time'].view(np.int64)

    buffer = np.empty((CHUNK_BLOCKS, BLOCK_SIZE), dtype=np.uint8)
    for first in range(0, num_blocks, CHUNK_BLOCKS):
        stop = min(first + CHUNK_BLOCKS, num_blocks)
        chunk = buffer[:stop - first]
        _read_exact(stream, chunk)
        lo, hi = np.searchsorted(blocks.index, [first, stop])
        if lo == hi:
            continue
        rows = blocks.index[lo:hi] - first
        data = chunk if len(rows) == len(chunk) else chunk[rows]
        counts = blocks.count[lo:hi]
        width = int(counts.max())
        if width == 0:
            continue
        rows_out = slice(int(blocks.offsets[lo]), int(blocks.offsets[hi]))
        full = bool((counts == width).all())
        mask = None if full else np.arange(width)[None, :] < counts[:, None]

        # time = start + (i * (end - start)) / count, truncated to whole ms
        sample_times = np.arange(width, dtype=np.float64)[None, :] * (blocks.end[lo:hi] - blocks.start[lo:hi])[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            sample_times /= counts[:, None]
        sample_times += blocks.start[lo:hi, None]
        sample_times *= 1000.0
        millis = java_long(sample_times)  # (long) (time * 1000), saturating like Java
        millis = millis.reshape(-1) if mask is None else millis[mask]
        times[rows_out] = millis_to_nanos(millis)

        payload = np.ascontiguousarray(data[:, HEADER_SIZE:HEADER_SIZE + PAYLOAD_SIZE])
        # Blocks are decoded in groups sharing one packing and one declared axis count (both
        # 4-bit header fields), which together fix the sample layout and stride.
        layout = blocks.packing[lo:hi] * 16 + blocks.num_axes[lo:hi]
        values: Dict[str, Array]
        if (layout == layout[0]).all():
            values = _decode(payload, int(blocks.packing[lo]), int(blocks.num_axes[lo]), width, blocks, lo, hi)
        else:  # e.g. packed and unpacked AX3 blocks, or 6- and 9-axis blocks, in one file
            values = {name: np.zeros((hi - lo, width), dtype=np.float32) for name in names}
            for key in np.unique(layout):
                selected = layout == key
                part = _decode(payload[selected], int(key // 16), int(key % 16), width, blocks, lo, hi, selected)
                for name, decoded in part.items():
                    values[name][selected, :decoded.shape[1]] = decoded
        values['temperature'] = np.broadcast_to(blocks.temperature[lo:hi, None], (hi - lo, width))
        values['light'] = np.broadcast_to(blocks.light[lo:hi, None], (hi - lo, width))
        for name, decoded in values.items():
            # (blocks, width) values; only the first ``count`` of each block are samples
            target = columns[name][rows_out]
            if mask is None:
                target.reshape(-1, width)[...] = decoded
            else:
                target[...] = decoded[mask]
    return columns


def _decode(
    payload: NDArray[np.uint8],
    packing: int,
    num_axes: int,
    width: int,
    blocks: _BlockTable,
    lo: int,
    hi: int,
    selected: Optional[NDArray[np.bool_]] = None,
) -> Dict[str, Array]:
    """Decode up to ``width`` samples per block for blocks sharing one packing and axis count."""

    inverse_unit = blocks.inverse_unit[lo:hi]
    gyroscope_unit = blocks.gyroscope_unit[lo:hi]
    if selected is not None:
        inverse_unit, gyroscope_unit = inverse_unit[selected], gyroscope_unit[selected]
    scale = inverse_unit[:, None]
    values: Dict[str, Array] = {}
    if packing == 0:
        # Three signed 10-bit fields and a shared 2-bit exponent per 32-bit word.
        # Units are powers of two, so value * (1 / unit) equals Java's value / unit.
        packed = payload.view('<u4')[:, :width]
        exponent = (packed >> 30).astype(np.int32)
        for name, shift in (('x', 0), ('y', 10), ('z', 20)):
            field = ((packed >> shift) << 22).view(np.int32) >> 22
            field <<= exponent
            values[name] = field.astype(np.float32) * scale
        return values
    # Unpacked samples are ``num_axes`` 16-bit values each, as declared in the block header:
    # gyroscope x, y, z first if present, then acceleration x, y, z. Any further axes (e.g. a
    # magnetometer in 9-axis blocks) are skipped, as in the Java reader.
    words = payload.view('<i2')
    first_axis = 3 if blocks.has_gyroscope else 0
    for axis, name in enumerate(('x', 'y', 'z')):
        values[name] = words[:, first_axis + axis::num_axes][:, :width].astype(np.float32) * scale
    if blocks.has_gyroscope:
        for axis, name in enumerate(('gyro_x', 'gyro_y', 'gyro_z')):
            values[name] = words[:, axis::num_axes][:, :width].astype(np.float32) / gyroscope_unit[:, None]
    return values
