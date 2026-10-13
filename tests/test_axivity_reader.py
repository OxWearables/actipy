import logging
import struct
import warnings

import numpy as np
import pytest

import actipy
from actipy import axivity_reader as axivity
from actipy._reader_support import FormatError


@pytest.fixture
def axivity_reader():
    def run(input_file):
        return axivity.read(str(input_file))

    return run


def _cwa_timestamp(year, month, day, hour, minute, second):
    return (
        ((year - 2000) << 26)
        | (month << 22)
        | (day << 17)
        | (hour << 12)
        | (minute << 6)
        | second
    )


def _axivity_block(
        samples,
        frequency=4,
        raw_light=341,
        raw_temperature=270,
        rate_code=0,
        timestamp_offset=0,
        packing=2,
        timestamp=(2024, 1, 2, 3, 4, 5)):
    num_axes = len(samples[0])
    block = bytearray(512)
    block[0:2] = b"AX"
    struct.pack_into("<H", block, 2, 508)
    struct.pack_into(
        "<I", block, 14, _cwa_timestamp(*timestamp)
    )
    struct.pack_into("<H", block, 18, raw_light)
    struct.pack_into("<H", block, 20, raw_temperature)
    block[24] = rate_code
    block[25] = (num_axes << 4) | packing
    block_value = frequency if rate_code == 0 else timestamp_offset
    struct.pack_into("<h", block, 26, block_value)
    struct.pack_into("<H", block, 28, len(samples))
    for sample_number, sample in enumerate(samples):
        if packing == 2:
            struct.pack_into(
                "<" + "h" * num_axes,
                block,
                30 + 2 * num_axes * sample_number,
                *sample,
            )
        elif packing == 0:
            assert num_axes == 3
            x, y, z = (value & 0x3FF for value in sample)
            packed = x | (y << 10) | (z << 20)
            struct.pack_into("<I", block, 30 + 4 * sample_number, packed)
        else:
            raise ValueError(f"Unsupported packing mode: {packing}")

    if rate_code != 0:
        words = struct.unpack("<255h", block[:510])
        struct.pack_into("<H", block, 510, (-sum(words)) & 0xFFFF)

    return bytes(block)


def test_axivity_reader_decodes_ax3_samples_and_metadata(
        axivity_reader, tmp_path):
    input_file = tmp_path / "sample.cwa"
    input_file.write_bytes(
        _axivity_block([(256, -256, 128), (512, 0, -512)])
    )

    data, info = axivity_reader(input_file)
    assert tuple(data) == (
        "time", "x", "y", "z", "temperature", "light"
    )
    np.testing.assert_array_equal(
        data["time"],
        np.array(
            ["2024-01-02T03:04:05.000", "2024-01-02T03:04:05.250"],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(data["x"], [1.0, 2.0])
    np.testing.assert_array_equal(data["y"], [-1.0, 0.0])
    np.testing.assert_array_equal(data["z"], [0.5, -2.0])
    np.testing.assert_array_equal(data["temperature"], [20.0, 20.0])
    np.testing.assert_allclose(data["light"], [10.0, 10.0])
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


def test_axivity_reader_detects_ax6_gyroscope_layout(
        axivity_reader, tmp_path):
    input_file = tmp_path / "sample-ax6.cwa"
    raw_light = (2 << 10) | 341
    input_file.write_bytes(
        _axivity_block(
            [
                (16384, -8192, 0, 256, -256, 128),
                (8192, 0, -4096, 512, 0, -512),
                (0, 4096, -8192, -256, 128, 256),
            ],
            raw_light=raw_light,
        )
    )

    data, info = axivity_reader(input_file)
    assert tuple(data) == (
        "time", "x", "y", "z", "gyro_x", "gyro_y", "gyro_z",
        "temperature", "light",
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [2.0, 0.0, -2.0], [-1.0, 0.5, 1.0]],
    )
    np.testing.assert_allclose(
        np.column_stack((data["gyro_x"], data["gyro_y"], data["gyro_z"])),
        [[1000.0, -500.0, 0.0], [500.0, 0.0, -250.0], [0.0, 250.0, -500.0]],
    )


_GYROSCOPE_RAW_LIGHT = (2 << 10) | 341  # gyroscope range code 2 (2000 dps), light 341
_ACCELERATION_UNIT = np.float32(256)
_GYROSCOPE_UNIT = np.float32(32768) / np.float32(2000)


def _expected_motion(samples):
    """Java's values for unpacked samples: gyroscope x, y, z first if present, then acceleration."""

    raw = np.array(samples, dtype=np.float32)
    if raw.shape[1] < 6:
        return {"xyz": raw[:, 0:3] / _ACCELERATION_UNIT}
    return {"xyz": raw[:, 3:6] / _ACCELERATION_UNIT, "gyro": raw[:, 0:3] / _GYROSCOPE_UNIT}


@pytest.mark.parametrize("num_axes", [4, 5, 7, 8, 9, 15])
def test_axivity_reader_uses_declared_axis_count_as_sample_stride(
        axivity_reader, tmp_path, num_axes):
    # Each sample holds the declared number of 16-bit values; axes beyond the gyroscope and
    # accelerometer (e.g. a magnetometer) are skipped, as in the Java reader, rather than read
    # as the start of the next sample. Every value is distinct, so a wrong stride shows.
    capacity = 480 // (2 * num_axes)
    samples = [[16 * sample + axis for axis in range(num_axes)] for sample in range(capacity)]
    input_file = tmp_path / f"axes-{num_axes}.cwa"
    input_file.write_bytes(_axivity_block(samples, raw_light=_GYROSCOPE_RAW_LIGHT))

    data, info = axivity_reader(input_file)

    expected = _expected_motion(samples)
    np.testing.assert_array_equal(np.column_stack((data["x"], data["y"], data["z"])), expected["xyz"])
    if num_axes >= 6:
        np.testing.assert_array_equal(
            np.column_stack((data["gyro_x"], data["gyro_y"], data["gyro_z"])), expected["gyro"]
        )
    else:
        assert "gyro_x" not in data
    assert info["ReadErrors"] == 0


@pytest.mark.parametrize(
    "layouts",
    [
        pytest.param([(6, 2), (9, 2), (6, 2)], id="6-and-9-axis"),
        pytest.param([(3, 2), (4, 2), (3, 0), (5, 2)], id="3-4-5-axis-and-packed"),
    ],
)
def test_axivity_reader_decodes_each_block_with_its_own_layout(
        axivity_reader, tmp_path, layouts):
    # Blocks with different axis counts (or packings) can share one output schema (AX3 or AX6);
    # each block must still be decoded with its own sample stride.
    blocks, expected = [], []
    for number, (num_axes, packing) in enumerate(layouts):
        count = 480 // (4 if packing == 0 else 2 * num_axes) - number  # full and partly filled blocks
        samples = [
            [(100 * number + 16 * sample + axis) % 500 for axis in range(num_axes)]  # fits 10-bit packing
            for sample in range(count)
        ]
        blocks.append(_axivity_block(
            samples, packing=packing, raw_light=_GYROSCOPE_RAW_LIGHT, timestamp=(2024, 1, 2, 3, 4 + number, 5)
        ))
        expected.append(_expected_motion(samples))
    input_file = tmp_path / "mixed-layouts.cwa"
    input_file.write_bytes(b"".join(blocks))

    data, info = axivity_reader(input_file)

    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        np.concatenate([block["xyz"] for block in expected]),
    )
    if layouts[0][0] >= 6:
        np.testing.assert_array_equal(
            np.column_stack((data["gyro_x"], data["gyro_y"], data["gyro_z"])),
            np.concatenate([block["gyro"] for block in expected]),
        )
    assert info["ReadErrors"] == 0


def test_axivity_reader_decodes_packed_samples(axivity_reader, tmp_path):
    input_file = tmp_path / "packed.cwa"
    input_file.write_bytes(
        _axivity_block([(256, -256, 128)], packing=0)
    )

    data, info = axivity_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )


@pytest.mark.parametrize(
    ("timestamp_offset", "expected_times"),
    [
        (25, ["2024-01-02T03:04:04.750", "2024-01-02T03:04:04.760"]),
        (-25, ["2024-01-02T03:04:05.250", "2024-01-02T03:04:05.260"]),
    ],
)
def test_axivity_reader_applies_signed_rate_code_timestamp_offset(
        axivity_reader, tmp_path, timestamp_offset, expected_times):
    input_file = tmp_path / "timestamp-offset.cwa"
    input_file.write_bytes(
        _axivity_block(
            [(256, 0, 0), (512, 0, 0)],
            rate_code=10,
            timestamp_offset=timestamp_offset,
        )
    )

    data, info = axivity_reader(input_file)
    np.testing.assert_array_equal(
        data["time"],
        np.array(expected_times, dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(data["x"], [1.0, 2.0])
    assert info["SampleRate"] == 100.0


def test_axivity_reader_skips_bad_checksum_and_continues(
        axivity_reader, tmp_path):
    input_file = tmp_path / "bad-checksum.cwa"
    corrupt_block = bytearray(
        _axivity_block([(256, 256, 256)], rate_code=10)
    )
    corrupt_block[-1] ^= 0x01
    valid_block = _axivity_block(
        [(512, 768, 1024)],
        rate_code=10,
        timestamp=(2024, 1, 2, 3, 4, 6),
    )
    input_file.write_bytes(bytes(corrupt_block) + valid_block)

    data, info = axivity_reader(input_file)
    np.testing.assert_array_equal(data["x"], [2.0])
    np.testing.assert_array_equal(data["y"], [3.0])
    np.testing.assert_array_equal(data["z"], [4.0])
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 100.0,
    }


def test_axivity_reader_skips_block_with_zero_sample_rate(
        axivity_reader, tmp_path):
    input_file = tmp_path / "bad-rate.cwa"
    input_file.write_bytes(
        _axivity_block([(256, 256, 256)], frequency=0)
        + _axivity_block([(512, 768, 1024)], frequency=4)
    )

    data, info = axivity_reader(input_file)
    np.testing.assert_array_equal(data["x"], [2.0])
    np.testing.assert_array_equal(data["y"], [3.0])
    np.testing.assert_array_equal(data["z"], [4.0])
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 4.0,
    }


def test_axivity_reader_ignores_corrupt_first_block_when_selecting_schema(
        axivity_reader, tmp_path):
    input_file = tmp_path / "corrupt-layout.cwa"
    corrupt = bytearray(
        _axivity_block([(1, 2, 3)], rate_code=10)
    )
    corrupt[25] = (6 << 4) | 2
    valid = _axivity_block(
        [(256, 512, 768)],
        rate_code=10,
        timestamp=(2024, 1, 2, 3, 4, 6),
    )
    input_file.write_bytes(bytes(corrupt) + valid)

    data, info = axivity_reader(input_file)
    assert tuple(data) == (
        "time", "x", "y", "z", "temperature", "light"
    )
    np.testing.assert_array_equal(data["x"], [1.0])
    assert info["ReadErrors"] == 1


def test_axivity_reader_keeps_samples_before_partial_trailing_block(
        axivity_reader, tmp_path, caplog):
    input_file = tmp_path / "partial.cwa"
    input_file.write_bytes(_axivity_block([(256, 0, 0)]) + b"AX")

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = axivity_reader(input_file)

    assert "Unexpected partial CWA block" in caplog.text
    np.testing.assert_array_equal(data["x"], [1.0])
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 4.0,
    }


class _TrickleStream:
    """A stream returning at most ``step`` bytes per ``readinto`` call, as a slow or remote file may."""

    def __init__(self, data, step):
        self.data, self.step, self.position = data, step, 0

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        chunk = self.data[self.position:self.position + min(self.step, len(view))]
        view[:len(chunk)] = chunk
        self.position += len(chunk)
        return len(chunk)


def test_axivity_exact_read_fills_the_buffer_across_short_reads():
    # The buffer is two 512-byte rows; a 600-byte first read must not be mistaken for a full one.
    data = bytes(range(256)) * 4
    buffer = np.zeros((2, 512), dtype=np.uint8)

    axivity._read_exact(_TrickleStream(data, 600), buffer)

    assert buffer.tobytes() == data


def test_axivity_exact_read_reports_premature_end_of_file():
    buffer = np.zeros((2, 512), dtype=np.uint8)

    with pytest.raises(FormatError, match="Unexpected end of CWA file"):
        axivity._read_exact(_TrickleStream(bytes(600), 600), buffer)


def test_axivity_reader_recovery_is_not_affected_by_warning_filters(
        axivity_reader, tmp_path):
    # Recoverable problems are logged, not raised with warnings.warn, so running with
    # warnings as errors (e.g. pytest -W error) still returns the valid samples.
    input_file = tmp_path / "partial.cwa"
    input_file.write_bytes(_axivity_block([(256, 0, 0)]) + b"AX")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data, info = axivity_reader(input_file)

    np.testing.assert_array_equal(data["x"], [1.0])
    assert info["ReadErrors"] == 1


def test_read_device_recovers_from_truncated_cwa_when_warnings_are_errors(tmp_path):
    # The same through the public API: no reader warning escapes read_device.
    input_file = tmp_path / "partial.cwa"
    input_file.write_bytes(
        b"".join(_axivity_block([(256, 0, 0)] * 4, timestamp=(2024, 1, 2, 3, 4 + i, 5)) for i in range(3))
        + b"AX"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data, info = actipy.read_device(
            str(input_file),
            lowpass_hz=None,
            calibrate_gravity=False,
            detect_nonwear=False,
            resample_hz=None,
            verbose=False,
        )

    assert len(data) == 12
    assert info["ReadErrors"] == 1


def test_axivity_reader_rejects_layout_changes(
        axivity_reader, tmp_path):
    input_file = tmp_path / "mixed-layout.cwa"
    input_file.write_bytes(
        _axivity_block([(256, 0, 0)], rate_code=10)
        + _axivity_block(
            [(0, 0, 0, 256, 0, 0)],
            rate_code=10,
            timestamp=(2024, 1, 2, 3, 4, 6),
        )
    )

    with pytest.raises(FormatError, match="CWA axis layout changes from AX3 to AX6"):
        axivity_reader(input_file)
