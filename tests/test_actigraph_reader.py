import logging
import operator
import struct
import warnings
import zipfile
from functools import reduce

import numpy as np
import pytest

from actipy import actigraph_reader as actigraph
from actipy._reader_support import FormatError


@pytest.fixture
def actigraph_reader():
    def run(input_file):
        return actigraph.read(str(input_file))

    return run


def _actigraph_packet(timestamp, payload, record_type=26, sync_byte=0x1E):
    header = struct.pack("<BBIH", sync_byte, record_type, timestamp, len(payload))
    checksum = (~reduce(operator.xor, header + payload)) & 0xFF
    return header + payload + bytes([checksum])


def _actigraph_metadata(sample_rate, acceleration_scale, start_millis=0):
    ticks = 621355968000000000 + start_millis * 10_000
    return (
        "Serial Number: MOS123\n"
        f"Sample Rate: {sample_rate}\n"
        f"Start Date: {ticks}\n"
        "Acceleration Min: -8.0\n"
        "Acceleration Max: 8.0\n"
        f"Acceleration Scale: {acceleration_scale}\n"
    )


def _write_gt3x_v2(path, metadata, packets):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.txt", metadata)
        archive.writestr("log.bin", b"".join(packets))


def _pack_12_bit_pair(first, second):
    x1, y1, z1 = (value & 0xFFF for value in first)
    x2, y2, z2 = (value & 0xFFF for value in second)
    return bytes(
        [
            y1 >> 4,
            ((y1 & 0xF) << 4) | (x1 >> 8),
            x1 & 0xFF,
            z1 >> 4,
            ((z1 & 0xF) << 4) | (y2 >> 8),
            y2 & 0xFF,
            x2 >> 4,
            ((x2 & 0xF) << 4) | (z2 >> 8),
            z2 & 0xFF,
        ]
    )


def _pack_12_bit_sample(sample, unused_nibble=0xF):
    packed = bytearray(_pack_12_bit_pair(sample, (0, 0, 0)))
    packed[4] |= unused_nibble & 0xF
    return bytes(packed[:5])


def _write_gt3x_v1(path, metadata, payload):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.txt", metadata)
        archive.writestr("activity.bin", payload)
        archive.writestr("lux.bin", b"")


def test_actigraph_reader_decodes_v2_samples_and_timestamps(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "sample-v2.gt3x"
    timestamp = 1_700_000_000
    payload = struct.pack("<hhhhhh", 256, -256, 128, -512, 64, -128)
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [_actigraph_packet(timestamp, payload)],
    )

    data, info = actigraph_reader(input_file)
    assert tuple(data) == ("time", "x", "y", "z")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000, timestamp * 1_000_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


@pytest.mark.parametrize("usb_record_type", [0, 26])
def test_actigraph_reader_ignores_one_byte_usb_activity_records(
        actigraph_reader, tmp_path, usb_record_type):
    input_file = tmp_path / "usb-event.gt3x"
    timestamp = 1_700_000_000
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [
            _actigraph_packet(timestamp, b"\x00", record_type=usb_record_type),
            _actigraph_packet(
                timestamp + 1,
                struct.pack("<hhh", 256, -256, 128),
            ),
        ],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [(timestamp + 1) * 1_000_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


def test_actigraph_reader_preserves_java_midpoint_rounding(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "rounded-v2.gt3x"
    timestamp = 1_700_000_000
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(30, 256),
        [_actigraph_packet(timestamp, struct.pack("<hhh", 16, -16, 24))],
    )

    data, info = actigraph_reader(input_file)
    # Match Java Math.round, which rounds negative half-values toward zero.
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        np.array([[0.063, -0.062, 0.094]], dtype=np.float32),
    )


def test_actigraph_reader_decodes_v1_packed_sample_pairs(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "sample-v1.gt3x"
    start_millis = 1_700_000_000_000
    payload = _pack_12_bit_pair(
        (256, -256, 128),
        (-512, 64, -128),
    )
    _write_gt3x_v1(
        input_file,
        _actigraph_metadata(4, 256, start_millis),
        payload,
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [start_millis * 1_000_000, start_millis * 1_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


def test_actigraph_reader_decodes_v1_odd_packed_sample(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "odd-v1.gt3x"
    start_millis = 1_700_000_000_000
    payload = _pack_12_bit_pair(
        (256, -256, 128),
        (-512, 64, -128),
    ) + _pack_12_bit_sample((768, -768, 384))
    _write_gt3x_v1(
        input_file,
        _actigraph_metadata(4, 256, start_millis),
        payload,
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5], [3.0, -3.0, 1.5]],
    )
    assert info["ReadErrors"] == 0


def test_actigraph_reader_preserves_unsigned_v2_timestamp(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "unsigned-timestamp.gt3x"
    timestamp = 2_200_000_000
    payload = struct.pack("<hhh", 256, 0, -256)
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [_actigraph_packet(timestamp, payload)],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000],
    )


def test_actigraph_reader_uses_in_stream_parameter_scale(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "parameter-scale.gt3x"
    timestamp = 1_700_000_000
    encoded_scale_256 = 0x09400000
    parameter = bytes([0, 0, 55, 0]) + struct.pack("<I", encoded_scale_256)
    activity = struct.pack("<hhh", 256, -256, 128)
    metadata = (
        "Serial Number: UNKNOWN123\n"
        "Sample Rate: 4\n"
        "Start Date: 621355968000000000\n"
        "Acceleration Min: -8.0\n"
        "Acceleration Max: 8.0\n"
    )
    _write_gt3x_v2(
        input_file,
        metadata,
        [
            _actigraph_packet(timestamp, parameter, record_type=21),
            _actigraph_packet(timestamp, activity),
        ],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )


def test_actigraph_reader_decodes_high_bit_parameter_scale_byte(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "high-bit-parameter-scale.gt3x"
    timestamp = 1_700_000_000
    encoded_scale = 0x09400080
    parameter = bytes([0, 0, 55, 0]) + struct.pack("<I", encoded_scale)
    activity = struct.pack("<hhh", 32767, -32768, 16384)
    metadata = (
        "Serial Number: UNKNOWN123\n"
        "Sample Rate: 4\n"
        "Start Date: 621355968000000000\n"
        "Acceleration Min: -8.0\n"
        "Acceleration Max: 8.0\n"
    )
    _write_gt3x_v2(
        input_file,
        metadata,
        [
            _actigraph_packet(timestamp, parameter, record_type=21),
            _actigraph_packet(timestamp, activity),
        ],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        np.array([[127.992, -127.996, 63.998]], dtype=np.float32),
    )


def test_actigraph_reader_decodes_v2_packed_activity(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "packed-v2.gt3x"
    timestamp = 1_700_000_000
    payload = _pack_12_bit_pair(
        (256, -256, 128),
        (-512, 64, -128),
    )
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [_actigraph_packet(timestamp, payload, record_type=0)],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000, timestamp * 1_000_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )


def test_actigraph_reader_decodes_v2_odd_packed_sample(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "odd-packed-v2.gt3x"
    timestamp = 1_700_000_000
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [
            _actigraph_packet(
                timestamp,
                _pack_12_bit_pair((256, -256, 128), (-512, 64, -128)),
                record_type=0,
            ),
            _actigraph_packet(
                timestamp + 1,
                _pack_12_bit_sample((768, -768, 384)),
                record_type=0,
            ),
        ],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5], [3.0, -3.0, 1.5]],
    )
    assert info["ReadErrors"] == 0


def test_actigraph_reader_reuses_payload_buffer_for_smaller_short_packet(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "shrinking-payload-v2.gt3x"
    timestamp = 1_700_000_000
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [
            _actigraph_packet(
                timestamp,
                struct.pack("<hhhhhh", 256, 0, 0, 512, 0, 0),
            ),
            _actigraph_packet(
                timestamp + 1,
                struct.pack("<hhh", 768, 0, 0),
            ),
        ],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(data["x"], [1.0, 2.0, 3.0])
    assert info["ReadErrors"] == 0


def test_actigraph_reader_rejects_wrong_packet_sync_byte(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "wrong-sync.gt3x"
    packet = _actigraph_packet(
        1_700_000_000,
        struct.pack("<hhh", 256, 0, 0),
        sync_byte=0x00,
    )
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [packet],
    )

    with pytest.raises(FormatError, match="Invalid GT3X packet synchronization byte"):
        actigraph_reader(input_file)


def test_actigraph_reader_rejects_bad_packet_checksum(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "bad-checksum.gt3x"
    packet = bytearray(
        _actigraph_packet(1_700_000_000, struct.pack("<hhh", 256, 0, 0))
    )
    packet[-1] ^= 0x01
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [bytes(packet)],
    )

    with pytest.raises(FormatError, match="GT3X packet checksum mismatch"):
        actigraph_reader(input_file)


def test_actigraph_reader_rejects_truncated_packet_without_fabricating_samples(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "truncated.gt3x"
    packet = _actigraph_packet(
        1_700_000_000,
        struct.pack("<hhhhhh", 256, -256, 128, -512, 64, -128),
    )
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [packet[:-4]],
    )

    with pytest.raises(FormatError, match="Unexpected end of V2 packet payload"):
        actigraph_reader(input_file)


@pytest.mark.parametrize("scale", ["0x1p1024", "-0x1p1024", "0x1.fffffffffffff8p1023"])
def test_actigraph_reader_falls_back_to_device_scale_when_scale_overflows(
        actigraph_reader, tmp_path, caplog, scale):
    # Java's Double.parseDouble returns +-Infinity for an overflowing hexadecimal value (the
    # last one overflows only after rounding), so the scale is ignored as invalid and the MOS
    # serial's default of 256 is used, rather than the read being aborted.
    input_file = tmp_path / "overflowing-scale.gt3x"
    payload = struct.pack("<hhhhhh", 256, -256, 128, -512, 64, -128)
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, scale),
        [_actigraph_packet(1_700_000_000, payload)],
    )

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = actigraph_reader(input_file)

    assert "Ignoring invalid Acceleration Scale" in caplog.text
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )
    assert info["ReadErrors"] == 0


_LONG_MAX, _LONG_MIN = 2**63 - 1, -(2**63)


@pytest.mark.parametrize(
    ("version", "expected_nanos"),
    [
        # firstSampleTime (0 here) + Math.round(1e19): Math.round saturates at Long.MAX_VALUE
        (1, [0, _LONG_MAX]),
        # timestamp * 1000 + Math.round(1e19) wraps around as a Java long; toNanos then saturates
        (2, [1_700_000_000 * 1_000_000_000, _LONG_MIN]),
    ],
)
def test_actigraph_reader_saturates_timestamps_like_java_at_extreme_sample_rate(
        actigraph_reader, tmp_path, version, expected_nanos):
    # A sample rate of 1e-16 Hz puts the second sample 1e19 ms after the first, beyond the long
    # range. The expected times are the Java reader's output for these files. Converting them must
    # not warn either (NumPy warns on out-of-range float-to-int casts).
    input_file = tmp_path / f"extreme-rate-v{version}.gt3x"
    metadata = _actigraph_metadata(1e-16, 256)
    if version == 1:
        _write_gt3x_v1(input_file, metadata, _pack_12_bit_pair((256, -256, 128), (-512, 64, -128)))
    else:
        payload = struct.pack("<hhhhhh", 256, -256, 128, -512, 64, -128)
        _write_gt3x_v2(input_file, metadata, [_actigraph_packet(1_700_000_000, payload)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data, info = actigraph_reader(input_file)

    np.testing.assert_array_equal(data["time"].view("int64"), expected_nanos)
    np.testing.assert_array_equal(data["x"], [1.0, -2.0])
    assert info["ReadErrors"] == 0


def test_actigraph_reader_keeps_samples_before_truncated_trailing_packet(
        actigraph_reader, tmp_path, caplog):
    input_file = tmp_path / "truncated-tail.gt3x"
    valid = _actigraph_packet(
        1_700_000_000,
        struct.pack("<hhh", 256, -256, 128),
    )
    truncated = _actigraph_packet(
        1_700_000_001,
        struct.pack("<hhh", 512, 0, 0),
    )[:-4]
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [valid, truncated],
    )

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = actigraph_reader(input_file)

    assert "Stopping at truncated GT3X data" in caplog.text
    np.testing.assert_array_equal(data["x"], [1.0])
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 4.0,
    }


def test_actigraph_reader_rejects_invalid_archive(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "invalid.gt3x"
    with zipfile.ZipFile(input_file, "w") as archive:
        archive.writestr("info.txt", _actigraph_metadata(4, 256))

    with pytest.raises(FormatError, match="not a supported V1 or V2 GT3X archive"):
        actigraph_reader(input_file)


def test_actigraph_reader_rejects_missing_required_metadata(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "missing-rate.gt3x"
    metadata = _actigraph_metadata(4, 256).replace("Sample Rate: 4\n", "")
    _write_gt3x_v2(input_file, metadata, [])

    with pytest.raises(FormatError, match="positive finite Sample Rate"):
        actigraph_reader(input_file)


def test_actigraph_reader_preserves_existing_timezone_semantics(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "timezone.gt3x"
    timestamp = 1_700_000_000
    metadata = _actigraph_metadata(4, 256) + "TimeZone: -05:00:00\n"
    _write_gt3x_v2(
        input_file,
        metadata,
        [_actigraph_packet(timestamp, struct.pack("<hhh", 256, 0, 0))],
    )

    data, info = actigraph_reader(input_file)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000],
    )
