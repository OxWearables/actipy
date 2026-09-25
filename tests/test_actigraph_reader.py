import operator
import shutil
import struct
import subprocess
import zipfile
from functools import reduce
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "actipy"


@pytest.fixture(scope="session")
def actigraph_reader(tmp_path_factory):
    java = shutil.which("java")
    javac = shutil.which("javac")
    if java is None or javac is None:
        missing = [
            command
            for command, executable in (("java", java), ("javac", javac))
            if executable is None
        ]
        pytest.skip("Java toolchain unavailable: missing " + ", ".join(missing))

    classes = tmp_path_factory.mktemp("actigraph-reader-classes")
    subprocess.run(
        [
            javac,
            "-d",
            str(classes),
            str(SOURCE_DIR / "NpyWriter.java"),
            str(SOURCE_DIR / "ActigraphReader.java"),
        ],
        check=True,
    )

    def run(input_file, output_dir, check=True):
        output_dir.mkdir()
        return subprocess.run(
            [
                java,
                "-cp",
                str(classes),
                "ActigraphReader",
                "-i",
                str(input_file),
                "-o",
                str(output_dir),
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    return run


def _read_info(output_dir):
    return dict(
        line.split(":", 1)
        for line in (output_dir / "info.txt").read_text().splitlines()
    )


def _actigraph_packet(timestamp, payload, record_type=26):
    header = struct.pack("<BBIH", 0x1E, record_type, timestamp, len(payload))
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


def _write_gt3x_v1(path, metadata, payload):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("info.txt", metadata)
        archive.writestr("activity.bin", payload)
        archive.writestr("lux.bin", b"")


def test_actigraph_reader_decodes_v2_samples_and_timestamps(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "sample-v2.gt3x"
    output_dir = tmp_path / "output"
    timestamp = 1_700_000_000
    payload = struct.pack("<hhhhhh", 256, -256, 128, -512, 64, -128)
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [_actigraph_packet(timestamp, payload)],
    )

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    assert data.dtype.names == ("time", "x", "y", "z")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000, timestamp * 1_000_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


@pytest.mark.parametrize("usb_record_type", [0, 26])
def test_actigraph_reader_ignores_one_byte_usb_activity_records(
        actigraph_reader, tmp_path, usb_record_type):
    input_file = tmp_path / "usb-event.gt3x"
    output_dir = tmp_path / "output"
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

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [(timestamp + 1) * 1_000_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


def test_actigraph_reader_preserves_java_midpoint_rounding(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "rounded-v2.gt3x"
    output_dir = tmp_path / "output"
    timestamp = 1_700_000_000
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(30, 256),
        [_actigraph_packet(timestamp, struct.pack("<hhh", 16, -16, 24))],
    )

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    # Java Math.round sends a negative midpoint toward zero; this expected
    # value preserves that behavior.
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        np.array([[0.063, -0.062, 0.094]], dtype=np.float32),
    )


def test_actigraph_reader_decodes_v1_packed_sample_pairs(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "sample-v1.gt3x"
    output_dir = tmp_path / "output"
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

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [start_millis * 1_000_000, start_millis * 1_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


def test_actigraph_reader_preserves_unsigned_v2_timestamp(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "unsigned-timestamp.gt3x"
    output_dir = tmp_path / "output"
    timestamp = 2_200_000_000
    payload = struct.pack("<hhh", 256, 0, -256)
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [_actigraph_packet(timestamp, payload)],
    )

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000],
    )


def test_actigraph_reader_uses_in_stream_parameter_scale(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "parameter-scale.gt3x"
    output_dir = tmp_path / "output"
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

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )


def test_actigraph_reader_decodes_high_bit_parameter_scale_byte(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "high-bit-parameter-scale.gt3x"
    output_dir = tmp_path / "output"
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

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        np.array([[127.992, -127.996, 63.998]], dtype=np.float32),
    )


def test_actigraph_reader_decodes_v2_packed_activity(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "packed-v2.gt3x"
    output_dir = tmp_path / "output"
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

    actigraph_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [timestamp * 1_000_000_000, timestamp * 1_000_000_000 + 250_000_000],
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5], [-2.0, 0.25, -0.5]],
    )


def test_actigraph_reader_rejects_bad_packet_checksum(
        actigraph_reader, tmp_path):
    input_file = tmp_path / "bad-checksum.gt3x"
    output_dir = tmp_path / "output"
    packet = bytearray(
        _actigraph_packet(1_700_000_000, struct.pack("<hhh", 256, 0, 0))
    )
    packet[-1] ^= 0x01
    _write_gt3x_v2(
        input_file,
        _actigraph_metadata(4, 256),
        [bytes(packet)],
    )

    result = actigraph_reader(input_file, output_dir, check=False)

    assert result.returncode != 0
