import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "actipy"


@pytest.fixture(scope="session")
def axivity_reader(tmp_path_factory):
    java = shutil.which("java")
    javac = shutil.which("javac")
    if java is None or javac is None:
        missing = [
            command
            for command, executable in (("java", java), ("javac", javac))
            if executable is None
        ]
        pytest.skip("Java toolchain unavailable: missing " + ", ".join(missing))

    classes = tmp_path_factory.mktemp("axivity-reader-classes")
    subprocess.run(
        [
            javac,
            "-d",
            str(classes),
            str(SOURCE_DIR / "NpyWriter.java"),
            str(SOURCE_DIR / "AxivityReader.java"),
        ],
        check=True,
    )

    def run(input_file, output_dir):
        output_dir.mkdir()
        return subprocess.run(
            [
                java,
                "-cp",
                str(classes),
                "AxivityReader",
                "-i",
                str(input_file),
                "-o",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return run


def _read_info(output_dir):
    return dict(
        line.split(":", 1)
        for line in (output_dir / "info.txt").read_text().splitlines()
    )


def _cwa_timestamp(year, month, day, hour, minute, second):
    return (
        ((year - 2000) << 26)
        | (month << 22)
        | (day << 17)
        | (hour << 12)
        | (minute << 6)
        | second
    )


def _axivity_block(samples, frequency=4, raw_light=341, raw_temperature=270):
    num_axes = len(samples[0])
    block = bytearray(512)
    block[0:2] = b"AX"
    struct.pack_into(
        "<I", block, 14, _cwa_timestamp(2024, 1, 2, 3, 4, 5)
    )
    struct.pack_into("<H", block, 18, raw_light)
    struct.pack_into("<H", block, 20, raw_temperature)
    block[24] = 0
    block[25] = (num_axes << 4) | 2
    struct.pack_into("<h", block, 26, frequency)
    struct.pack_into("<H", block, 28, len(samples))
    for sample_number, sample in enumerate(samples):
        struct.pack_into(
            "<" + "h" * num_axes,
            block,
            30 + 2 * num_axes * sample_number,
            *sample,
        )
    return bytes(block)


def test_axivity_reader_decodes_ax3_samples_and_metadata(
        axivity_reader, tmp_path):
    input_file = tmp_path / "sample.cwa"
    output_dir = tmp_path / "output"
    input_file.write_bytes(
        _axivity_block([(256, -256, 128), (512, 0, -512)])
    )

    axivity_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    assert data.dtype.names == (
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
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


def test_axivity_reader_detects_ax6_gyroscope_layout(
        axivity_reader, tmp_path):
    input_file = tmp_path / "sample-ax6.cwa"
    output_dir = tmp_path / "output"
    raw_light = (2 << 10) | 341
    input_file.write_bytes(
        _axivity_block(
            [(16384, -8192, 0, 256, -256, 128)],
            raw_light=raw_light,
        )
    )

    axivity_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    assert data.dtype.names == (
        "time", "x", "y", "z", "gyro_x", "gyro_y", "gyro_z",
        "temperature", "light",
    )
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[1.0, -1.0, 0.5]],
    )
    np.testing.assert_allclose(
        np.column_stack((data["gyro_x"], data["gyro_y"], data["gyro_z"])),
        [[1000.0, -500.0, 0.0]],
    )


def test_axivity_reader_skips_block_with_zero_sample_rate(
        axivity_reader, tmp_path):
    input_file = tmp_path / "bad-rate.cwa"
    output_dir = tmp_path / "output"
    input_file.write_bytes(
        _axivity_block([(256, 256, 256)], frequency=0)
        + _axivity_block([(512, 768, 1024)], frequency=4)
    )

    axivity_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(data["x"], [2.0])
    np.testing.assert_array_equal(data["y"], [3.0])
    np.testing.assert_array_equal(data["z"], [4.0])
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "1",
        "SampleRate": "4.0",
    }
