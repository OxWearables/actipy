import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "actipy"


def _geneactiv_header(gains, offsets, page_count):
    lines = [f"Header line {number}" for number in range(1, 48)]
    lines.extend(
        [
            f"x gain:{gains[0]}",
            f"x offset:{offsets[0]}",
            f"y gain:{gains[1]}",
            f"y offset:{offsets[1]}",
            f"z gain:{gains[2]}",
            f"z offset:{offsets[2]}",
            "Volts:300",
            "Lux:800",
            "",
            "Memory Status",
            f"Number of Pages:{page_count}",
            "",
        ]
    )
    assert len(lines) == 59
    return lines


def _encode_sample(x, y, z, auxiliary=0):
    return (
        "".join(f"{value & 0xFFF:03X}" for value in (x, y, z))
        + f"{auxiliary & 0xFFF:03X}"
    )


def _geneactiv_block(sequence, timestamp, temperature, frequency, payload):
    return [
        "Recorded Data",
        "Device Unique Serial Code:012345",
        f"Sequence Number:{sequence}",
        f"Page Time:{timestamp}",
        "Unassigned:",
        f"Temperature:{temperature}",
        "Battery voltage:4.0",
        "Device Status:Recording",
        f"Measurement Frequency:{frequency}",
        payload,
    ]


def _write_geneactiv(path, gains, offsets, blocks):
    lines = _geneactiv_header(gains, offsets, len(blocks))
    for block in blocks:
        lines.extend(block)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture(scope="session")
def geneactiv_reader(tmp_path_factory):
    java = shutil.which("java")
    javac = shutil.which("javac")
    if java is None or javac is None:
        missing = [
            command
            for command, executable in (("java", java), ("javac", javac))
            if executable is None
        ]
        pytest.skip("Java toolchain unavailable: missing " + ", ".join(missing))

    classes = tmp_path_factory.mktemp("geneactiv-reader-classes")
    subprocess.run(
        [
            javac,
            "-d",
            str(classes),
            str(SOURCE_DIR / "NpyWriter.java"),
            str(SOURCE_DIR / "GENEActivReader.java"),
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
                "GENEActivReader",
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


def test_geneactiv_reader_decodes_samples_and_page_metadata(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "sample.bin"
    output_dir = tmp_path / "output"
    gains = (200, 250, 400)
    offsets = (100, -50, 20)
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:600",
            21.5,
            4,
            _encode_sample(3, -4, 5) + _encode_sample(-2047, 2047, -1),
        ),
        _geneactiv_block(
            1,
            "2024-01-02 03:04:06:100",
            19.25,
            4,
            _encode_sample(0, 1, -2),
        ),
    ]
    _write_geneactiv(input_file, gains, offsets, blocks)

    geneactiv_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    assert data.dtype.names == ("time", "x", "y", "z", "temperature")
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        [
            datetime(2024, 1, 2, 3, 4, 5, 600000, tzinfo=timezone.utc).timestamp()
            * 1_000_000_000,
            datetime(2024, 1, 2, 3, 4, 5, 850000, tzinfo=timezone.utc).timestamp()
            * 1_000_000_000,
            datetime(2024, 1, 2, 3, 4, 6, 100000, tzinfo=timezone.utc).timestamp()
            * 1_000_000_000,
        ],
    )
    np.testing.assert_allclose(
        np.column_stack((data["x"], data["y"], data["z"])),
        [
            [1.0, -1.4, 1.2],
            [-1024.0, 819.0, -0.3],
            [-0.5, 0.6, -0.55],
        ],
    )
    np.testing.assert_array_equal(data["temperature"], [21.5, 21.5, 19.25])

    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


def test_geneactiv_reader_skips_rest_of_corrupt_page(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "corrupt-page.bin"
    output_dir = tmp_path / "output"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            2,
            _encode_sample(1, 2, 3) + "ZZZ000000000" + _encode_sample(7, 8, 9),
        ),
        _geneactiv_block(
            1,
            "2024-01-02 03:04:06:000",
            21,
            2,
            _encode_sample(4, 5, 6),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    geneactiv_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(data["x"], [1.0, 4.0])
    np.testing.assert_array_equal(data["y"], [2.0, 5.0])
    np.testing.assert_array_equal(data["z"], [3.0, 6.0])
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "1",
        "SampleRate": "2.0",
    }


def test_geneactiv_reader_decodes_most_negative_12_bit_value(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "signed-boundary.bin"
    output_dir = tmp_path / "output"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            4,
            _encode_sample(-2048, 0, 2047),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    geneactiv_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(
        np.column_stack((data["x"], data["y"], data["z"])),
        [[-2048.0, 0.0, 2047.0]],
    )
    assert _read_info(output_dir) == {
        "ReadOK": "1",
        "ReadErrors": "0",
        "SampleRate": "4.0",
    }


def test_geneactiv_reader_preserves_sample_stride_with_auxiliary_bits(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "auxiliary-bits.bin"
    output_dir = tmp_path / "output"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            4,
            _encode_sample(1, 2, 3, auxiliary=0xFFE)
            + _encode_sample(4, 5, 6, auxiliary=0x004),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    geneactiv_reader(input_file, output_dir)

    data = np.load(output_dir / "data.npy")
    np.testing.assert_array_equal(data["x"], [1.0, 4.0])
    np.testing.assert_array_equal(data["y"], [2.0, 5.0])
    np.testing.assert_array_equal(data["z"], [3.0, 6.0])
    np.testing.assert_array_equal(
        data["time"],
        np.array(
            ["2024-01-02T03:04:05.000", "2024-01-02T03:04:05.250"],
            dtype="datetime64[ns]",
        ),
    )
