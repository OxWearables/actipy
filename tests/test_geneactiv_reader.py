import logging
import warnings
from datetime import datetime, timezone

import numpy as np
import pytest

from actipy import geneactiv_reader as geneactiv
from actipy._reader_support import FormatError


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


def _full_page_payload(*samples):
    assert len(samples) <= 300
    return "".join(samples) + _encode_sample(0, 0, 0) * (300 - len(samples))


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


def _write_geneactiv(
        path, gains, offsets, blocks, declared_page_count=None):
    page_count = (
        len(blocks) if declared_page_count is None else declared_page_count
    )
    lines = _geneactiv_header(gains, offsets, page_count)
    for block in blocks:
        lines.extend(block)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


@pytest.fixture
def geneactiv_reader():
    def run(input_file):
        return geneactiv.read(str(input_file))

    return run


def test_geneactiv_reader_decodes_samples_and_page_metadata(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "sample.bin"
    gains = (200, 250, 400)
    offsets = (100, -50, 20)
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:600",
            21.5,
            4,
            _full_page_payload(
                _encode_sample(3, -4, 5),
                _encode_sample(-2047, 2047, -1),
            ),
        ),
        _geneactiv_block(
            1,
            "2024-01-02 03:04:06:100",
            19.25,
            4,
            _full_page_payload(_encode_sample(0, 1, -2)),
        ),
    ]
    _write_geneactiv(input_file, gains, offsets, blocks)

    data, info = geneactiv_reader(input_file)
    assert tuple(data) == ("time", "x", "y", "z", "temperature")
    assert data["time"].shape == (600,)
    selected = {name: values[[0, 1, 300]] for name, values in data.items()}
    np.testing.assert_array_equal(
        selected["time"].view("int64"),
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
        np.column_stack((selected["x"], selected["y"], selected["z"])),
        [
            [1.0, -1.4, 1.2],
            [-1024.0, 819.0, -0.3],
            [-0.5, 0.6, -0.55],
        ],
    )
    np.testing.assert_array_equal(
        selected["temperature"], [21.5, 21.5, 19.25])

    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


@pytest.mark.parametrize(
    "corrupt_sample",
    ["ZZZ000000000", "000000000Z00"],
    ids=["axis", "auxiliary"],
)
def test_geneactiv_reader_skips_entire_corrupt_page(
        geneactiv_reader, tmp_path, corrupt_sample):
    input_file = tmp_path / "corrupt-page.bin"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            2,
            _full_page_payload(
                _encode_sample(1, 2, 3),
                corrupt_sample,
                _encode_sample(7, 8, 9),
            ),
        ),
        _geneactiv_block(
            1,
            "2024-01-02 03:04:06:000",
            21,
            2,
            _full_page_payload(_encode_sample(4, 5, 6)),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    data, info = geneactiv_reader(input_file)
    assert data["time"].shape == (300,)
    np.testing.assert_array_equal(
        np.array([data["x"][0], data["y"][0], data["z"][0]]),
        [4.0, 5.0, 6.0],
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 2.0,
    }


def test_geneactiv_reader_decodes_most_negative_12_bit_value(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "signed-boundary.bin"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            4,
            _full_page_payload(_encode_sample(-2048, 0, 2047)),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    data, info = geneactiv_reader(input_file)
    np.testing.assert_array_equal(
        np.column_stack((data["x"][:1], data["y"][:1], data["z"][:1])),
        [[-2048.0, 0.0, 2047.0]],
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 0,
        "SampleRate": 4.0,
    }


def test_geneactiv_reader_preserves_sample_stride_with_lowercase_auxiliary_bits(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "auxiliary-bits.bin"
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            4,
            _full_page_payload(
                _encode_sample(1, 2, 3, auxiliary=0xFFE),
                _encode_sample(4, 5, 6, auxiliary=0x004),
            ).lower(),
        ),
    ]
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), blocks)

    data, info = geneactiv_reader(input_file)
    np.testing.assert_array_equal(data["x"][:2], [1.0, 4.0])
    np.testing.assert_array_equal(data["y"][:2], [2.0, 5.0])
    np.testing.assert_array_equal(data["z"][:2], [3.0, 6.0])
    np.testing.assert_array_equal(
        data["time"][:2],
        np.array(
            ["2024-01-02T03:04:05.000", "2024-01-02T03:04:05.250"],
            dtype="datetime64[ns]",
        ),
    )


@pytest.mark.parametrize(
    ("field_index", "replacement"),
    [
        (3, "Page Time:not-a-time"),
        (5, "Temperature:not-a-number"),
        # Java parses an overflowing hexadecimal value as +-Infinity, which is not a finite temperature
        (5, "Temperature:0x1p1024"),
        (5, "Temperature:-0x1p1024"),
        (8, "Measurement Frequency:0"),
    ],
)
def test_geneactiv_reader_skips_page_with_invalid_header_without_reusing_state(
        geneactiv_reader, tmp_path, field_index, replacement):
    input_file = tmp_path / "invalid-page-header.bin"
    first = _geneactiv_block(
        0,
        "2024-01-02 03:04:05:000",
        20,
        2,
        _full_page_payload(_encode_sample(1, 2, 3)),
    )
    second = _geneactiv_block(
        1,
        "2024-01-02 03:04:06:000",
        21,
        4,
        _full_page_payload(_encode_sample(4, 5, 6)),
    )
    first[field_index] = replacement
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), [first, second])

    data, info = geneactiv_reader(input_file)
    assert data["time"].shape == (300,)
    np.testing.assert_array_equal(data["x"][:1], [4.0])
    np.testing.assert_array_equal(data["temperature"][:1], [21.0])
    np.testing.assert_array_equal(
        data["time"][:1],
        np.array(["2024-01-02T03:04:06.000"], dtype="datetime64[ns]"),
    )
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 4.0,
    }


def test_geneactiv_reader_saturates_timestamps_like_java_at_extreme_sample_rate(
        geneactiv_reader, tmp_path):
    # At 1e-16 Hz every sample after the first lies beyond the long range of milliseconds:
    # Java's (long) cast saturates and toNanos keeps Long.MAX_VALUE (the Java reader's output).
    input_file = tmp_path / "extreme-rate.bin"
    block = _geneactiv_block(
        0, "2024-01-02 03:04:05:000", 20, "1e-16", _full_page_payload(_encode_sample(1, 2, 3))
    )
    _write_geneactiv(input_file, (100, 100, 100), (0, 0, 0), [block])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data, info = geneactiv_reader(input_file)

    times = data["time"].view("int64")
    assert times[0] == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc).timestamp() * 1_000_000_000
    assert (times[1:] == 2**63 - 1).all()
    assert info["ReadErrors"] == 0


def test_geneactiv_reader_calibration_does_not_signal_for_codes_absent_from_file(
        geneactiv_reader, tmp_path):
    # Calibration is tabulated for all 4,096 codes. With a tiny gain the unused codes overflow
    # float32; Java computes only the samples present (all zero here) and returns zeros without
    # signalling, whatever the caller's warning or floating-point settings.
    input_file = tmp_path / "tiny-gain.bin"
    block = _geneactiv_block(0, "2024-01-02 03:04:05:000", 20, 100, _full_page_payload())
    _write_geneactiv(input_file, ("1e-34", 100, 100), (0, 0, 0), [block])

    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        data, info = geneactiv_reader(input_file)

    for axis in ("x", "y", "z"):
        np.testing.assert_array_equal(data[axis], np.zeros(300, dtype=np.float32))
    assert info["ReadErrors"] == 0


@pytest.mark.parametrize("declared_page_count", [0, 2])
def test_geneactiv_reader_keeps_data_when_page_count_differs_from_header(
        geneactiv_reader, tmp_path, caplog, declared_page_count):
    input_file = tmp_path / "page-count-mismatch.bin"
    payload = _encode_sample(1, 2, 3) * 300
    blocks = [
        _geneactiv_block(
            0,
            "2024-01-02 03:04:05:000",
            20,
            100,
            payload,
        ),
    ]
    _write_geneactiv(
        input_file,
        (100, 100, 100),
        (0, 0, 0),
        blocks,
        declared_page_count=declared_page_count,
    )

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = geneactiv_reader(input_file)

    assert "page count differs from header" in caplog.text
    assert data["time"].shape == (300,)
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 100.0,
    }


def test_geneactiv_reader_keeps_complete_page_before_truncated_page(
        geneactiv_reader, tmp_path, caplog):
    input_file = tmp_path / "truncated-page.bin"
    complete = _geneactiv_block(
        0,
        "2024-01-02 03:04:05:000",
        20,
        100,
        _encode_sample(1, 2, 3) * 300,
    )
    lines = _geneactiv_header((100, 100, 100), (0, 0, 0), 2)
    lines.extend(complete)
    lines.extend(["Recorded Data", "Device Unique Serial Code:012345"])
    input_file.write_text("\n".join(lines) + "\n", encoding="ascii")

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = geneactiv_reader(input_file)

    assert "Stopping at truncated GENEActiv data" in caplog.text
    assert data["time"].shape == (300,)
    assert info == {
        "ReadOK": 1,
        "ReadErrors": 1,
        "SampleRate": 100.0,
    }


def test_geneactiv_reader_rejects_matching_count_with_no_valid_pages(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "no-valid-pages.bin"
    malformed = _geneactiv_block(
        0,
        "not-a-time",
        20,
        100,
        _full_page_payload(_encode_sample(1, 2, 3)),
    )
    _write_geneactiv(
        input_file,
        (100, 100, 100),
        (0, 0, 0),
        [malformed],
    )

    with pytest.raises(FormatError, match="No valid GENEActiv pages were decoded"):
        geneactiv_reader(input_file)


@pytest.mark.parametrize("sample_count", [0, 299])
def test_geneactiv_reader_rejects_whole_sample_page_truncation(
        geneactiv_reader, tmp_path, caplog, sample_count):
    input_file = tmp_path / f"truncated-{sample_count}.bin"
    truncated = _geneactiv_block(
        0,
        "2024-01-02 03:04:05:000",
        20,
        100,
        _encode_sample(1, 2, 3) * sample_count,
    )
    _write_geneactiv(
        input_file,
        (100, 100, 100),
        (0, 0, 0),
        [truncated],
    )

    with caplog.at_level(logging.WARNING, logger="actipy"), \
            pytest.raises(FormatError, match="No valid GENEActiv pages were decoded"):
        geneactiv_reader(input_file)

    assert "expected 3600 hexadecimal characters" in caplog.text


def test_geneactiv_reader_rejects_truncated_file_header(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "truncated-header.bin"
    input_file.write_text("Header line 1\n", encoding="ascii")

    with pytest.raises(FormatError, match="Unexpected end of GENEActiv file header"):
        geneactiv_reader(input_file)


def test_geneactiv_reader_rejects_zero_calibration_gain(
        geneactiv_reader, tmp_path):
    input_file = tmp_path / "zero-gain.bin"
    _write_geneactiv(input_file, (0, 100, 100), (0, 0, 0), [])

    with pytest.raises(FormatError, match="Calibration gain must be finite and non-zero"):
        geneactiv_reader(input_file)
