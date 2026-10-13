import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

import actipy
from actipy import axivity_reader

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "data" / "parser-fixtures"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())
SUSTAINED_FIXTURES = {
    "actigraph-v1.gt3x",
    "actigraph-v1-middle.gt3x",
    "actigraph-v1-end.gt3x",
    "actigraph-leap-v2.gt3x",
    "actigraph-leap-v2-middle.gt3x",
    "actigraph-leap-v2-end.gt3x",
    "axivity-ax3.cwa",
    "axivity-ax3-middle.cwa",
    "axivity-ax3-end.cwa",
    "geneactiv.bin",
    "geneactiv-middle.bin",
    "geneactiv-end.bin",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_real_device_fixtures_cover_sustained_windows():
    window_seconds = MANIFEST["window_seconds"]
    fixture_specs = {
        spec["fixture"]: spec
        for spec in MANIFEST["fixtures"]
        if spec["fixture"] in SUSTAINED_FIXTURES
    }

    assert window_seconds == 10 * 60
    assert set(fixture_specs) == SUSTAINED_FIXTURES
    for spec in fixture_specs.values():
        assert spec["rows"] >= spec["sample_rate"] * window_seconds


@pytest.mark.parametrize(
    "fixture_spec",
    MANIFEST["fixtures"],
    ids=lambda spec: spec["fixture"],
)
def test_real_device_numerical_regression(fixture_spec):
    fixture_path = FIXTURE_DIR / fixture_spec["fixture"]
    expected_path = FIXTURE_DIR / fixture_spec["expected"]

    assert _sha256(fixture_path) == fixture_spec["fixture_sha256"]
    assert _sha256(expected_path) == fixture_spec["expected_sha256"]
    data, info = actipy.read_device(
        str(fixture_path),
        lowpass_hz=None,
        calibrate_gravity=False,
        detect_nonwear=False,
        resample_hz=None,
        verbose=False,
    )

    with np.load(expected_path) as expected:
        assert list(data.columns) == [
            field for field in fixture_spec["fields"] if field != "time"
        ]
        np.testing.assert_array_equal(
            data.index.to_numpy().view("int64"), expected["time"]
        )
        for column in data.columns:
            assert data[column].dtype == np.dtype("float32")
            np.testing.assert_array_equal(
                data[column].to_numpy(), expected[column]
            )

    assert len(data) == fixture_spec["rows"]
    assert info["Device"] == fixture_spec["device"]
    assert info["DeviceID"] == fixture_spec["device_id"]
    assert info["ReadOK"] == fixture_spec["read_ok"]
    assert info["ReadErrors"] == fixture_spec["read_errors"]
    assert info["SampleRate"] == fixture_spec["sample_rate"]


def test_axivity_reader_returns_data_before_truncated_tail(tmp_path, caplog):
    source = FIXTURE_DIR / "axivity-ax3.cwa"
    input_file = tmp_path / "truncated-tail.cwa"
    source_bytes = source.read_bytes()
    data_block = next(
        source_bytes[offset:offset + 512]
        for offset in range(0, len(source_bytes), 512)
        if source_bytes[offset:offset + 2] == b"AX"
    )
    input_file.write_bytes(data_block + b"AX")

    with caplog.at_level(logging.WARNING, logger="actipy"):
        data, info = axivity_reader.read(str(input_file))

    assert "Unexpected partial CWA block" in caplog.text
    assert len(data["time"]) > 0
    assert info["ReadOK"] == 1
    assert info["ReadErrors"] == 1
