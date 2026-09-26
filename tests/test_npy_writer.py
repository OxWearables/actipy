import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "actipy"
NPY_WRITER_SOURCE = SOURCE_DIR / "NpyWriter.java"
HARNESS_SOURCE = PROJECT_ROOT / "tests" / "java" / "NpyWriterHarness.java"
BASE_TIME = 1_700_000_000_000_000_000
FIELD_NAMES = {
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "temperature"),
    5: ("x", "y", "z", "temperature", "light"),
    8: (
        "x", "y", "z", "gyro_x", "gyro_y", "gyro_z",
        "temperature", "light",
    ),
}


def require_java_toolchain(java, javac):
    missing = [
        command
        for command, path in (("java", java), ("javac", javac))
        if path is None
    ]
    if missing:
        pytest.skip("Java toolchain unavailable: missing " + ", ".join(missing))
    return java, javac


@pytest.fixture(scope="session")
def npy_writer_classes(tmp_path_factory):
    java, javac = require_java_toolchain(
        shutil.which("java"),
        shutil.which("javac"),
    )
    classes = tmp_path_factory.mktemp("npy-writer-classes")
    subprocess.run(
        [
            javac,
            "-d",
            str(classes),
            str(NPY_WRITER_SOURCE),
            str(HARNESS_SOURCE),
        ],
        check=True,
    )
    return classes, java


def run_harness(npy_writer_classes, output, mode, float_columns, rows, check=True):
    classes, java = npy_writer_classes
    return subprocess.run(
        [
            java,
            "-cp",
            str(classes),
            "NpyWriterHarness",
            str(output),
            mode,
            str(float_columns),
            str(rows),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def assert_output(output, field_names, rows):
    data = np.load(output)

    assert data.dtype.names == ("time", *field_names)
    assert data.dtype["time"] == np.dtype("datetime64[ns]")
    for field_name in field_names:
        assert data.dtype[field_name] == np.dtype("float32")

    assert data.shape == (rows,)
    np.testing.assert_array_equal(
        data["time"].view("int64"),
        BASE_TIME + np.arange(rows, dtype=np.int64),
    )
    for column, field_name in enumerate(field_names):
        expected = (
            np.arange(1, rows + 1, dtype=np.float32) * (column + 1) / 8.0
        )
        np.testing.assert_array_equal(data[field_name], expected)


def assert_exact_file_size(output, float_columns, rows):
    header_bytes = 6 + 2 + 2 + 256
    row_bytes = 8 + 4 * float_columns
    assert output.stat().st_size == header_bytes + row_bytes * rows


@pytest.mark.parametrize("float_columns", [3, 4, 5, 8])
@pytest.mark.parametrize("rows", [0, 1, 8191, 8192, 8193])
def test_primitive_rows_across_buffer_boundary(
        npy_writer_classes, tmp_path, float_columns, rows):
    output = tmp_path / "primitive.npy"
    run_harness(
        npy_writer_classes,
        output,
        "primitive",
        float_columns,
        rows,
    )
    assert_output(output, FIELD_NAMES[float_columns], rows)
    assert_exact_file_size(output, float_columns, rows)


def test_map_api_remains_compatible(npy_writer_classes, tmp_path):
    output = tmp_path / "map.npy"
    run_harness(npy_writer_classes, output, "map", 4, 3)
    assert_output(output, ("f0", "f1", "f2", "f3"), rows=3)


def test_failed_map_row_does_not_corrupt_following_rows(
        npy_writer_classes, tmp_path):
    output = tmp_path / "map-recover.npy"
    run_harness(npy_writer_classes, output, "map-recover", 4, 3)
    assert_output(output, ("f0", "f1", "f2", "f3"), rows=3)
    assert_exact_file_size(output, 4, 3)


@pytest.mark.parametrize("float_columns", [3, 4, 5, 8])
@pytest.mark.parametrize(
    "mode",
    ["wrong-order", "wrong-leading-type", "wrong-trailing-type", "wrong-arity"],
)
def test_primitive_rows_reject_incompatible_schema(
        npy_writer_classes, tmp_path, float_columns, mode):
    output = tmp_path / f"{mode}.npy"
    result = run_harness(
        npy_writer_classes,
        output,
        mode,
        float_columns,
        rows=1,
        check=False,
    )

    assert result.returncode != 0
    assert "NpyWriter$SchemaMismatchException" in result.stderr


@pytest.mark.parametrize("float_columns", [3, 4, 5, 8])
def test_schema_is_snapshotted_at_construction(
        npy_writer_classes, tmp_path, float_columns):
    output = tmp_path / "mutated-schema.npy"
    run_harness(
        npy_writer_classes,
        output,
        "mutate-schema",
        float_columns,
        rows=3,
    )
    assert_output(output, FIELD_NAMES[float_columns], rows=3)


def test_close_is_idempotent(npy_writer_classes, tmp_path):
    output = tmp_path / "double-close.npy"
    run_harness(npy_writer_classes, output, "double-close", 3, 2)
    assert_output(output, FIELD_NAMES[3], rows=2)
    assert_exact_file_size(output, 3, 2)


def test_write_after_close_is_rejected(npy_writer_classes, tmp_path):
    output = tmp_path / "write-after-close.npy"
    result = run_harness(
        npy_writer_classes,
        output,
        "write-after-close",
        3,
        rows=1,
        check=False,
    )
    assert result.returncode != 0
    assert "Cannot write to a closed NpyWriter" in result.stderr


def test_close_failure_is_propagated(npy_writer_classes, tmp_path):
    output = tmp_path / "close-failure.npy"
    result = run_harness(
        npy_writer_classes,
        output,
        "close-failure",
        3,
        rows=1,
        check=False,
    )
    assert result.returncode != 0
    assert "UncheckedIOException" in result.stderr
    assert "Could not finalize" in result.stderr


@pytest.mark.parametrize("mode", ["compress-failure", "compress-source"])
def test_failed_compression_preserves_writer_state(
        npy_writer_classes, tmp_path, mode):
    output = tmp_path / f"{mode}.npy"

    run_harness(npy_writer_classes, output, mode, 3, rows=2)

    assert_output(output, FIELD_NAMES[3], rows=3)
    assert_exact_file_size(output, 3, rows=3)


@pytest.mark.parametrize(
    ("java", "javac", "missing"),
    [
        (None, "/usr/bin/javac", "java"),
        ("/usr/bin/java", None, "javac"),
        (None, None, "java, javac"),
    ],
)
def test_missing_java_toolchain_skips(java, javac, missing):
    with pytest.raises(
        pytest.skip.Exception,
        match=f"Java toolchain unavailable: missing {missing}",
    ):
        require_java_toolchain(java, javac)
