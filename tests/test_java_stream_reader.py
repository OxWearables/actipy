import io
import struct
import subprocess

import numpy as np
import pytest

import actipy.reader as reader_module


class _FailedJavaProcess:
    def __init__(self, return_code, output=b"", poll_result=0):
        self.stdout = (
            None if output is None
            else io.BufferedReader(io.BytesIO(output))
        )
        self.return_code = return_code
        self.poll_result = poll_result
        self.killed = False
        self.wait_count = 0
        self.poll_count = 0

    def wait(self):
        self.wait_count += 1
        return self.return_code

    def kill(self):
        self.killed = True

    def poll(self):
        self.poll_count += 1
        return self.poll_result


def _encoded_stream(schema_code, chunk_sizes):
    fields = reader_module._JAVA_STREAM_SCHEMAS[schema_code]
    rows = sum(chunk_sizes)
    expected = {}
    for column, field in enumerate(fields):
        if field == "time":
            expected[field] = np.arange(rows, dtype="i8").view(
                "datetime64[ns]"
            )
        else:
            expected[field] = np.arange(rows, dtype="f4") + column

    payload = bytearray(reader_module._JAVA_STREAM_MAGIC)
    payload.append(schema_code)
    offset = 0
    for chunk_rows in chunk_sizes:
        payload.extend(struct.pack("<I", chunk_rows))
        chunk_end = offset + chunk_rows
        for field in fields:
            payload.extend(expected[field][offset:chunk_end].tobytes())
        offset = chunk_end
    payload.extend(struct.pack("<I", 0))
    return io.BufferedReader(io.BytesIO(payload)), expected


def test_java_stream_reports_early_parser_failure(monkeypatch, tmp_path):
    process = _FailedJavaProcess(return_code=7)
    monkeypatch.setattr(
        reader_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        reader_module._java_read_device_stream(
            str(tmp_path / "invalid.cwa"), str(tmp_path)
        )

    assert error.value.returncode == 7
    assert process.stdout.closed
    assert process.wait_count == 1
    assert not process.killed


@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (b"", EOFError, "ended unexpectedly"),
        (b"BADMAGIC", ValueError, "Invalid Java parser stream header"),
        (
            reader_module._JAVA_STREAM_MAGIC,
            EOFError,
            "ended unexpectedly",
        ),
        (
            reader_module._JAVA_STREAM_MAGIC + b"\xff",
            ValueError,
            "Unsupported Java parser stream schema",
        ),
        (
            reader_module._JAVA_STREAM_MAGIC + b"\x03\x01\x00",
            EOFError,
            "ended unexpectedly",
        ),
        (
            reader_module._JAVA_STREAM_MAGIC
            + b"\x03"
            + struct.pack("<I", 8193),
            ValueError,
            "Invalid Java parser stream chunk size",
        ),
        (
            reader_module._JAVA_STREAM_MAGIC
            + b"\x03"
            + struct.pack("<I", 1)
            + b"\x00" * 10,
            EOFError,
            "ended unexpectedly",
        ),
    ],
)
def test_java_stream_rejects_malformed_frames(payload, error, message):
    stream = io.BufferedReader(io.BytesIO(payload))

    with pytest.raises(error, match=message):
        reader_module._read_java_stream_arrays(stream)


def test_java_stream_rejects_rows_beyond_addressable_limit(monkeypatch):
    stream, _ = _encoded_stream(schema_code=3, chunk_sizes=[2])
    row_size = 8 + 5 * 4
    limit = type("AddressLimit", (), {"max": row_size})()
    monkeypatch.setattr(reader_module.np, "iinfo", lambda dtype: limit)

    with pytest.raises(ValueError, match="stream is too large"):
        reader_module._read_java_stream_arrays(stream)


@pytest.mark.parametrize(
    ("poll_result", "expected_kill"),
    [(None, True), (4, False)],
)
def test_java_stream_protocol_failure_closes_and_reaps_process(
        monkeypatch, tmp_path, poll_result, expected_kill):
    process = _FailedJavaProcess(
        return_code=4,
        output=b"BADMAGIC",
        poll_result=poll_result,
    )
    monkeypatch.setattr(
        reader_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(ValueError, match="Invalid Java parser stream header"):
        reader_module._java_read_device_stream(
            str(tmp_path / "invalid.cwa"), str(tmp_path)
        )

    assert process.stdout.closed
    assert process.poll_count == 1
    assert process.killed is expected_kill
    assert process.wait_count == 1


def test_java_stream_checks_exit_after_valid_framing(monkeypatch, tmp_path):
    stream, _ = _encoded_stream(schema_code=3, chunk_sizes=[])
    process = _FailedJavaProcess(return_code=9, output=stream.read())
    monkeypatch.setattr(
        reader_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        reader_module._java_read_device_stream(
            str(tmp_path / "invalid.cwa"), str(tmp_path)
        )

    assert error.value.returncode == 9
    assert process.stdout.closed
    assert process.wait_count == 1
    assert not process.killed


def test_java_stream_reaps_process_when_stdout_is_unavailable(
        monkeypatch, tmp_path):
    process = _FailedJavaProcess(return_code=1, output=None)
    monkeypatch.setattr(
        reader_module.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )

    with pytest.raises(RuntimeError, match="Could not open"):
        reader_module._java_read_device_stream(
            str(tmp_path / "invalid.cwa"), str(tmp_path)
        )

    assert process.killed
    assert process.wait_count == 1


@pytest.mark.parametrize("chunk_sizes", [[], [1], [8192, 1]])
def test_java_stream_grows_from_observed_rows_with_exact_ownership(
        chunk_sizes):
    stream, expected = _encoded_stream(schema_code=3, chunk_sizes=chunk_sizes)

    arrays = reader_module._read_java_stream_arrays(stream)

    assert arrays.keys() == expected.keys()
    for field, values in arrays.items():
        np.testing.assert_array_equal(values, expected[field])
        assert values.base is None
        assert values.flags.owndata
        assert values.nbytes == expected[field].nbytes
