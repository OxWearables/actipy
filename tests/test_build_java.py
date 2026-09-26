import importlib.util
import subprocess
from pathlib import Path

import pytest

BUILD_JAVA_PATH = Path(__file__).parents[1] / "build_java.py"
SPEC = importlib.util.spec_from_file_location("build_java", BUILD_JAVA_PATH)
assert SPEC is not None and SPEC.loader is not None
build_java = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_java)


@pytest.mark.parametrize(
    ("version_output", "expected"),
    [
        ("javac 1.8.0_402", []),
        ("javac 8.0.402", []),
        ("javac 11.0.24", ["--release", "8"]),
        ("javac 21", ["--release", "8"]),
    ],
)
def test_javac_target_args(version_output, expected):
    assert build_java.javac_target_args(version_output) == expected


@pytest.mark.parametrize("version_output", ["unknown", "javac 1.7.0"])
def test_javac_target_args_rejects_unsupported_versions(version_output):
    with pytest.raises(RuntimeError):
        build_java.javac_target_args(version_output)


def test_compile_java_cleans_target_and_invokes_compiler(
        monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "Reader.java").write_text("class Reader {}")
    stale_class = target_dir / "Reader$Stale.class"
    stale_class.write_bytes(b"stale")
    calls = []

    monkeypatch.setattr(build_java.shutil, "which", lambda name: "/jdk/javac")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-1] == "-version":
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr="javac 17.0.12"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_java.subprocess, "run", run)

    build_java.compile_java(source_dir, target_dir)

    assert not stale_class.exists()
    assert calls[0][0] == ["/jdk/javac", "-version"]
    assert calls[1][0] == [
        "/jdk/javac",
        "--release",
        "8",
        "-d",
        str(target_dir.resolve()),
        str((source_dir / "Reader.java").resolve()),
    ]
    assert calls[1][1] == {"check": True}


def test_compile_java_requires_compiler(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "Reader.java").write_text("class Reader {}")
    monkeypatch.setattr(build_java.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="javac is required"):
        build_java.compile_java(source_dir, tmp_path / "target")
