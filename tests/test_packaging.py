import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE = (
    PROJECT_ROOT / "tests" / "data" / "parser-fixtures" / "axivity-ax3.cwa"
)


def _require_java_toolchain():
    missing = [
        command for command in ("java", "javac")
        if shutil.which(command) is None
    ]
    if missing:
        pytest.skip("Java toolchain unavailable: missing " + ", ".join(missing))


def _copy_clean_source(destination):
    destination.mkdir()
    for name in (
        "LICENSE.md",
        "MANIFEST.in",
        "README.md",
        "build_java.py",
        "pyproject.toml",
        "setup.py",
        "versioneer.py",
    ):
        shutil.copy2(PROJECT_ROOT / name, destination / name)
    shutil.copytree(
        PROJECT_ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("*.class", "__pycache__"),
    )


def test_clean_sdist_and_editable_builds_compile_java_readers(tmp_path):
    _require_java_toolchain()
    checkout = tmp_path / "checkout"
    _copy_clean_source(checkout)
    assert not list(checkout.glob("src/actipy/*.class"))

    editable_dist = tmp_path / "editable-dist"
    editable_dist.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(editable_dist),
            "--editable",
            str(checkout),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (checkout / "src" / "actipy" / "AxivityReader.class").exists()

    sdist_dir = tmp_path / "sdist"
    subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(sdist_dir)],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    sdist = next(sdist_dir.glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    assert any(name.endswith("/build_java.py") for name in names)
    assert any(name.endswith("/src/actipy/AxivityReader.java") for name in names)
    assert not any(name.endswith(".class") for name in names)

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(sdist),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    with ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert "actipy/AxivityReader.class" in wheel_names
    assert "actipy/ReaderSupport.class" in wheel_names

    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    output_dir = tmp_path / "reader-output"
    output_dir.mkdir()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, sys; "
                "from actipy import reader; "
                "assert pathlib.Path(sys.argv[3]).resolve() in "
                "pathlib.Path(reader.__file__).resolve().parents; "
                "arrays, info = reader._java_read_device_stream("
                "sys.argv[1], sys.argv[2], verbose=False); "
                "assert len(arrays['time']) > 0; "
                "assert info['ReadOK'] == 1"
            ),
            str(FIXTURE),
            str(output_dir),
            str(installed),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
