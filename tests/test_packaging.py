import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).parents[1]
FIXTURE = (
    PROJECT_ROOT / "tests" / "data" / "parser-fixtures" / "axivity-ax3.cwa"
)


def _copy_clean_source(destination):
    destination.mkdir()
    for name in (
        "LICENSE.md",
        "MANIFEST.in",
        "README.md",
        "pyproject.toml",
        "setup.py",
        "versioneer.py",
    ):
        shutil.copy2(PROJECT_ROOT / name, destination / name)
    shutil.copytree(
        PROJECT_ROOT / "src",
        destination / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )


def test_clean_sdist_and_wheel_are_pure_python(tmp_path):
    checkout = tmp_path / "checkout"
    _copy_clean_source(checkout)

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
    assert any(name.endswith("/src/actipy/axivity_reader.py") for name in names)
    assert not any(name.endswith((".java", ".class")) for name in names)

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
    assert wheel.name.endswith("-py3-none-any.whl")
    with ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert "actipy/axivity_reader.py" in wheel_names
    assert not any(name.endswith((".java", ".class")) for name in wheel_names)

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
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, sys; "
                "from actipy import reader; "
                "assert pathlib.Path(sys.argv[2]).resolve() in "
                "pathlib.Path(reader.__file__).resolve().parents; "
                "arrays, info = reader._read_device_columns(sys.argv[1]); "
                "assert len(arrays['time']) > 0; "
                "assert info['ReadOK'] == 1"
            ),
            str(FIXTURE),
            str(installed),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
