"""Build helpers for the Java device readers shipped with actipy."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]


def javac_target_args(version_output: str) -> List[str]:
    """Return compiler flags that produce Java 8-compatible bytecode."""

    match = re.search(r"\bjavac\s+(\d+)(?:\.(\d+))?", version_output)
    if match is None:
        raise RuntimeError(
            f"Could not determine the javac version from: {version_output!r}"
        )

    first = int(match.group(1))
    second = int(match.group(2) or 0)
    major = second if first == 1 else first
    if major < 8:
        raise RuntimeError("actipy requires javac 8 or newer")
    if major == 8:
        return []
    return ["--release", "8"]


def compile_java(source_dir: PathLike, target_dir: PathLike) -> None:
    """Compile all reader sources into a clean target directory."""

    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()
    sources = sorted(source_path.glob("*.java"))
    if not sources:
        raise RuntimeError(f"No Java sources found in {source_path}")

    javac = shutil.which("javac")
    if javac is None:
        raise RuntimeError(
            "javac is required to build actipy's device readers"
        )

    version = subprocess.run(
        [javac, "-version"],
        check=True,
        capture_output=True,
        text=True,
    )
    flags = javac_target_args(version.stdout + version.stderr)

    target_path.mkdir(parents=True, exist_ok=True)
    for bytecode in target_path.glob("*.class"):
        bytecode.unlink()

    subprocess.run(
        [
            javac,
            *flags,
            "-d",
            str(target_path),
            *(str(source) for source in sources),
        ],
        check=True,
    )
