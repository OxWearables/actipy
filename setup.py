import codecs
import os.path
import sys

# https://github.com/python-versioneer/python-versioneer/issues/193
sys.path.insert(0, os.path.dirname(__file__))

import setuptools

import versioneer
from build_java import compile_java

cmdclass = versioneer.get_cmdclass()
versioneer_build_py = cmdclass["build_py"]


class build_py(versioneer_build_py):
    """Build Python modules and compile the bundled Java readers."""

    def run(self):
        super().run()
        source_dir = os.path.join("src", "actipy")
        target_dir = (
            source_dir
            if getattr(self, "editable_mode", False)
            else os.path.join(self.build_lib, "actipy")
        )
        compile_java(source_dir, target_dir)


cmdclass["build_py"] = build_py


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="src/actipy/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="actipy",
    python_requires=">=3.8",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="Python package to process wearable accelerometer data",
    keywords="wearable accelerometer data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OxWearables/actipy",
    download_url="https://github.com/OxWearables/actipy",
    author=get_string("__author__"),
    maintainer=get_string("__maintainer__"),
    maintainer_email=get_string("__maintainer_email__"),
    license=get_string("__license__"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    packages=setuptools.find_packages(where="src", exclude=("test", "tests")),
    package_dir={"": "src"},
    package_data={"actipy": ["py.typed", "*.class"]},
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.7",
        "pandas>=1.3",
        "statsmodels>=0.13",
        "tqdm"
    ],
    extras_require={
        "dev": [
            "ruff",
            "autopep8",
            "ipython",
            "ipdb",
            "twine",
            "tomli",
            "pytest",
            "joblib",
        ] + (["memray"] if not sys.platform.startswith("win") else []),  # memray not supported on Windows
        "docs": [
            "sphinx>=4.2",
            "sphinx_rtd_theme>=1.0",
            "readthedocs-sphinx-search>=0.1",
            "docutils<0.18",
        ],
    },
    entry_points={
        "console_scripts": [
            "read_cwa=actipy.read_cwa:main",
        ]
    }
)
