import setuptools
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_string(string, rel_path="actipy/__init__.py"):
    for line in read(rel_path).splitlines():
        if line.startswith(string):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError(f"Unable to find {string}.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="actipy",
    python_requires=">=3.8",
    version=get_string("__version__"),
    description="Python package to process wearable accelerometer data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/activityMonitoring/actipy",
    author=get_string("__author__"),
    author_email=get_string("__email__"),
    license=get_string("__license__"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Unix",
    ],
    packages=setuptools.find_packages(exclude=("test", "tests")),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.7",
        "pandas>=1.3",
        "statsmodels>=0.13",
        "Jpype1>=1.3",
    ],
    extras_require={
        "dev": [
            "flake8",
            "autopep8",
            "ipython",
            "ipdb",
            "memory-profiler",
            "twine",
        ],
        "docs": [
            "sphinx>=4.2",
            "sphinx_rtd_theme>=1.0",
            "readthedocs-sphinx-search>=0.1",
            "docutils<0.18",
        ]
    }
)
