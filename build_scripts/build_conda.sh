#!/bin/bash

# Be sure to have packaged for PyPI first
# conda install conda-build
conda skeleton pypi actipy --output-dir conda-recipe
conda build -c conda-forge conda-recipe/actipy
# anaconda login
# anaconda upload --user oxwear /path/to/package.tar.bz2
