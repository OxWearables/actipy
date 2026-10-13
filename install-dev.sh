#!/bin/sh

# DEVELOPER INSTALLATION SCRIPT

pip install -e .[dev,docs] # install in edit mode

# Download tiny sample data used for unit tests
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/data/samples/ax3/tiny-sample.cwa.gz
# Copies for multiprocessing unit tests
cp tests/data/tiny-sample.cwa.gz tests/data/tiny-sample1.cwa.gz
cp tests/data/tiny-sample.cwa.gz tests/data/tiny-sample2.cwa.gz
# Test data for unit tests
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/actipy/tests/data/read.csv.gz
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/actipy/tests/data/lowpass.csv.gz
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/actipy/tests/data/calib.csv.gz
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/actipy/tests/data/nonwear.csv.gz
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/actipy/tests/data/resample.csv.gz
