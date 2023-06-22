#!/bin/sh

# DEVELOPER INSTALLATION SCRIPT

# You must have Java Development Kit (JDK) 8 (1.8). If higher (>8) then it must
# support --release flag to pin down the version when compiling.
# Always compile with 8 (1.8) to keep backward compatibility.
# In conda, you can get a JDK version that supports --release flag:
# conda install openjdk
javac --version &&  # java version
javac --release 8 src/actipy/*.java &&  # compile java files (using release 8)
pip install -e .[dev,docs] # install in edit mode

# Download tiny sample data used for unit tests
wget -P tests/data/ https://wearables-files.ndph.ox.ac.uk/files/data/samples/ax3/tiny-sample.cwa.gz
