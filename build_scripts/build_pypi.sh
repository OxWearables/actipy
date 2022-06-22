#!/bin/bash

javac --release 8 actipy/*.java  # compile java stuff; use a low version e.g. 8 (1.8)
python setup.py sdist bdist_wheel
twine check dist/*
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*
