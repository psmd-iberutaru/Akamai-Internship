@echo off

:: Go out to the main directory for api compilation.
cd ..
sphinx-apidoc -e -f -o docs/source ./Active_Codebase

:: Go back into the doc directory and compile the html.
cd docs
make clean && make html

