#!/bin/bash
set -o errexit

./lint.sh
python -m pytest tests/ -n 2 --dist load --cov=oncoref --cov-report=term-missing
