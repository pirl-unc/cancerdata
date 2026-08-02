#!/bin/bash
set -o errexit

./lint.sh
python -m pytest tests/ -n 2 --dist loadgroup --durations 20 --durations-min 0.5 --cov=oncoref --cov-report=term-missing
