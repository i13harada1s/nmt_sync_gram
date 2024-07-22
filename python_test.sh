#!/bin/sh

# Use your python3 interpreter.
python='poetry run python'

for file in `find ./src -name "*_test.py" | sort`; do
  # Remove './' prefix, remove '.py' suffix and replace all '/' to '.'.
  import=`echo ${file} | sed -e 's/^\.\///' -e 's/\.py$//' -e 's/\//./g'`

  # Run a test by import.
  echo ${import}
done | xargs -n 1 -P 8 ${python} -m
