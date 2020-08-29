#!/bin/bash

if [[ $((black . --check) 2>&1 | grep unchanged) ]] ; then
  echo "Success in black."
else
  echo "Failed. The code will be formatted by black."
  black .
  exit 1
fi

if [[ $(flake8 .) ]] ; then
  exit 1
fi
echo "Success in flake8."

if [[ ! $((mypy .) 2>&1 | grep Success) ]] ; then
  mypy .
  exit 1
fi
echo "Success in mypy"

