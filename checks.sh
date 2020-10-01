#!/bin/bash

# As described in `CONTRIBUTING.md`, this script checks Optuna's source codes for formatting,
# coding style and type hints. The `-u` option enables automatic formatting by `black` and `isort`.

git update-index --chmod=+x

res_pip_list=$(pip freeze)
missing_dependencies=()
if [ ! "$(echo $res_pip_list | grep black)" ] ; then
  missing_dependencies+=(black)
fi
if [ ! "$(echo $res_pip_list | grep flake8)" ] ; then
  missing_dependencies+=(flake8)
fi
if [ ! "$(echo $res_pip_list | grep isort)" ] ; then
  missing_dependencies+=(isort)
fi
if [ ! "$(echo $res_pip_list | grep mypy)" ] ; then
  missing_dependencies+=(mypy)
fi
if [ ! ${#missing_dependencies[@]} -eq 0 ]; then
  echo "The following dependencies are missed:" "${missing_dependencies[@]}"
  read -p "Would you like to install the missing dependencies? (y/N): " yn
  case "$yn" in [yY]*) ;; *) echo "abort." ; exit ;; esac
  pip install "${missing_dependencies[@]}"
fi

update=0
while getopts "u" OPT
do
  case $OPT in
    u) update=1
       ;;
    *) ;;
  esac
done

black_target="examples optuna tests"
res_black=$(black $black_target --check --diff 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "black failed. The code will be formatted by black."
    black .
  else
    echo "$res_black"
    echo "black failed."
    exit 1
  fi
else
  echo "black succeeded."
fi

res_flake8=$(flake8 .)
if [ $? -eq 1 ] ; then
  echo "$res_flake8"
  echo "flake8 failed."
  exit 1
fi
echo "flake8 succeeded."

res_isort=$(isort . --check 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "isort failed. The code will be formatted by isort."
    isort .
  else
    echo "$res_isort"
    echo "isort failed."
    exit 1
  fi
else
  echo "isort succeeded."
fi

res_mypy=$(mypy .)
if [ $? -eq 1 ] ; then
  echo "$res_mypy"
  echo "mypy failed."
  exit 1
fi
echo "mypy succeeded."
