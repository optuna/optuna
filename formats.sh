#!/bin/bash
# As described in `CONTRIBUTING.md`, this script checks and formats Optuna's source codes by
# `ruff`. If you pass `-n` as an option, this script checks codes without updating codebase.


missing_dependencies=()
command -v ruff &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(ruff)
fi
command -v mypy &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(mypy)
fi
if [ ! ${#missing_dependencies[@]} -eq 0 ]; then
  echo "The following dependencies are missing:" "${missing_dependencies[@]}"
  read -p "Would you like to install the missing dependencies? (y/N): " yn
  case "$yn" in [yY]*) ;; *) echo "abort." ; exit ;; esac
  pip install "${missing_dependencies[@]}"
fi

update=1
while getopts "n" OPT
do
  case $OPT in
    n) update=0
       ;;
    *) ;;
  esac
done

target="optuna tests benchmarks tutorial"
mypy_target="optuna tests benchmarks"
res_all=0

res_ruff=$(ruff check $target 2>&1 & ruff format $target --check --diff 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "ruff failed. The code will be formatted by ruff."
    ruff check $target --fix
    ruff format $target
  else
    echo "$res_ruff"
    echo "ruff failed."
    res_all=1
  fi
else
  echo "ruff succeeded."
fi

res_mypy=$(mypy $mypy_target)
if [ $? -eq 1 ] ; then
  echo "$res_mypy"
  echo "mypy failed."
  res_all=1
else
  echo "mypy succeeded."
fi

if [ $res_all -eq 1 ] ; then
  exit 1
fi
