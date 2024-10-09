#!/bin/bash
# As described in `CONTRIBUTING.md`, this script checks and formats Optuna's source codes by
# `black`, `blackdoc`, and `isort`. If you pass `-n` as an option, this script checks codes
# without updating codebase.


missing_dependencies=()
command -v black &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(black)
fi
command -v blackdoc &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(blackdoc)
fi
command -v flake8 &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(flake8)
fi
command -v isort &> /dev/null
if [ $? -eq 1 ] ; then
  missing_dependencies+=(isort)
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

res_black=$(black $target --check --diff 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "black failed. The code will be formatted by black."
    black $target
  else
    echo "$res_black"
    echo "black failed."
    res_all=1
  fi
else
  echo "black succeeded."
fi

res_blackdoc=$(blackdoc $target --check --diff 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "blackdoc failed. The docstrings will be formatted by blackdoc."
    blackdoc $target
  else
    echo "$res_blackdoc"
    echo "blackdoc failed."
    res_all=1
  fi
else
  echo "blackdoc succeeded."
fi

res_flake8=$(flake8 $target)
if [ $? -eq 1 ] ; then
  echo "$res_flake8"
  echo "flake8 failed."
  res_all=1
else
  echo "flake8 succeeded."
fi

res_isort=$(isort $target --check 2>&1)
if [ $? -eq 1 ] ; then
  if [ $update -eq 1 ] ; then
    echo "isort failed. The code will be formatted by isort."
    isort $target
  else
    echo "$res_isort"
    echo "isort failed."
    res_all=1
  fi
else
  echo "isort succeeded."
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
