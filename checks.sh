#!/bin/bash

res_pip_list=$(pip freeze)
if [[ ! $(echo $res_pip_list | grep black) ]] ; then
  pip install black
fi
if [[ ! $(echo $res_pip_list | grep flake8) ]] ; then
  pip install flake8
fi
if [[ ! $(echo $res_pip_list | grep mypy) ]] ; then
  pip install mypy
fi

black_update=0
while getopts "u" OPT
do
  case $OPT in
    u) black_update=1
       ;;
    *) ;;
  esac
done

res_black=$(black . --check 2>&1)
if [[ $(echo $res_black | grep reformatted) ]] ; then
  if [ $black_update == 1 ] ; then
    echo "Failed with black. The code will be formatted by black."
    black .
  else
    echo "Failed with black."
    echo "$res_black"
    exit 1
  fi
else
  echo "Success in black."
fi

res_flake8=$(flake8 .)
if [[ $res_flake8 ]] ; then
  echo "Failed with flake8."
  echo "$res_flake8"
  exit 1
fi
echo "Success in flake8."

res_mypy=$(mypy .)
if [[ ! $(echo $res_mypy | grep Success) ]] ; then
  echo "Failed with mypy."
  echo "$res_mypy"
  exit 1
fi
echo "Success in mypy"

