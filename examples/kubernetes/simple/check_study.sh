#!/bin/sh
optuna studies --storage "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" | grep kubernetes > /dev/null
echo $?
