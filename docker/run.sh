#!/bin/bash
# Convenient runner from inside docker folder
cd "$(dirname "$0")/.."
./run.sh "$@"
