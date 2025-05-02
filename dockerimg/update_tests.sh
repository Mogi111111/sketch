#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURRENT_DIR=$(pwd)

if [ "$SCRIPT_DIR" != "$CURRENT_DIR" ]; then
    echo "Error: This script must be run from its own directory: $SCRIPT_DIR" >&2
    exit 1
fi

go test -httprecord ".*" -rewritewant
