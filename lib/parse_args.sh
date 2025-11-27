#!/bin/bash

# Simple version - just run the parsing
if [ -z "${TEMP_DIR:-}" ] || [ -z "${MODEL_DIR:-}" ]; then
    while getopts d:m: option; do
        case "${option}" in
            d) TEMP_DIR=${OPTARG};;
            m) MODEL_DIR=${OPTARG};;
        esac
    done
    shift $((OPTIND-1))
    
    if [ -z "${TEMP_DIR:-}" ] || [ -z "${MODEL_DIR:-}" ]; then
        echo "Error: Both -d and -m options are required when running standalone" >&2
        exit 1
    fi
    export TEMP_DIR MODEL_DIR
fi
