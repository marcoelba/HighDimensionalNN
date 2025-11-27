#!/bin/bash

# source the parser
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIB_DIR/parse_args.sh"
echo "Running script with TEMP_DIR: $TEMP_DIR, on MODEL_DIR: $MODEL_DIR"


setup_temp_directory() {
    echo "Setting up temporary directory..."
    
    # Create TEMP directory if it doesn't exist (-p)
    mkdir -p "$TEMP_DIR"
    
    echo "Copying files to $TEMP_DIR..."

    # copy if files have been updated (-u) or are missing (-p)
    cp -r -u -p data $TEMP_DIR 2>/dev/null || echo "No files to copy or source directory doesn't exist"
    cp -r -u -p src $TEMP_DIR 2>/dev/null || echo "No files to copy or source directory doesn't exist"
    cp -r -u -p lib $TEMP_DIR 2>/dev/null || echo "No files to copy or source directory doesn't exist"

    cp -u -p $MODEL_DIR/config.ini $TEMP_DIR 2>/dev/null || echo "No files to copy or source directory doesn't exist"
    cp -u -p $MODEL_DIR/full_model.py $TEMP_DIR 2>/dev/null || echo "No files to copy or source directory doesn't exist"
    
    # Create a marker file to indicate TEMP is initialized
    touch "$TEMP_DIR/.initialized"

    echo "TEMP directory initialized/updated"
    
    export TEMP_DIR
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_temp_directory "$@"
fi
