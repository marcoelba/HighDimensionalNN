#!/bin/bash

# define temporary variable names
while getopts d:m:r option; do
    case "${option}" in
        d) TEMP_DIR=${OPTARG};;
        m) MODEL_DIR=${OPTARG};;
        r) RESTART_FLAG=true;;
    esac
done
export TEMP_DIR MODEL_DIR
export RESTART_FLAG=${RESTART_FLAG:-false} # set to false by default if not provided

set -e

# Source setup scripts
source ./lib/prepare_temp_directory.sh
# Setup TEMP directory
setup_temp_directory

# Configuration
STATUS_FILE="./pipeline_log.txt"
SCRIPTS=(
    "model_fitting" 
    "model_analysis" 
    "generate_shapley_values" 
    "global_shapley_analysis" 
    "patient_predictions" 
    "patient_shapley_explanations" 
    "generate_latent_space_shapley_values" 
    "latent_shapley_analysis" 
)

# Initialize or read status file
initialize_status() {
    if [ ! -f "$STATUS_FILE" ] || [ "$RESTART_FLAG" = "true" ]; then
        echo "Initializing status tracking..."
        > "$STATUS_FILE"  # Create or clear the file
    fi
}

# Check if script should run
should_run_script() {
    local script_name="$1"
    ! grep -q "^SUCCESS:$script_name$" "$STATUS_FILE"
}

# Mark script as successful
mark_success() {
    local script_name="$1"
    echo "SUCCESS:$script_name" >> "$STATUS_FILE"
}

# Mark script as failed
mark_failed() {
    local script_name="$1"
    echo "FAILED:$script_name" >> "$STATUS_FILE"
}

# get Python absolute path using PWD
PYTHON_INTERPRETER="$PWD/python_env/bin/python3.12"
# Export so it's available to child processes
export PYTHON_INTERPRETER
echo "Python interpreter: $PYTHON_INTERPRETER"

# move into $TEMP_DIR
cd $TEMP_DIR

# Run shell scripts in order
initialize_status "$@"


for script in "${SCRIPTS[@]}"; do
    if should_run_script "$script"; then
        echo "=== Running $script ==="
        if ./lib/shell_scripts/"$script".sh > stdout_$script; then
            echo "✓ $script completed successfully"
            mark_success "$script"
        else
            echo "✗ $script failed with exit code $?"
            mark_failed "$script".sh
            # Optional: stop on first failure
            echo "Stopping due to failure in $script"
            exit 1
        fi
    else
        echo "--- Skipping $script (already completed) ---"
    fi
done

# remove when done
cd ../

rm -r $TEMP_DIR/data
rm -r $TEMP_DIR/src
rm -r $TEMP_DIR/lib
rm -r $TEMP_DIR/__pycache__
