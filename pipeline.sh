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
    "model_fitting.sh" 
    "model_analysis.sh" 
    "generate_shapley_values.sh" 
    "global_shapley_analysis.sh" 
    "patient_predictions.sh" 
    "patient_shapley_explanations.sh" 
    "generate_latent_space_shapley_values.sh" 
    "latent_shapley_analysis.sh" 
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
        if ./lib/shell_scripts/"$script" > stdout_$script; then
            echo "✓ $script completed successfully"
            mark_success "$script"
        else
            echo "✗ $script failed with exit code $?"
            mark_failed "$script"
            # Optional: stop on first failure
            echo "Stopping due to failure in $script"
            exit 1
        fi
    else
        echo "--- Skipping $script (already completed) ---"
    fi
done

echo "Running model fitting"
./lib/shell_scripts/model_fitting.sh > stdout_model_fitting
# create a log file
echo "model_fitting" > pipeline_log.txt

echo "Running model analysis"
./lib/shell_scripts/model_analysis.sh > stdout_model_analysis
echo "model_analysis" >> pipeline_log.txt

echo "Running shapley values generation"
./lib/shell_scripts/generate_shapley_values.sh > stdout_generate_shapley_values
echo "generate_shapley_values" >> pipeline_log.txt

echo "Running shapley analysis"
./lib/shell_scripts/global_shapley_analysis.sh > stdout_global_shapley_analysis
echo "global_shapley_analysis" >> pipeline_log.txt

echo "Running patients predictions"
./lib/shell_scripts/patient_predictions.sh > stdout_patient_predictions
echo "patient_predictions" >> pipeline_log.txt

echo "Running patients shapley"
./lib/shell_scripts/patient_shapley_explanations.sh > stdout_patient_shapley_explanations
echo "patient_shapley_explanations" >> pipeline_log.txt

echo "Running latent shapley generation"
./lib/shell_scripts/generate_latent_space_shapley_values.sh > stdout_latent_space_shapley_values
echo "generate_latent_space_shapley_values" >> pipeline_log.txt

echo "Running latent shapley analysis"
./lib/shell_scripts/latent_shapley_analysis.sh > stdout_latent_shapley_analysis
echo "latent_shapley_analysis" >> pipeline_log.txt

# remove when done
cd ../

rm -r $TEMP_DIR/data
rm -r $TEMP_DIR/src
rm -r $TEMP_DIR/lib
rm -r $TEMP_DIR/__pycache__
