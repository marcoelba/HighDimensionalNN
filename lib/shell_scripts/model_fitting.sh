#!/bin/bash

# Source setup scripts
#source ./lib/prepare_temp_directory.sh
# Setup TEMP directory
#setup_temp_directory
echo "Running script with TEMP_DIR: $TEMP_DIR, on MODEL_DIR: $MODEL_DIR"

# copy files to parent directory
cp src/real_data_analysis/model_explanation/model_fitting.py ./

# Run python file
"$PYTHON_INTERPRETER" model_fitting.py -c config.ini

# remove when done
rm ./model_fitting.py
