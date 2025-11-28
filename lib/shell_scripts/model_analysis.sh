#!/bin/bash

# Source setup scripts
# source ./lib/prepare_temp_directory.sh
# Setup TEMP directory
#setup_temp_directory
echo "Running script with TEMP_DIR: $TEMP_DIR, on MODEL_DIR: $MODEL_DIR"

# copy files to parent directory
cp src/real_data_analysis/model_explanation/post_analysis.py ./

# run something
"$PYTHON_INTERPRETER" post_analysis.py -c config.ini

# remove when done
rm ./post_analysis.py
