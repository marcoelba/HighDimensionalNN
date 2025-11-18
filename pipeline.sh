#!/bin/bash

# define temporary variable names
while getopts d:m: option; do
    case "${option}" in
        d) TEMP_DIR=${OPTARG};;
        m) MODEL_DIR=${OPTARG};;
    esac
done

# TEMP_DIR="TEMP"
# MODEL_DIR="running_model"

# make temporary working directory
mkdir -p $TEMP_DIR

# get Python absolute path using PWD
PYTHON_INTERPRETER="$PWD/python_env/bin/python3.12"
# Export so it's available to child processes
export PYTHON_INTERPRETER
echo "Python interpreter: $PYTHON_INTERPRETER"

# copy files to working directory
cp -r data $TEMP_DIR
cp -r src $TEMP_DIR
cp -r real_data_analysis $TEMP_DIR

cp -r real_data_analysis/shell_scripts/* $TEMP_DIR

cp $MODEL_DIR/config.ini $TEMP_DIR
cp $MODEL_DIR/full_model.py $TEMP_DIR

# move into $TEMP_DIR
cd $TEMP_DIR

# Run shell scripts in order
set -e

echo "Running model fitting"
source ./model_fitting.sh > stdout_model_fitting

echo "Running model analysis"
source ./model_analysis.sh > stdout_model_analysis

echo "Running shapley values generation"
source ./generate_shapley_values.sh > stdout_generate_shapley_values

echo "Running shapley analysis"
source ./global_shapley_analysis.sh > stdout_global_shapley_analysis

echo "Running patients predictions"
source ./patient_predictions.sh > stdout_patient_predictions

echo "Running patients shapley"
source ./patient_shapley_explanations.sh > stdout_patient_shapley_explanations

# remove when done
cd ../

rm -r $TEMP_DIR/data
rm -r $TEMP_DIR/src
rm -r $TEMP_DIR/real_data_analysis
rm -r $TEMP_DIR/*.sh
rm -r $TEMP_DIR/__pycache__
