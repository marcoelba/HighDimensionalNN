#!/bin/bash

# make temporary working directory
mkdir -p TEMP

# get Python absolute path using PWD
PYTHON_INTERPRETER="$PWD/python_env/bin/python3.12"
# Export so it's available to child processes
export PYTHON_INTERPRETER
echo "Python interpreter: $PYTHON_INTERPRETER"

# copy files to working directory
cp -r data TEMP
cp -r src TEMP
cp -r real_data_analysis TEMP

cp -r real_data_analysis/shell_scripts/* TEMP

cp running_model/model_fitting.py TEMP
cp running_model/config.ini TEMP
cp running_model/full_model.py TEMP

# move into TEMP
cd ./TEMP

# Run shell scripts in order
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

rm -r TEMP/data
rm -r TEMP/src
rm -r TEMP/real_data_analysis
