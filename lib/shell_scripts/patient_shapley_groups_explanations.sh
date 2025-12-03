#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/patient_groups_shap.py ./

# run something
"$PYTHON_INTERPRETER" patient_groups_shap.py -c config.ini

# remove when done
rm ./patient_groups_shap.py
