#!/bin/bash

# copy files to parent directory
cp real_data_analysis/model_explanation/patient_specific_shap.py ./

# run something
"$PYTHON_INTERPRETER" patient_specific_shap.py -c config.ini

# remove when done
rm ./patient_specific_shap.py

