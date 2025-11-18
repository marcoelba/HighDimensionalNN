#!/bin/bash

# copy files to parent directory
cp real_data_analysis/model_explanation/shap_explanations.py ./

# run something
"$PYTHON_INTERPRETER" shap_explanations.py -c config.ini

# remove when done
rm ./shap_explanations.py
