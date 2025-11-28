#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/shap_out_analysis.py ./

# run something
"$PYTHON_INTERPRETER" shap_out_analysis.py -c config.ini

# remove when done
rm ./shap_out_analysis.py
