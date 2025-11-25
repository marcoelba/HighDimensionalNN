#!/bin/bash

# copy files to parent directory
cp real_data_analysis/model_explanation/latent_space_shap_explanations.py ./

# run something
"$PYTHON_INTERPRETER" latent_space_shap_explanations.py -c config.ini

# remove when done
rm ./latent_space_shap_explanations.py
