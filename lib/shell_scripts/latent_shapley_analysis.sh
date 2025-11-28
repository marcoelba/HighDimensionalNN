#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/latent_space_analysis.py ./

# run something
"$PYTHON_INTERPRETER" latent_space_analysis.py -c config.ini

# remove when done
rm ./latent_space_analysis.py
