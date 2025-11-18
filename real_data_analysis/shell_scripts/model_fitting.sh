#!/bin/bash

# copy files to parent directory
cp real_data_analysis/model_explanation/model_fitting.py ./

# Run python file
"$PYTHON_INTERPRETER" model_fitting.py -c config.ini

# remove when done
rm ./model_fitting.py
