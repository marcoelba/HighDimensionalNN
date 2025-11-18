#!/bin/bash

# copy files to parent directory
cp real_data_analysis/model_explanation/post_analysis.py ./

# run something
"$PYTHON_INTERPRETER" post_analysis.py -c config.ini

# remove when done
rm ./post_analysis.py
