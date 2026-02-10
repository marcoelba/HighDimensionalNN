#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/generate_shapley_values.py ./

# run something
"$PYTHON_INTERPRETER" generate_shapley_values.py -c config.ini
PYTHON_EXIT=$?

# remove when done
rm ./generate_shapley_values.py
exit $PYTHON_EXIT
