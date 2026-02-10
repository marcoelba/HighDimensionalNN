#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/global_shapley_analysis.py ./

# run something
"$PYTHON_INTERPRETER" global_shapley_analysis.py -c config.ini
PYTHON_EXIT=$?

# remove when done
rm ./global_shapley_analysis.py
exit $PYTHON_EXIT
