#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/patient_shapley_groups_analysis.py ./

# run something
"$PYTHON_INTERPRETER" patient_shapley_groups_analysis.py -c config.ini
PYTHON_EXIT=$?

# remove when done
rm ./patient_shapley_groups_analysis.py
exit $PYTHON_EXIT
