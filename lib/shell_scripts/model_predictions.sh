#!/bin/bash

# copy files to parent directory
cp src/real_data_analysis/model_explanation/model_predictions.py ./

# run something
"$PYTHON_INTERPRETER" model_predictions.py -c config.ini
PYTHON_EXIT=$?

# remove when done
rm ./model_predictions.py
exit $PYTHON_EXIT

