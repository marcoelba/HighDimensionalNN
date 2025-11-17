# bash
# export PYTHONPATH=../

# copy files to parent directory
cp scripts/model_fitting.py ./
cp scripts/config.ini ./
cp scripts/full_model.py ./

./python_env/bin/python3.12 model_fitting.py -c config.ini

# remove when done
rm ./model_fitting.py
rm ./config.ini
rm ./full_model.py
