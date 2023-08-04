#!/bin/bash
# Check for interactive environment. Need this or conda will not activate
# chmod +x run_scripts.sh
set -e
if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i run_scripts.sh'
	exit
fi
echo "conda activate deep_cpu"
conda activate deep_cpu
echo "delete old models"
rm -f all_data_regressor.csv year_count.yaml deep_learning_mlb_regress_test.h5 deep_learning_mlb_class_test.h5
python mlb_ml_classify_deep_learn_test.py test
git add .
current_date=$(date)
git commit -m "ADD: latest models and data $current_date"
git push
echo "data and models have been pushed to github"