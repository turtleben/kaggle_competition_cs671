# kaggle_competition_cs671

To run my model, we can just type ./run.sh after installing all the package.
In run.sh, the usage is 
python -u train.py \
    --output_path results/prediction0.csv  \
    --whole_training 1 \
    --model_type voting \
    --feature_select 0

output_path means the testing prediction result path.
whole_training = 1 means that we want to perform training on the whole training set and generate testing output.
whole_training = 0 means that we train the model based on cross-validation.
model_type includes ["random_forest", 'gradient_boosting', 'adaboost', 'mlp', 'xgb', 'voting], you can indicate which model you want to use to train our model
feature_select = 0 means that you do want the feature engineering and just perform training on the whole training dataset
feature_select = 1 means that we first use feature engineering and then train our model.

The default setting is the model with the best accuracy I got so far.
