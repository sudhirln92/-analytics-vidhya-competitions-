export TRAINING_DATA=input/train_folds.csv 
export TEST_DATA=input/test.csv 

# stage 1
MODEL=lgbm STAGE=1 SAVE_VALID=True TARGET_COL='Top-up Month' python3 -m src.train