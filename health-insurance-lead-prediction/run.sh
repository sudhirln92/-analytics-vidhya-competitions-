export TRAINING_DATA=input/train_folds.csv 
export TEST_DATA=input/test.csv 

# stage 1
MODEL=lgbm STAGE=1 SAVE_VALID=False TARGET_COL='Response' python3 -m src.train