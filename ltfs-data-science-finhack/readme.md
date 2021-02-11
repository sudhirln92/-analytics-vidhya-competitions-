# LTFS Data Science FinHack 3
LTFS Top-up loan Up-sell prediction

| Name | Type  | Public Kernel |Metric | Date |
| ------ | ------ | ------ | ------ | ------ | 
| [LTFS Data Science FinHack 3](https://datahack.analyticsvidhya.com/contest/ltfs-data-science-finhack-3/#ProblemStatement) | Classification | NA | F1 Score | Jan 2021 |

# Steps to Build Model and predict output

1. Download data from analytics vidhya
[analyticsvidhya](https://datahack.analyticsvidhya.com/contest/ltfs-data-science-finhack-3/#DiscussTab/)
keep data in input folder. Convert data from excel format to csv format by using pandas for faster reading data.
2. tune hyper parameter
```python
python3 -m src.hyper_tune --m lgbm --t 'Top-up Month'
```
3. update new hyper parameter in dispatcher script
4. train model
```bash
sh run.sh
```
5. create feature pipeline and feature engineering data by running
```python
python3 -m src.create_folds
```
6. Predict model output by running
```python
python3 -m src.predict
```
