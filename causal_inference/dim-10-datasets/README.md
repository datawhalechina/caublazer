## Data Description

data.columns = [['x0', 'x1','x2','x3','x4','x5','x6','x7','x8','x9','treatment','y','true_ite']]

- x0 ~ x9: the data of `X`

- treatment: the causal estimation effect target

- y : the data of `y`

- true_ite: the ite value of this sample

## Task

- You need to split the dataset of `train data` and `test data` with the `ratio` of 0.8 and the  `random_seed` of 2022

- You need to estimate the treatment effect of each sample and get the `est_ite` value, calculate the `ABS loss` and `MSE loss` with the `true_ite`

| dataset type | 1000    | 3000 | 5000 | 10000 | 30000 |
| ------------ | ------- | ---- | ---- | ----- | ----- |
| train_ABS    | 0.04*** | **** | **** | ****  | ****  |
| test_ABS     | 0.07*** | **** | **** | ****  | ****  |

| dataset type | 1000    | 3000 | 5000 | 10000 | 30000 |
| ------------ | ------- | ---- | ---- | ----- | ----- |
| train_MSE    | 0.04*** | **** | **** | ****  | ****  |
| test_MSE     | 0.07*** | **** | **** | ****  | ****  |

- You need to get the result of `1000/3000/5000/10000/30000` samples, and the different number of dataset has been prepared.
