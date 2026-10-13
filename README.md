# Gradient Boosting from Scratch

This repository contains my solution for **Practical Homework 6** in the **Machine Learning course at HSE FCS**: implementing gradient boosting by hand, tuning it with Optuna and comparing it with CatBoost.

**Assignment**: [link](https://github.com/esokolov/ml-course-hse/tree/master/2024-fall/homework-practice/homework-practice-06-boosting)

## Project Structure

```
├── boosting.py                                  # Boosting implementation
├── homework_practice_06_boosting-Anikin.ipynb   # Experiments, tuning, comparison with CatBoost
├── images                                       # Screenshots of the Optuna storage and CatBoost learning curves
└── README.md
```

The data (`x.npz`, `y.npy`) is in the assignment folder: binary classification, 23.5K objects, 169 sparse features. Split 80/10/10 into train/valid/test with `random_state=1337`.

## Implementation

`Boosting` in `boosting.py` is written from scratch in NumPy for log-loss; `sklearn.tree.DecisionTreeRegressor` is used as the base model. Each tree is fitted to the antigradient, and its step size γ is found by line search over [0, 1].

Implemented parameters, mirroring XGBoost / LightGBM / CatBoost:

| Feature | Parameters | Library analogue |
|---|---|---|
| Early stopping on validation loss | `early_stopping_rounds` | `early_stopping_rounds` |
| Bernoulli bootstrap | `subsample`, `bootstrap_type='Bernoulli'` | CatBoost `bootstrap_type` |
| Bayesian bootstrap, weights w = (−log U)^t | `bagging_temperature`, `bootstrap_type='Bayesian'` | CatBoost `bagging_temperature` |
| Gradient-based One-Side Sampling | `goss`, `goss_k`, `subsample` | LightGBM `goss`, CatBoost `MVS` |
| Random feature subspace | `rsm` | CatBoost `rsm`, `colsample_bytree` |
| Feature quantization | `quantization_type='Uniform' / 'Quantile'`, `nbins` | `max_bin`, CatBoost `quantize` |
| Feature importance | `feature_importances_` | `get_feature_importance()` |

## Results

ROC-AUC on the test set. Each feature is checked separately on top of the base configuration (100 trees, `max_depth=4`).

| Configuration | Test ROC-AUC |
|---|---|
| Base | 0.9334 |
| Bernoulli bootstrap, `subsample=0.2` | 0.9343 |
| Bayesian bootstrap, `bagging_temperature=0.2` | 0.9324 |
| GOSS, `goss_k=0.2`, `subsample=0.3` | 0.9333 |
| Quantile quantization | 0.9342 |
| Uniform quantization | 0.9314 |
| RSM 0.8 + `subsample=0.2` | 0.9176 |
| **Own boosting, tuned in Optuna** | **0.9393** |
| **CatBoost, tuned in Optuna** | **0.9393** |

Hyperparameters were searched with Optuna (100 trials, TPE) on the validation set: tree depth and leaf sizes, number of trees, learning rate, `subsample`, `early_stopping_rounds`. The study is saved to a database (SQLite in the notebook; the screenshots in `images/` show an earlier run stored in MySQL), and the training loss history and fit time of each trial are saved as user attributes. CatBoost was tuned the same way over `depth`, `n_estimators`, `learning_rate`, `subsample` and `rsm`.

Findings:
- After tuning, my implementation matches CatBoost on test ROC-AUC (0.939). CatBoost is much faster: 100 Optuna trials took about 1.5 minutes versus about 9.5 minutes for my implementation.
- Most regularisation options change quality by less than 0.002 on their own; the gain comes from tuning them together. RSM hurts noticeably here, likely because two features (#166 and #164) account for 75% of the total importance, and in this implementation a dropped feature is lost for every tree.
- Both the tuned boosting and logistic regression are reasonably well calibrated on this data.

## Known limitations

- **Bernoulli bootstrap and GOSS sample with replacement** (`np.random.choice(n, num)`). Bernoulli sampling should include each object independently with probability `subsample`: `replace=False` or a mask `rand < subsample`.
- **Quantization bins are computed separately for each set**: `quantization(X_val)` uses the validation set's own bins. Bins should be fitted on train and reused, otherwise the same split means different values on train and validation.
- **RSM picks one feature subset for the whole model**, with repetitions. In CatBoost `rsm` resamples features at each split.
- The best `learning_rate` is 1.0, which is expected: γ is found by line search over [0, 1], so `learning_rate` only bounds the step from above.
