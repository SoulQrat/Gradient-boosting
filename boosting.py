from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional

import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int | None = None,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        goss: bool | None = False,
        goss_k: float | int = 0.2,
        rsm: float | int = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255,
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(np.where(y==1, 1, -1) * z)).mean()
        self.loss_derivative = lambda y, z: -np.where(y==1, 1, -1) * self.sigmoid(-np.where(y==1, 1, -1) * z) # -y * (1 - self.sigmoid(y * z))

        self.goss = goss
        self.goss_k = goss_k

        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.feature_importances_: list = []

    def partial_fit(self, X, y, old_predictions):
        n = len(y)
        w = None
        if self.goss:
            num_s = int(n * self.subsample)
            num_b = int(n * self.goss_k)
            sorted = np.argsort(np.abs(self.loss_derivative(y, old_predictions)))[::-1]
            topSet = sorted[:num_b]
            randSet = sorted[num_b:]
            randSet = randSet[np.random.choice(len(randSet), num_s)]
            usedSet = np.hstack([topSet, randSet])
            w = np.ones(n)
            w[randSet] *= ((1 - self.goss_k) / self.subsample)
            ids = usedSet
        elif self.bootstrap_type == 'Bernoulli':
            num = int(n * self.subsample)
            ids = np.random.choice(n, num)
        elif self.bootstrap_type == 'Bayesian':
            w = np.power(-np.log(np.random.uniform(0, 1, n)), self.bagging_temperature)
            ids = np.arange(n)
        else:
            ids = np.arange(n)

        model = self.base_model_class(**self.base_model_params).fit(X[ids], -self.loss_derivative(y, old_predictions)[ids], sample_weight=(w[ids] if w is not None else w))
        self.models.append(model)

        gamma = self.find_optimal_gamma(y, old_predictions, model.predict(X))
        self.gammas.append(gamma)

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        X_train = self.quantization(X_train)
        X_val = self.quantization(X_val)

        self.models.clear()
        self.gammas.clear()

        self.feature_importances_ = np.zeros(X_train.shape[1])

        flag = False if y_val is None else True

        train_predictions = np.zeros(y_train.shape[0])
        if flag:
            val_predictions = np.zeros(y_val.shape[0])

        stop = 0
        
        for _ in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)
            train_predictions += self.models[-1].predict(X_train) * self.learning_rate * self.gammas[-1]

            self.feature_importances_ += self.models[-1].feature_importances_

            self.history['train_loss'].append(self.loss_fn(y_train, train_predictions))
            self.history['train_roc_auc'].append(self.score(X_train, y_train))
            if flag:
                val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1].predict(X_val)
                self.history['val_loss'].append(self.loss_fn(y_val, val_predictions))
                self.history['val_roc_auc'].append(self.score(X_val, y_val))

            if self.early_stopping_rounds is not None:
                if stop == self.early_stopping_rounds:
                    break
                
                if len(self.history['val_loss']) > 1:
                    if self.history['val_loss'][-1] >= self.history['val_loss'][-2]:
                        stop += 1
                    else:
                        stop = 0

                if stop == self.early_stopping_rounds:
                    break

        self.feature_importances_ /= len(self.models)
        self.feature_importances_ /= np.sum(self.feature_importances_)

        if plot:
            nums = np.arange(1, len(self.models) + 1)
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].set_xlabel('Estimators')
            axs[0].set_ylabel('Loss value')
            axs[0].set_title('Loss')
            axs[1].set_xlabel('Estimators')
            axs[1].set_ylabel('ROC-AUC value')
            axs[1].set_title('ROC-AUC')

            axs[0].plot(nums, self.history['train_loss'], color='royalblue')
            axs[1].plot(nums, self.history['train_roc_auc'], color='royalblue')
            if flag:
                axs[0].plot(nums, self.history['val_loss'], color='red')
                axs[1].plot(nums, self.history['val_roc_auc'], color='red')
                axs[1].legend(['Train', 'Validation'])
            else:
                axs[1].legend(['Train'])
            plt.show()


    def predict_proba(self, X):
        predictions = np.zeros(X.shape[0])
        for i in range(len(self.models)):
            predictions += self.models[i].predict(X) * self.gammas[i] * self.learning_rate
        probs = self.sigmoid(predictions)
        return np.vstack([1 - probs, probs]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        loss = np.zeros(len(self.models))
        predictions = np.zeros(len(y))
        for i in range(len(self.models)):
            predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(X)
            loss[i] = self.loss_fn(y, predictions)
        
        nums = np.arange(1, len(self.models) + 1)
        plt.figure(figsize=(6, 4))
        plt.xlabel('Estimators')
        plt.ylabel('Loss value')
        plt.title('Boosting losses')
        plt.plot(nums, loss, color='royalblue')
        plt.show()

    def quantization(self, X):
        rng = np.random.default_rng(seed=28)
        if X is None:
            return None
        elif self.rsm != 1:
            n_features = X.shape[1]
            num = int(self.rsm * n_features)
            ids = rng.choice(n_features, num)
            X = X.T[ids].T
        if self.quantization_type is not None:
            n_features = X.shape[1]
            if self.quantization_type == 'Uniform':
                X = np.array(X.todense()).T
                for i in range(n_features):
                    bins = np.linspace(np.min(X[i]), np.max(X[i]), self.nbins) 
                    x_bins = np.digitize(X[i], bins)
                    X[i] = x_bins
                X = csr_matrix(X.T)
            elif self.quantization_type == 'Quantile':
                X = np.array(X.todense()).T
                for i in range(n_features):                    
                    bins = np.percentile(X[i], q=np.linspace(0, 100, self.nbins)) 
                    x_bins = np.digitize(X[i], np.sort(bins))
                    X[i] = x_bins
                X = csr_matrix(X.T)
            elif self.quantization_type == 'MinEntropy':
                pass
            elif self.quantization_type == 'Piecewise':
                pass
        return X
