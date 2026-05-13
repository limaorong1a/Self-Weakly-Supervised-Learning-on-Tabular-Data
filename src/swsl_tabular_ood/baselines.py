from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier


def train_logistic_regression(x: np.ndarray, y: np.ndarray, seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, n_jobs=-1).fit(x, y)


def train_random_forest(x: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)


def train_sklearn_mlp(x: np.ndarray, y: np.ndarray, seed: int) -> SklearnMLPClassifier:
    return SklearnMLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=80,
        random_state=seed,
        early_stopping=True,
    ).fit(x, y)
