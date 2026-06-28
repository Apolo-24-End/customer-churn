"""
Two-layer stacking ensemble.
Layer 1: base pipelines (already trained on full X_train).
Layer 2: LogisticRegression meta-learner trained on out-of-fold predictions.
Implements predict / predict_proba so it's drop-in compatible with evaluate_all().
"""
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import config


class StackedModel:
    def __init__(self, base_pipelines: dict, cv_folds: int = config.CV_FOLDS):
        self.base_pipelines = base_pipelines  # {name: fitted_pipeline}
        self.cv_folds = cv_folds
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE, class_weight="balanced")
        self._fitted = False

    def fit(self, X_train, y_train) -> "StackedModel":
        """Generate OOF predictions from base models, then fit the meta-learner."""
        print("[stacker] Generating out-of-fold predictions ...")
        n_base = len(self.base_pipelines)
        oof = np.zeros((len(y_train), n_base))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

        for i, (name, pipeline) in enumerate(self.base_pipelines.items()):
            print(f"[stacker]   OOF for {name} ...")
            for tr_idx, val_idx in cv.split(X_train, y_train):
                X_tr = X_train.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_val = X_train.iloc[val_idx]
                clone = copy.deepcopy(pipeline)
                clone.fit(X_tr, y_tr)
                oof[val_idx, i] = clone.predict_proba(X_val)[:, 1]

        self.meta_learner.fit(oof, y_train)
        self._fitted = True
        print("[stacker] Meta-learner fitted.")
        return self

    def _base_probs(self, X) -> np.ndarray:
        return np.column_stack([
            pipe.predict_proba(X)[:, 1]
            for pipe in self.base_pipelines.values()
        ])

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_learner.predict_proba(self._base_probs(X))

    def predict(self, X) -> np.ndarray:
        return self.meta_learner.predict(self._base_probs(X))
