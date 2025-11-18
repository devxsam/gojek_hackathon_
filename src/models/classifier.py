from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, roc_auc_score

class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def evaluate(self, df_test: pd.DataFrame):
        y_true = df_test[self.target].values
        y_proba = self.clf.predict_proba(df_test[self.features].values)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        
        f1 = f1_score(y_true, y_pred)
        precision=precision_score(y_true,y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        return {
            "f1_score": f1,
            "roc_auc": roc_auc,
            
        }
    
    
    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]
