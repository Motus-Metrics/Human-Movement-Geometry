# eval/evaluate.py
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

def eval_ab(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    features: [N, F], labels: [N] (0/1; 0=slow/comfort, 1=fast/fatigue)
    """
    clf = LogisticRegression(max_iter=200)
    clf.fit(features, labels)
    prob = clf.predict_proba(features)[:,1]
    auc = roc_auc_score(labels, prob)
    return {"AUC": float(auc)}
