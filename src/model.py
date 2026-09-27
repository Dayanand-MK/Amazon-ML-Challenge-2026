"""Deterministic local statistical models; no pretrained weights or services."""
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def basic_features(x):
    return x[:, :18]


def estimators():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300,random_state=2026)),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,
            learning_rate=.08,l2_regularization=2.,min_samples_leaf=40,early_stopping=False,random_state=2026),
        "hist_gradient_boosting_text_only": make_pipeline(FunctionTransformer(basic_features),
            HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.08,
                l2_regularization=2.,min_samples_leaf=40,early_stopping=False,random_state=2026)),
    }


def save_model(path, model, threshold, feature_names, metadata):
    with open(path,"wb") as f:
        pickle.dump({"model":model,"threshold":threshold,"feature_names":feature_names,"metadata":metadata},f)


def load_model(path):
    with open(path,"rb") as f:
        return pickle.load(f)
