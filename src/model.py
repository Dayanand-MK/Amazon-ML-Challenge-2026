"""Deterministic local statistical models; no pretrained weights or services."""
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def basic_features(x):
    return x[:, :18]


def estimators():
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300,random_state=2026)),
        "lightgbm_15": LGBMClassifier(n_estimators=160, num_leaves=15, learning_rate=.08,
            reg_lambda=2., min_child_samples=40, random_state=2026, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True),
        "lightgbm_31": LGBMClassifier(n_estimators=240, num_leaves=31, learning_rate=.05,
            reg_lambda=2., min_child_samples=40, random_state=2026, n_jobs=1, verbosity=-1,
            deterministic=True, force_col_wise=True),
        "lightgbm_text_only": make_pipeline(FunctionTransformer(basic_features),
            LGBMClassifier(n_estimators=160, num_leaves=15, learning_rate=.08,
                reg_lambda=2., min_child_samples=40, random_state=2026, n_jobs=1, verbosity=-1,
                deterministic=True, force_col_wise=True)),
    }


def save_model(path, model, threshold, feature_names, metadata):
    with open(path,"wb") as f:
        pickle.dump({"model":model,"threshold":threshold,"feature_names":feature_names,"metadata":metadata},f)


def load_model(path):
    with open(path,"rb") as f:
        return pickle.load(f)
