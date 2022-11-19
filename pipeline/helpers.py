"""Module for supporting functions"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
import pandas as pd


def create_model(name, param):
    param_string = ""
    for n, p in param.items():
        param_string += f"{n}={p},"
    eval_string = f"{name}({param_string})"
    model = eval(eval_string)
    return model


def create_scoring(settings):
    scoring = {}
    for name, param in settings["scoring"].items():
        # param_string = ''
        # for n, p in param.items():
        #    param_string += f'{n}={p},'
        eval_string = f"make_scorer({param['class']})"
        score = eval(eval_string)
        scoring[name] = score
    return scoring


def to_int(x):
    return pd.DataFrame(x).astype(int)
