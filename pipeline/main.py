"""Pipeline Template Module"""
import yaml
from yaml.loader import SafeLoader
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import load_breast_cancer
import pipeline.helpers as helper


# Read
## Setting
with open("./pipeline/models.yaml", "r") as f:
    settings = yaml.load(f, Loader=SafeLoader)
# Data
data = load_breast_cancer(as_frame=True)
# Getting the features we are interested in.
# feats = pd.read_csv('feature selection h dosw20changeresults.csv', sep=';')
# nr_feats= 15
X = data.data
y = data.target
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=1121218
)
# Transform
numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object"]).columns
numeric_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", MinMaxScaler()),
        ("to_int", FunctionTransformer(helper.to_int)),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        # ('imputer', IterativeImputer(sample_posterior = True))
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("number", numeric_pipeline, numerical_features),
        ("category", categorical_pipeline, categorical_features),
    ]
)
pipelines = []
for model, attr in settings["models"].items():

    model = helper.create_model(model, attr["param"])
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
    pipelines.append(pipeline)
# Score
final = []
train_final = []
scoring = helper.create_scoring(settings)
for idx, pipe in enumerate(pipelines):
    results = {}
    train_results = {}
    model = pipe.named_steps["classifier"]
    scores = cross_validate(
        pipe,
        X_train,
        y_train.values.ravel(),
        scoring=scoring,
        cv=5,
        return_train_score=True,
        n_jobs=1,
        error_score="raise",
    )
    results["model"] = model
    for score in scoring.keys():
        results[f"{score}_mean"] = scores[f"test_{score}"].mean()
        results[f"{score}_std"] = scores[f"test_{score}"].std()
    final.append(results)

print("-" * 30)
print(final)
print("- " * 15)
