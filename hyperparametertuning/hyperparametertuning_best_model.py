# %%
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from catboost import CatBoostRegressor, Pool

# from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost.utils import eval_metric
import optuna
from optuna.samplers import TPESampler
from hyperopt import Trials, tpe, hp, fmin, space_eval

import warnings

warnings.filterwarnings("ignore")

# %%
path = "/home/pydev/Music/work_files/latest_broko_code"
df = pd.read_csv(path + "/Dataset/ML_CLEAN_DATA__Bdv2.1_RES.csv")
df.info(verbose=True, show_counts=True)

# %%
drop_cols = [
    "ML_Number",
    "Sold_Date",
    "Address",
    "Area",
    "Postal_Code",
    "Air_Conditioning",
    "Exterior1",
    "Rooms",
    "Tax_Year",
    "Water_Included",
    "property_type",
    "lat",
    "lng",
]
df = df.drop(drop_cols, axis=1)
df.isnull().sum()

# %%
data = df.dropna(axis=0, how="any")

# %%
data.isnull().sum()

# %%
X = data.drop("Sold_Price", axis=1)
y = data["Sold_Price"]

# %%
# categorical features
categorical_features = [column for column, dtype in X.dtypes.items() if dtype == object]

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=0.2, random_state=42)


# cat_train_pool = Pool(X_train, y_train, cat_features=categorical_features)
# cat_val_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
# cat_test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# %%
from sklearn.preprocessing import LabelEncoder


def xgb_model(params):
    # Create an instance of LabelEncoder for each categorical column
    label_encoders = {}

    # categorical_columns = [column for column, dtype in X_train.dtypes.items() if dtype==object]
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    model = XGBRegressor(**params["xgboost_params"])
    model.fit(
        X_train,
        y_train,
        verbose=0,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=10,
    )
    y_pred = model.predict(X_valid)
    return eval_metric(y_valid, y_pred, "MAPE")


# %%
def catboost_model(params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    cat_train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    cat_val_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
    cat_test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    model = CatBoostRegressor(**params["catboost_params"])

    model.fit(cat_train_pool, eval_set=cat_val_pool, early_stopping_rounds=10)

    y_pred = model.predict(cat_val_pool)

    return eval_metric(cat_val_pool.get_label(), y_pred, "MAPE")


# %%
# def calc_test_quality(train_pool=cat_train_pool, val_pool=cat_val_pool, test_pool=cat_test_pool, **kwargs):
#     model = CatBoostRegressor(**kwargs, random_seed=42)
#     model.fit(train_pool, verbose=0, eval_set=val_pool)
#     y_pred = model.predict(test_pool)
#     return eval_metric(test_pool.get_label(), y_pred, 'MAPE')


# %%
def hyperopt_objective(params):
    print(params)
    if params["model"] == "xgboost":
        return xgb_model(params)
    else:
        return catboost_model(params)


space = hp.choice(
    "model",
    [
        {
            "model": "xgboost",
            "xgboost_params": {
                "max_depth": hp.randint("xgboost_max_depth", 5, 30),
                "learning_rate": hp.quniform("xgboost_learning_rate", 0.01, 0.5, 0.01),
                "n_estimators": hp.randint("xgboost_n_estimators", 5, 50),
                "reg_lambda": hp.uniform("xgboost_reg_lambda", 0, 1),
                "reg_alpha": hp.uniform("xgboost_reg_alpha", 0, 1),
            },
        },
        {
            "model": "catboost",
            "catboost_params": {
                "learning_rate": hp.uniform("catboost_learning_rate", 0.01, 0.1),
                "depth": hp.randint("catboost_depth", 5, 10),
                "l2_leaf_reg": hp.uniform("catboost_l2_leaf_reg", 1, 10),
                "boosting_type": hp.choice(
                    "catboost_boosting_type", ["Ordered", "Plain"]
                ),
                "max_ctr_complexity": hp.quniform(
                    "catboost_max_ctr_complexity", 2, 8, 1
                ),
            },
        },
    ],
)


# %%
# Hyperopts Trials() records all the model and run artifacts.
trials = Trials()

# Fmin will call the objective funbction with selective param set.
# The choice of algorithm will narrow the searchspace.

best = fmin(
    hyperopt_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    rstate=np.random.seed(123),
    trials=trials,
)

# Best_params of the best model
best_params = space_eval(space, best)

# %%
print("Best model: ", best_params)
print("Best hyperparameters: ", best_params[best_params["model"] + "_params"])

# %%
best_params["model"]

# %%
best_df = pd.DataFrame([best_params])
best_df.to_csv(path + "/Dataset/best.csv", index=False)
