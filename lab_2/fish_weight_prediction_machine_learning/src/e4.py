import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    CATEGORICAL_COLUMNS,
    DATA_PATH,
    DATA_SPLIT,
    NUMERIC_COLUMNS,
    SEED,
    TARGET_COLUMN,
)


CV_SPLITS = 5
N_CANDIDATES = 4
SEARCH_FITS_PER_VERSION = CV_SPLITS * N_CANDIDATES
VPROXY_COLUMN = "Vproxy"


def load_fixed_split():
    data_frame = pd.read_csv(DATA_PATH)
    with DATA_SPLIT.open(encoding="utf-8") as file:
        split_indices = json.load(file)

    return (
        data_frame.loc[split_indices["train_index"]].copy(),
        data_frame.loc[split_indices["test_index"]].copy(),
    )


def make_preprocessor(numeric_columns):
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def add_vproxy(data_frame):
    return data_frame.assign(
        Vproxy=data_frame["Length3"] * data_frame["Height"] * data_frame["Width"]
    )


def evaluate(train_df, test_df, numeric_columns, label):
    feature_columns = numeric_columns + CATEGORICAL_COLUMNS
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_columns)),
            ("regressor", RandomForestRegressor(random_state=SEED, n_jobs=1)),
        ]
    )
    search = GridSearchCV(
        estimator=model,
        param_grid={
            "regressor__n_estimators": [300],
            "regressor__max_depth": [None, 12],
            "regressor__min_samples_leaf": [1, 2],
        },
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED),
        n_jobs=2,
    )
    search.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    predictions = search.best_estimator_.predict(test_df[feature_columns])

    print(f"\n{label}")
    print(f"Лучшие параметры: {search.best_params_}")
    print(f"CV MSE: {-search.best_score_:.3f}")
    print(f"Test MSE: {mean_squared_error(test_df[TARGET_COLUMN], predictions):.3f}")
    print(f"Test MAE: {mean_absolute_error(test_df[TARGET_COLUMN], predictions):.3f}")
    print(f"Test R²: {r2_score(test_df[TARGET_COLUMN], predictions):.3f}")


def main():
    train_df, test_df = load_fixed_split()

    print("Вопрос: улучшает ли Vproxy = Length3 × Height × Width прогноз массы?")
    print(
        "Бюджет: "
        f"{SEARCH_FITS_PER_VERSION} CV-обучений + итоговое обучение на версию."
    )
    evaluate(train_df, test_df, NUMERIC_COLUMNS, "Версия A: базовые признаки")
    evaluate(
        add_vproxy(train_df),
        add_vproxy(test_df),
        NUMERIC_COLUMNS + [VPROXY_COLUMN],
        "Версия B: базовые признаки + Vproxy",
    )


if __name__ == "__main__":
    main()
