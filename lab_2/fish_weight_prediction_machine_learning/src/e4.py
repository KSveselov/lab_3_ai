from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    SEED,
    TARGET_COLUMN,
)
from e1 import prepare
from split_data import load_data


CV_SPLITS = 5
N_CANDIDATES = 4
SEARCH_FITS_PER_VERSION = CV_SPLITS * N_CANDIDATES
VPROXY_COLUMN = "Vproxy"


def add_vproxy(data_frame):
    return data_frame.assign(
        Vproxy=data_frame["Length3"] * data_frame["Height"] * data_frame["Width"]
    )


def evaluate(train_df, test_df, numeric_columns, label):
    feature_columns = numeric_columns + CATEGORICAL_COLUMNS
    model = Pipeline(
        [
            ("preprocess", prepare(numeric_columns)),
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
    train_df, test_df = load_data()

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
