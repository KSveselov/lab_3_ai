from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEED, TARGET_COLUMN
from e1 import prepare
from split_data import load_data


CV_SPLITS = 5
N_CANDIDATES = 4
SEARCH_FITS = CV_SPLITS * N_CANDIDATES


def main():
    train_df, test_df = load_data()
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    model = Pipeline([
        ("preprocess", prepare()),
        ("regressor", RandomForestRegressor(random_state=SEED, n_jobs=2)),
    ])

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
    search.fit(x_train, y_train)

    predictions = search.best_estimator_.predict(x_test)

    print(f"Бюджет поиска: {SEARCH_FITS} CV-обучений + итоговое обучение")
    print(f"Лучшие параметры: {search.best_params_}")
    print(f"CV MSE: {-search.best_score_:.3f}")
    print(f"Test MSE: {mean_squared_error(y_test, predictions):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
