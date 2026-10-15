import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN, SEED
from e1 import prepare
from split_data import load_data


def train_and_predict():
    train_df, test_df = load_data()
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]
    model = Pipeline([
        ("preprocess", prepare()),
        ("regressor", Ridge()),
    ])
    search = GridSearchCV(
        estimator=model,
        param_grid={"regressor__alpha": [0.01, 0.1, 1, 10, 100]},
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search, y_test, search.best_estimator_.predict(x_test)


def main():
    search, y_test, predictions = train_and_predict()

    best_index = search.best_index_
    cv_mse = -search.cv_results_["mean_test_score"][best_index]
    cv_mse_std = search.cv_results_["std_test_score"][best_index]

    print(f"Лучший alpha: {search.best_params_['regressor__alpha']}")
    print(f"CV MSE: {cv_mse:.3f} ± {cv_mse_std:.3f}")
    print(f"Test MSE: {mean_squared_error(y_test, predictions):.3f}")
    print(f"Test MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Test R²: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
