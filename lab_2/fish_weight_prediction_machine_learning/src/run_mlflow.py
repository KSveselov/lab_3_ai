from uuid import uuid4

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

from e1 import train_and_predict as train_linear
from e2 import train_and_predict as train_ridge
from e3 import train_and_predict as train_forest
from e4 import TARGET_COLUMN, train_and_predict as train_vproxy


def log_model(name, model, y_test, prediction, config):
    with mlflow.start_run(run_name="{}-{}".format(name, uuid4().hex[:8])):
        mlflow.log_params(config)
        mlflow.log_metrics({
            "test_mse": mean_squared_error(y_test, prediction),
            "test_mae": mean_absolute_error(y_test, prediction),
            "test_r2": r2_score(y_test, prediction),
        })
        mlflow.sklearn.log_model(model, artifact_path="model")


def main():
    model, _, y_test, prediction = train_linear()
    log_model("e1-linear", model, y_test, prediction, {"model": "LinearRegression"})

    search, y_test, prediction = train_ridge()
    log_model("e2-ridge", search.best_estimator_, y_test, prediction, search.best_params_)

    search, y_test, prediction = train_forest()
    log_model("e3-forest", search.best_estimator_, y_test, prediction, search.best_params_)

    for label, test_df, search, prediction in train_vproxy():
        log_model(
            "e4-{}".format("vproxy" if "Vproxy" in label else "baseline"),
            search.best_estimator_,
            test_df[TARGET_COLUMN],
            prediction,
            search.best_params_,
        )


if __name__ == "__main__":
    main()
