import platform
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from e1 import train_and_predict as train_linear
from e2 import train_and_predict as train_ridge
from e3 import train_and_predict as train_forest
from e4 import add_vproxy, train_and_predict as train_vproxy
from config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROJECT_DIR,
    SEED,
    TARGET_COLUMN,
)
from error_analysis import log_error_analysis
from split_data import load_data
from report_labels import REPORT_LABELS


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_DIR), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def log_model(name, model, y_test, prediction, config, search=None, cv_scores=None):
    with mlflow.start_run(run_name="{}-{}".format(name, uuid4().hex[:8])) as run:
        train_df, test_df = load_data()
        data_len = len(test_df) + len(train_df)
        features = [
            column
            for _, _, columns in model.named_steps["preprocess"].transformers
            for column in columns
        ]

        config.update(
            {
                "train_data": len(train_df) / data_len * 100,
                "test_data": len(test_df) / data_len * 100,
                "features": " | ".join(
                    REPORT_LABELS.get(column, column) for column in features
                ),
                "seed": SEED,
            }
        )
        mlflow.log_params(
            {REPORT_LABELS.get(key, key): value for key, value in config.items()}
        )
        metrics = {
            "test_mse": mean_squared_error(y_test, prediction),
            "test_mae": mean_absolute_error(y_test, prediction),
            "test_r2": r2_score(y_test, prediction),
        }
        if search is not None:
            best_index = search.best_index_
            metrics.update(
                {
                    "cv_mse_mean": -search.cv_results_["mean_test_score"][best_index],
                    "cv_mse_std": search.cv_results_["std_test_score"][best_index],
                }
            )
        elif cv_scores is not None:
            metrics.update(
                {
                    "cv_mse_mean": -cv_scores.mean(),
                    "cv_mse_std": cv_scores.std(),
                }
            )
        mlflow.log_metrics(
            {REPORT_LABELS.get(key, key): value for key, value in metrics.items()}
        )

        mlflow.set_tags(
            {
                REPORT_LABELS["git_commit"]: get_git_commit(),
                REPORT_LABELS["python_version"]: platform.python_version(),
                REPORT_LABELS["mlflow_version"]: mlflow.__version__,
                REPORT_LABELS["sklearn_version"]: sklearn.__version__,
            }
        )
        if "Vproxy" in features:
            test_df = add_vproxy(test_df)
        log_error_analysis(train_df, test_df, y_test, prediction)
        input_example = test_df[features].iloc[:5]
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
    print(name)
    return {
        "run_id": run.info.run_id,
        "name": name,
        "metrics": metrics,
        "features": features,
    }


def log_best_model_passport(runs):
    best = min(runs, key=lambda item: item["metrics"]["cv_mse_mean"])
    metrics = best["metrics"]
    passport = "\n".join(
        [
            "# Паспорт модели",
            "",
            "Модель: {}".format(best["name"]),
            "Идентификатор запуска MLflow: {}".format(best["run_id"]),
            "Задача: прогноз массы рыбы по измерениям и виду.",
            "Входные признаки: {}".format(
                ", ".join(
                    REPORT_LABELS.get(column, column) for column in best["features"]
                )
            ),
            "Предобработка и регрессор сохранены вместе в артефакте model/.",
            "Критерий выбора: минимальная средняя квадратичная ошибка кросс-валидации (MSE) на обучающей выборке.",
            "Средняя квадратичная ошибка кросс-валидации (MSE): {:.3f} ± {:.3f}".format(
                metrics["cv_mse_mean"], metrics["cv_mse_std"]
            ),
            "Средняя квадратичная ошибка на тесте (MSE): {:.3f}".format(
                metrics["test_mse"]
            ),
            "Средняя абсолютная ошибка на тесте (MAE): {:.3f}".format(
                metrics["test_mae"]
            ),
            "Коэффициент детерминации на тесте (R²): {:.3f}".format(metrics["test_r2"]),
            "Ограничение: качество оценено на имеющейся тестовой выборке; "
            "перенос на другие данные отдельно не проверялся.",
            "",
        ]
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "паспорт_модели.md"
        path.write_text(passport, encoding="utf-8")
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.set_tag(REPORT_LABELS["best_model"], "Да")
            mlflow.log_artifact(str(path), artifact_path="документация")
    print("Лучшая модель: {} ({})".format(best["name"], best["run_id"]))


def main():
    mlflow.set_experiment("Прогноз массы рыбы")
    runs = []
    model, train_df, y_test, prediction = train_linear()
    cv_scores = cross_val_score(
        model,
        train_df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS],
        train_df[TARGET_COLUMN],
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
    )
    runs.append(
        log_model(
            "Э1 - Линейная регрессия",
            model,
            y_test,
            prediction,
            {"model": "Линейная регрессия"},
            cv_scores=cv_scores,
        )
    )

    search, y_test, prediction = train_ridge()
    runs.append(
        log_model(
            "Э2 - Регрессия с регуляризацией",
            search.best_estimator_,
            y_test,
            prediction,
            search.best_params_,
            search,
        )
    )

    search, y_test, prediction = train_forest()
    runs.append(
        log_model(
            "Э3 - Случайный лес",
            search.best_estimator_,
            y_test,
            prediction,
            search.best_params_,
            search,
        )
    )

    for label, test_df, search, prediction in train_vproxy():
        runs.append(
            log_model(
                (
                    "Э4-Б - Случайный лес с приблизительным объёмом"
                    if "Vproxy" in label
                    else "Э4-А - Случайный лес с базовыми признаками"
                ),
                search.best_estimator_,
                test_df[TARGET_COLUMN],
                prediction,
                search.best_params_,
                search,
            )
        )

    log_best_model_passport(runs)


if __name__ == "__main__":
    main()
