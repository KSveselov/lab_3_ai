import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.utils.validation import (
    _validate_metric_name,
    _validate_param_name,
    _validate_tag_name,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from e1 import train_and_predict
from error_analysis import log_error_analysis
from report_labels import REPORT_LABELS
from run_mlflow import log_best_model_passport, log_model
from model_passport import log_run_passport


class RussianReportsTest(unittest.TestCase):
    def test_log_model_passport_and_backfill_preserve_run(self):
        previous_uri = mlflow.get_tracking_uri()
        previous_experiment = mlflow.tracking.fluent._active_experiment_id
        try:
            with TemporaryDirectory() as directory:
                mlflow.set_tracking_uri(Path(directory).as_uri())
                mlflow.set_experiment("Проверка паспорта логарифмической модели")
                model, _, actual, prediction = train_and_predict(log_target=True)
                result = log_model("Э1-ln", model, actual, prediction, {})
                client = MlflowClient()
                before = client.get_run(result["run_id"])
                log_run_passport(result["run_id"])
                after = client.get_run(result["run_id"])
                self.assertEqual(before.info.status, after.info.status)
                self.assertEqual(before.data.metrics, after.data.metrics)
                path = client.download_artifacts(result["run_id"], "документация/паспорт_модели.md")
                passport = Path(path).read_text(encoding="utf-8")
                self.assertIn("exp(z)-1", passport)
                self.assertIn("сдвиг распределения", passport)
                self.assertIn("{:.6f}".format(result["metrics"]["test_mae"]), passport)
                loaded = mlflow.sklearn.load_model(after.info.artifact_uri + "/model")
                np.testing.assert_allclose(loaded.predict(model_input()), prediction)
        finally:
            mlflow.set_tracking_uri(previous_uri)
            mlflow.tracking.fluent._active_experiment_id = previous_experiment

    # Проверяет уникальность русских подписей, совпадение псевдонимов прогноза и допустимость подписей в MLflow.
    def test_labels_are_unique_and_accepted_by_mlflow(self):
        labels = {
            key: value for key, value in REPORT_LABELS.items() if key != "predicted"
        }
        self.assertEqual(len(labels), len(set(labels.values())))
        self.assertEqual(REPORT_LABELS["prediction"], REPORT_LABELS["predicted"])
        for label in REPORT_LABELS.values():
            _validate_metric_name(label)
            _validate_param_name(label)
            _validate_tag_name(label)

    # Проверяет, что запуск MLflow сохраняет русские отчёты и модель, воспроизводящую прогнозы после загрузки.
    def test_new_run_exports_russian_reports_and_loadable_model(self):
        previous_uri = mlflow.get_tracking_uri()
        previous_experiment = mlflow.tracking.fluent._active_experiment_id
        try:
            with TemporaryDirectory() as directory:
                mlflow.set_tracking_uri(Path(directory).as_uri())
                mlflow.set_experiment("Проверка русского отчёта")
                model, _, actual, prediction = train_and_predict()
                result = log_model(
                    "Э1 - Линейная регрессия",
                    model,
                    actual,
                    prediction,
                    {"model": "Линейная регрессия"},
                    cv_scores=np.array([-100.0, -200.0]),
                )
                # Паспорт нужен каждому запуску ещё до выбора лучшей модели.
                client = MlflowClient()
                documents = {item.path for item in client.list_artifacts(result["run_id"], "документация")}
                self.assertIn("документация/паспорт_модели.md", documents)
                self.assertIn("документация/вопросы_и_ответы.md", documents)
                log_best_model_passport([result])
                run = MlflowClient().get_run(result["run_id"])
                self.assertTrue(run.info.run_name.startswith("Э1 - Линейная регрессия"))
                self.assertAlmostEqual(
                    run.data.metrics[REPORT_LABELS["test_mae"]],
                    result["metrics"]["test_mae"],
                )
                self.assertEqual(run.data.tags["Лучшая модель"], "Да")
                self.assertTrue(all(not key.isascii() for key in run.data.metrics))
                self.assertTrue(all(not key.isascii() for key in run.data.params))
                artifacts = (
                    Path(directory)
                    / run.info.experiment_id
                    / result["run_id"]
                    / "artifacts"
                )
                report = pd.read_csv(
                    artifacts / "анализ_ошибок" / "прогнозы_на_тесте.csv"
                )
                self.assertIn("Вид рыбы", report)
                self.assertIn("Лещ", report["Вид рыбы"].values)
                np.testing.assert_allclose(report["Прогноз массы"], prediction)
                passport = (
                    artifacts / "документация" / "паспорт_модели.md"
                ).read_text()
                self.assertIn("Средняя абсолютная ошибка на тесте (MAE)", passport)
                self.assertIn("Средняя квадратичная ошибка на тесте (MSE)", passport)
                self.assertIn("Коэффициент детерминации на тесте (R²)", passport)
                self.assertNotIn("Test MAE", passport)
                loaded = mlflow.sklearn.load_model(str(artifacts / "model"))
                np.testing.assert_allclose(loaded.predict(model_input()), prediction)
        finally:
            mlflow.set_tracking_uri(previous_uri)
            mlflow.tracking.fluent._active_experiment_id = previous_experiment

    # Проверяет, что экспорт сохраняет исходные данные и внутренние ключи метрик, используя русские подписи в MLflow.
    def test_export_does_not_change_inputs_or_internal_metric_keys(self):
        from unittest.mock import patch
        from split_data import load_data

        train, test = load_data()
        original = test.copy(deep=True)
        with patch("error_analysis.mlflow.log_metrics") as logged, patch(
            "error_analysis.mlflow.set_tags"
        ), patch("error_analysis.mlflow.log_artifacts"):
            metrics = log_error_analysis(
                train, test, test["Weight"], test["Weight"].to_numpy() + 1.0
            )
        pd.testing.assert_frame_equal(test, original)
        self.assertIn("test_mae_species_bream", metrics)
        self.assertIn(REPORT_LABELS["test_mae_species_bream"], logged.call_args[0][0])
        self.assertEqual(metrics["test_mae_species_bream"], 1.0)


# Возвращает признаки тестовой выборки для проверки прогнозов загруженной модели.
def model_input():
    from config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
    from split_data import load_data

    _, test = load_data()
    return test[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]


if __name__ == "__main__":
    unittest.main()
