"""Дополняет документами успешные запуски, не переобучая модели."""

import mlflow

from config import PROJECT_DIR
from model_passport import log_run_passport


def main():
    # Тот же журнал, который открывает run_mlflow_ui.sh.
    mlflow.set_tracking_uri((PROJECT_DIR / "src" / "mlruns").as_uri())
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Прогноз массы рыбы")
    if experiment is None:
        raise RuntimeError("Эксперимент «Прогноз массы рыбы» не найден")
    updated = 0
    failures = []
    for run in client.search_runs([experiment.experiment_id], max_results=10000):
        if run.info.status != "FINISHED":
            continue
        if not any(item.path == "model" for item in client.list_artifacts(run.info.run_id)):
            continue
        try:
            log_run_passport(run.info.run_id)
        except Exception as error:
            failures.append((run.info.run_id, str(error)))
            continue
        updated += 1
    print("Паспорта и ответы добавлены к {} запускам.".format(updated))
    if failures:
        for run_id, error in failures:
            print("Не удалось обновить {}: {}".format(run_id, error))
        raise RuntimeError("Не обновлено запусков: {}".format(len(failures)))


if __name__ == "__main__":
    main()
