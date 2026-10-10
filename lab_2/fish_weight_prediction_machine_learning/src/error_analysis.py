import re
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from report_labels import REPORT_LABELS, SPECIES_LABELS


RAW_FEATURE_COLUMNS = [
    "Species",
    "Length1",
    "Length2",
    "Length3",
    "Height",
    "Width",
    "Vproxy",
]


def _metric_name(value):
    return re.sub(r"[^\w]", "_", str(value).lower()).strip("_")


def _size_ranges(lengths):
    try:
        return pd.qcut(lengths, q=3, duplicates="drop")
    except ValueError:
        return pd.Series("Все размеры", index=lengths.index)


def _systematic_zones(size_mae, overall_mae):
    zones = []
    for size_range, values in size_mae.iterrows():
        mean_residual = values["mean_residual"]
        if abs(mean_residual) < overall_mae:
            continue
        direction = "недооценивает" if mean_residual > 0 else "переоценивает"
        zones.append(
            "{}: модель {} массу в среднем на {:.1f} г".format(
                size_range, direction, abs(mean_residual)
            )
        )
    return zones


def _diagnose_error(species_mae, systematic_zones, invalid_predictions):
    if invalid_predictions:
        return (
            "Основная проблема — модель: найдены физически невозможные прогнозы. "
            "Нужны ограничение прогноза снизу и проверка выбросов."
        )

    if len(species_mae) > 1:
        mae_ratio = species_mae["mae"].max() / species_mae["mae"].min()
        if mae_ratio >= 1.5:
            return (
                "Ошибка зависит от вида; вероятнее всего, общая модель недостаточно "
                "учитывает различия формы рыб."
            )

    if systematic_zones:
        return (
            "Ошибка меняется с размером рыбы; вероятнее всего, модели не хватает "
            "нелинейности или признаков, описывающих объём."
        )

    return (
        "На тестовой выборке нет ярко выраженной систематической зоны. "
        "Крупные ошибки могут быть связаны с редкими наблюдениями или шумом данных."
    )


def log_error_analysis(train_df, test_df, y_test, prediction):
    """Сохраняет обязательный анализ ошибок текущей модели в активный MLflow run."""
    feature_columns = [column for column in RAW_FEATURE_COLUMNS if column in test_df]
    results = test_df.loc[y_test.index, feature_columns].copy()
    results.insert(0, "row_id", y_test.index)
    results["actual"] = y_test.to_numpy()
    results["prediction"] = np.asarray(prediction)
    results["residual"] = results["actual"] - results["prediction"]
    results["absolute_error"] = results["residual"].abs()
    results["size_range"] = _size_ranges(results["Length3"])

    overall_mae = mean_absolute_error(results["actual"], results["prediction"])
    species_mae = results.groupby("Species", observed=True).agg(
        observations=("actual", "size"),
        mae=("absolute_error", "mean"),
        mean_residual=("residual", "mean"),
    )
    size_mae = results.groupby("size_range", observed=True).agg(
        observations=("actual", "size"),
        mae=("absolute_error", "mean"),
        mean_residual=("residual", "mean"),
    )
    systematic_zones = _systematic_zones(size_mae, overall_mae)

    non_finite = int((~np.isfinite(results["prediction"])).sum())
    negative = int((results["prediction"] < 0).sum())
    non_positive = int((results["prediction"] <= 0).sum())
    train_min = train_df["Weight"].min()
    train_max = train_df["Weight"].max()
    outside_train_range = int(
        (
            (results["prediction"] < train_min) | (results["prediction"] > train_max)
        ).sum()
    )
    invalid_predictions = non_finite + non_positive

    metrics = {
        "error_negative_predictions": float(negative),
        "error_non_positive_predictions": float(non_positive),
        "error_non_finite_predictions": float(non_finite),
        "error_outside_train_weight_range": float(outside_train_range),
    }
    for species, values in species_mae.iterrows():
        metrics["test_mae_species_{}".format(_metric_name(species))] = values["mae"]
    for index, (_, values) in enumerate(size_mae.iterrows(), start=1):
        metrics["test_mae_size_group_{}".format(index)] = values["mae"]
    mlflow.log_metrics(
        {REPORT_LABELS.get(key, key): value for key, value in metrics.items()}
    )

    diagnosis = _diagnose_error(species_mae, systematic_zones, invalid_predictions)
    next_experiment = (
        "Обучить отдельные модели для каждого вида и сравнить их с общей моделью "
        "по средней абсолютной ошибке. Ожидаемый результат: ошибка снизится для видов с наибольшей ошибкой."
    )
    mlflow.set_tags(
        {
            REPORT_LABELS["error_analysis_diagnosis"]: diagnosis,
            REPORT_LABELS["next_experiment"]: next_experiment,
        }
    )

    report_results = results.assign(
        Species=results["Species"].replace(SPECIES_LABELS)
    ).rename(columns=REPORT_LABELS)
    top_errors = report_results.nlargest(5, REPORT_LABELS["absolute_error"])
    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        report_results.to_csv(directory_path / "прогнозы_на_тесте.csv", index=False)
        top_errors.to_csv(directory_path / "пять_крупнейших_ошибок.csv", index=False)
        species_mae.rename(index=SPECIES_LABELS, columns=REPORT_LABELS).rename_axis(
            REPORT_LABELS["Species"]
        ).to_csv(directory_path / "ошибка_по_видам.csv")
        size_mae.rename(columns=REPORT_LABELS).rename_axis(
            REPORT_LABELS["size_range"]
        ).to_csv(directory_path / "ошибка_по_размеру.csv")

        fig, ax = plt.subplots()
        ax.scatter(results["actual"], results["residual"])
        ax.axhline(0, color="red", linestyle="--")
        ax.set(
            xlabel="Фактическая масса, г",
            ylabel="Остаток: фактическая масса − прогноз, г",
            title="Остатки модели по фактической массе",
        )
        fig.savefig(
            directory_path / "остатки_по_фактической_массе.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.scatter(results["actual"], results["prediction"])
        lower = min(results["actual"].min(), results["prediction"].min())
        upper = max(results["actual"].max(), results["prediction"].max())
        ax.plot([lower, upper], [lower, upper], color="red", linestyle="--")
        ax.set(xlabel="Фактическая масса, г", ylabel="Прогноз, г")
        fig.savefig(
            directory_path / "фактическая_масса_и_прогноз.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        report = [
            "# Анализ ошибок модели",
            "",
            "## Систематические зоны ошибки",
            "",
            *(
                systematic_zones
                or [
                    "Смещение больше общей средней абсолютной ошибки ни в одном диапазоне размера не найдено."
                ]
            ),
            "",
            "## Проверка невозможных прогнозов",
            "",
            "- Отрицательных прогнозов: {}".format(negative),
            "- Неположительных прогнозов: {}".format(non_positive),
            "- Неконечных прогнозов: {}".format(non_finite),
            "- За пределами диапазона массы обучающей выборки [{:.1f}; {:.1f}] г: {}".format(
                train_min, train_max, outside_train_range
            ),
            "",
            "## Вывод о причине ошибок",
            "",
            diagnosis,
            "",
            "## Следующий эксперимент",
            "",
            next_experiment,
        ]
        (directory_path / "анализ_ошибок.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        mlflow.log_artifacts(str(directory_path), artifact_path="анализ_ошибок")

    return metrics
