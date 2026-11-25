# Лабораторная работа 4. Распознавание сущностей в медицинском тексте

Учебная система находит восемь типов сущностей в синтетических русскоязычных записях и заменяет их маркерами. Оценка регулярных выражений, SciBERT и RuBERT приведена в [отчёте](reports/lab04-report.md). Ограничения, ручная проверка и риски описаны в [паспорте модели](reports/model-card.md). Схема и происхождение данных приведены в [паспорте корпуса](data/synthetic/README.md).

## Быстрая проверка

Команды выполняются из каталога `lab_4`. Генерация корпуса, маскирование и анализ сохранённых предсказаний используют стандартную библиотеку Python.

```bash
python3 src/data_checks.py
python3 -m experiments.e0
python3 -m experiments.error_analysis
unzip -p experiments/e1_rubert_results.zip test-predictions.jsonl |
  python3 -m src.masking data/synthetic/test.jsonl --output reports/e1-rubert-test-masked.csv
```

Для полного набора тестов создайте окружение с Python 3.12 или новее и установите зависимости.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
python -m pip install "transformers==5.0.0" "tokenizers==0.22.2" pytest
python -m pytest -q test
```

## Эксперименты

Откройте ноутбук в Google Colab, выберите среду с GPU и выполните ячейки по порядку. Загружайте только синтетические файлы из репозитория.

| Эксперимент | Ноутбук | Вход |
| --- | --- | --- |
| SciBERT | [e0_scibert.ipynb](experiments/e0_scibert.ipynb) | `experiments/e0_scibert_data.zip` |
| RuBERT | [e1_rubert.ipynb](experiments/e1_rubert.ipynb) | `experiments/e1_rubert_data.zip` |

Ноутбук SciBERT в конце выгружает архив с контрольной точкой выбранной эпохи, токенизатором и результатами. SHA-256 исходного архива `e0_scibert_results_полное.zip` — `f6469f023ef5b000de013a2165e7ec8f054e5f1c6ebf1f68c51e127849c521a8`. Контрольную точку можно получить повторным запуском ноутбука. Совпадение SHA-256 относится к исходному архиву. Для проверки метрик и галереи доступны небольшие файлы [SciBERT](reports/e0-scibert-validation.json) и [RuBERT](experiments/e1_rubert_results.zip).
