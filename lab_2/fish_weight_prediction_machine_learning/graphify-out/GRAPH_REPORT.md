# Graph Report - fish_weight_prediction_machine_learning  (2026-10-01)

## Corpus Check
- Corpus is ~18,321 words - fits in a single context window. You may not need a graph.

## Summary
- 61 nodes · 140 edges · 8 communities (4 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Загрузка и линейные модели
- Предобработка и предсказание
- Деревья и пайплайны
- Конфигурация модели
- Интерфейс MLflow
- Логирование MLflow
- Паспорт лучшей модели

## God Nodes (most connected - your core abstractions)
1. `load_data()` - 13 edges
2. `prepare()` - 9 edges
3. `train_and_predict()` - 8 edges
4. `main()` - 7 edges
5. `train_and_predict()` - 6 edges
6. `train_and_predict()` - 6 edges
7. `train_and_predict()` - 6 edges
8. `log_model()` - 6 edges
9. `prep_data()` - 4 edges
10. `add_vproxy()` - 4 edges

## Surprising Connections (you probably didn't know these)
- `train_and_predict()` --calls--> `prepare()`  [EXTRACTED]
  src/e2.py → src/e1.py
- `train_and_predict()` --calls--> `load_data()`  [EXTRACTED]
  src/e1.py → src/split_data.py
- `main()` --calls--> `train_and_predict()`  [EXTRACTED]
  src/run_mlflow.py → src/e1.py
- `main()` --calls--> `train_and_predict()`  [EXTRACTED]
  src/run_mlflow.py → src/e2.py
- `train_and_predict()` --calls--> `load_data()`  [EXTRACTED]
  src/e3.py → src/split_data.py

## Import Cycles
- None detected.

## Communities (8 total, 4 thin omitted)

### Community 0 - "Загрузка и линейные модели"
Cohesion: 0.24
Nodes (5): main(), train_and_predict(), main(), train_and_predict(), load_data()

### Community 1 - "Предобработка и предсказание"
Cohesion: 0.29
Nodes (3): predict_from_csv(), handle_outlier(), prep_data()

### Community 3 - "Деревья и пайплайны"
Cohesion: 0.35
Nodes (7): prepare(), main(), train_and_predict(), add_vproxy(), main(), train_and_predict(), _train_and_predict()

### Community 6 - "Логирование MLflow"
Cohesion: 0.67
Nodes (3): get_git_commit(), log_evaluation_artifacts(), log_model()

## Knowledge Gaps
- **2 isolated node(s):** `run_mlflow_ui.sh script`, `PATH`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 16 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_data()` connect `Загрузка и линейные модели` to `Зависимости экспериментов`, `Деревья и пайплайны`, `Конфигурация модели`, `Логирование MLflow`?**
  _High betweenness centrality (0.075) - this node is a cross-community bridge._
- **Why does `train_and_predict()` connect `Деревья и пайплайны` to `Загрузка и линейные модели`, `Зависимости экспериментов`, `Паспорт лучшей модели`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `prepare()` connect `Деревья и пайплайны` to `Загрузка и линейные модели`, `Конфигурация модели`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **What connects `run_mlflow_ui.sh script`, `PATH` to the rest of the system?**
  _2 weakly-connected nodes found - possible documentation gaps or missing edges._