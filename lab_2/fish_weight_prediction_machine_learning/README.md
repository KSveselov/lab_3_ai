# Fish Weight Prediction Model:

![Fish](images/Fishlength.jpg)

## Requirements

Hello! Welcome to the famous Tsukiji fish market of Tokyo, Japan! We came here to collect data on some of the fish they have here 
but we didn't wake up at 5am for the tuna auction and by the time we showed up they were only left with a few species of fish. 
We got to work and gathered measurements from a few different species of fish and want you to train a regression model to predict
the weight of a fish using some of the features we were able to measure. We have no idea which features will be good predictors. 

We will hold out 30% of the data before we hand it to you and we will use that csv for scoring.

Here's what we need from you:
1. A function that accepts a csv path and returns the predictions of your regression model using our csv. The csv we use will contain all the columns. 
2. Use a pipenv and scikit learn to submit the final model. You may use R for model selection. 

With this function and it's output, we will rank the students by how well their model performed on predicting weight based on naive data. 
Your grade will be determined by ranking according to Mean-squared Error.
If your function does not return a list of predictions or we cannot compute the accuracy of your model that it will be an automatic F. 

## Used

*   Linear Regression
*   Lasso Regression
*   Random Forest
*   Scikit Learn
*   Python
*   Joblib
*   Matplotlib
*   Virtualenv

## Запуск проекта

Все команды выполняются из корня проекта.

### Установка зависимостей

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Подготовка фиксированного разбиения данных

```bash
python src/split_data.py
```

### Обучение моделей и запись результатов в MLflow

```bash
python src/run_mlflow.py
```

### Открытие локального интерфейса MLflow

```bash
mlflow ui --backend-store-uri "$(pwd)/mlruns" --host 127.0.0.1 --port 5000
```

После запуска откройте http://127.0.0.1:5000. Обучение и UI используют одну
папку `mlruns/`.

### Запуск тестов

```bash
python -m pip install "pytest<8"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_pipeline.py
```
