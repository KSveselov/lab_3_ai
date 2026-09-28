
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from split_data import load_data
from config import GRAF_PATH, TARGET_COLUMN, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS



def prepare():
    numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
    ])
    return preprocess

def main():
    train_df, test_df = load_data()

    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    preprocess = prepare()

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ])

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    residuals = y_test - predictions

    print(f"MSE: {mean_squared_error(y_test, predictions):.3f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"R²: {r2_score(y_test, predictions):.3f}")

    print("\nКорреляции длин:")
    print(train_df[["Length1", "Length2", "Length3"]].corr())

    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Предсказанная масса, г")
    plt.ylabel("Остаток: фактическая масса − прогноз, г")
    plt.title("Остатки линейной регрессии")
    plt.savefig(GRAF_PATH / "e1_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()