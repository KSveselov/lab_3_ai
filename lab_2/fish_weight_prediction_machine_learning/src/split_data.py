import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import DATA_PATH, DATA_SPLIT

SEED = 42

df = pd.read_csv(DATA_PATH)

# Выполняется один раз и сохраняется в split_indices.json.
train_index, test_index = train_test_split(
    df.index,
    test_size=0.2,
    random_state=SEED,
    stratify=df["Species"],
)

with open("split_indices.json", "w", encoding="utf-8") as file:
    json.dump(
        {"train_index": train_index.tolist(), "test_index": test_index.tolist()},
        file,
        indent=2,
    )

def load_data():
    f = pd.read_csv(DATA_PATH)

    with DATA_SPLIT.open(encoding="utf-8") as file:
        split_indices = json.load(file)

    train_df = df.loc[split_indices["train_index"]].copy()
    test_df = df.loc[split_indices["test_index"]].copy()

    return train_df, test_df
    