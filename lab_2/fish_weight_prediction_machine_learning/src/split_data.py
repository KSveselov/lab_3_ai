import json

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42

df = pd.read_csv("../fish_participant.csv")

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
