import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, DATA_SPLIT, SEED


def create_split():
    data_frame = pd.read_csv(DATA_PATH)
    train_index, test_index = train_test_split(
        data_frame.index,
        test_size=0.2,
        random_state=SEED,
        stratify=data_frame["Species"],
    )

    with DATA_SPLIT.open("w", encoding="utf-8") as file:
        json.dump(
            {"train_index": train_index.tolist(), "test_index": test_index.tolist()},
            file,
            indent=2,
        )


def load_data():
    data_frame = pd.read_csv(DATA_PATH)

    with DATA_SPLIT.open(encoding="utf-8") as file:
        split_indices = json.load(file)

    train_df = data_frame.loc[split_indices["train_index"]].copy()
    test_df = data_frame.loc[split_indices["test_index"]].copy()

    return train_df, test_df


if __name__ == "__main__":
    create_split()
    
