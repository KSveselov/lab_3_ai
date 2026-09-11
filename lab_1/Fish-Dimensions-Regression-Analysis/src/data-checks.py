from pathlib import Path
import pandas as pd


def main():
    DATA_PATH = Path("../assets/data/Fish.csv")
    df = pd.read_csv(DATA_PATH)

    assert not df.empty
    assert df.columns.is_unique
    assert len(df) == len(df.drop_duplicates()) # либо описать дубликаты

    quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing": df.isna().sum(),
    "unique": df.nunique(dropna=False),
    })

    check_data(df)
    check_consistency(df)

    print(quality)

def check_data(df):
    size = df.shape
    print("Число строк:", size[0], "Число столбцов:", size[1])
    print("Число дубликатов:", df.duplicated().sum())
    print("Число пропущенных значений:", df.isna().sum().sum())
    print("Число бесконечных значений:", (df == float("inf")).sum().sum())
    print("Число отрицательных значений:", (df < 0).sum().sum())
    print("Число нулевых значений:", (df == 0).sum().sum())

def check_consistency(df):
    



if __name__ == "__main__":
    main()

