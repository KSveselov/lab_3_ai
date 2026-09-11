from pathlib import Path
import pandas as pd
import numpy as np

dimensions = ["Length1", "Length2", "Length3", "Height", "Width"]

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
    print(quality)
    check_data(df)
    check_consistency(df)
    check_disbalance(df)
    check_almost_identical_rows(df)


    

def check_data(df):
    size = df.shape
    numbers = df.select_dtypes(include=["number"])
    print("Число строк:", size[0], "Число столбцов:", size[1])
    print("Число дубликатов:", numbers.duplicated().sum())
    print("Число пропущенных значений:", numbers.isna().sum().sum())
    print("Число бесконечных значений:", (numbers == float("inf")).sum().sum())
    print("Число отрицательных значений:", (numbers < 0).sum().sum())
    print("Число нулевых значений:", (numbers == 0).sum().sum())

def check_consistency(df):
    

    non_positive = df[dimensions].le(0).any(axis=1)

    wrong_order = ~(
        (df["Length1"] <= df["Length2"]) 
        & (df["Length2"] <= df["Length3"])
    )

    print("Число строк с неположительными значениями в измерениях:", non_positive.sum())
    print("Число строк с неправильным порядком значений (Length1 ≤ Length2 ≤ Length3):", wrong_order.sum())

    if non_positive.any():
        print("Строки с неположительными значениями в измерениях:")
        print(df[non_positive])

    if wrong_order.any():
        print("Строки с неправильным порядком значений (Length1 ≤ Length2 ≤ Length3):")
        print(df[wrong_order])

def check_disbalance(df):
    species_counts = df["Species"].value_counts()
    print("Доля экземпляров каждого вида:")
    for species, count in species_counts.items():
        proportion = count / len(df)
        print(f"{species}: {proportion:.2%} ({count} экземпляров)")

    imbalance_rate = species_counts.max() / species_counts.min()
    print(f"Коэффициент дисбаланса: {imbalance_rate:.2f}")



def check_almost_identical_rows(df):

    close_rows = None
    for i  in range(len(df)):
        for j in range(i + 1, len(df)):
            close_rows = np.isclose(df.loc[i, dimensions].to_numpy(dtype=float), df.loc[j, dimensions].to_numpy(dtype=float), rtol=0.01, atol=0.01).all()

            if close_rows:
                print(f"Строки {i} и {j} почти идентичны:")
                print(df.loc[[i, j]])
                


if __name__ == "__main__":
    main()

