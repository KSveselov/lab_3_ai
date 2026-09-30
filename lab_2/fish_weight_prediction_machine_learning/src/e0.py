from split_data import load_data

def train_and_predict():
    train_df, _ = load_data()
    mean_prediction = train_df["Weight"].mean()
    median_prediction = train_df["Weight"].median()
    return mean_prediction, median_prediction


def main():
    mean_prediction, median_prediction = train_and_predict()

    print("Средняя:", mean_prediction, "Медиана:", median_prediction)


if __name__ == "__main__":
    main()
