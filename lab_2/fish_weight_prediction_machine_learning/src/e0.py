from split_data import load_data

train_df, _ = load_data()
mean_prediction = train_df["Weight"].mean()
median_prediction = train_df["Weight"].median()

print("Средняя:", mean_prediction, "Медиана:", median_prediction)