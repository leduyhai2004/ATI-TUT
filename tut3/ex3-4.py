import pandas as pd

# Load Titanic dataset
df = pd.read_csv("titanic_all_numeric.csv")

# Xem tổng quan dữ liệu
print(df.head())
print(df.describe())

# Lấy giá trị tuổi lớn nhất
max_age = df["age"].max()
print("Maximum age of passengers:", max_age)
