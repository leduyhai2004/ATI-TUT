# Load dữ liệu
df = pd.read_csv("/content/hourly_wages.csv")


# Xem 5 dòng đầu tiên
print("5 dòng đầu tiên:")
print(df.head())

# Thống kê mô tả
print("\nMô tả dữ liệu:")
print(df.describe())

# Kiểm tra số lượng biến nhị phân (min=0 và max=1)
binary_cols = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:  # chỉ check cột số
        min_val, max_val = df[col].min(), df[col].max()
        if min_val == 0 and max_val == 1:
            binary_cols.append(col)

print("\nCác biến nhị phân:", binary_cols)
print("Số lượng biến nhị phân:", len(binary_cols))