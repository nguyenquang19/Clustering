import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Tải dữ liệu
df = pd.read_csv(r'D:/CODING_DATA/cs-training.csv')
if df.columns[0] == 'Unnamed: 0':
    df = df.drop(columns=['Unnamed: 0'])

# 2. Xử lý giá trị thiếu (Missing Values)
# Sử dụng Median vì thu nhập thường có phân phối lệch (skewed)
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

# 3. Loại bỏ giá trị ngoại lai (Outliers) bằng phương pháp IQR
# Chúng ta tập trung vào các biến có độ biến động cực lớn trong tài chính
cols_to_filter = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']

for col in cols_to_filter:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Loại bỏ các dòng nằm ngoài khoảng [lower, upper]
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 4. Chuẩn hóa dữ liệu bằng StandardScaler
# Tách nhãn (SeriousDlqin2yrs) để không chuẩn hóa nhãn này
features = df.drop(columns=['SeriousDlqin2yrs'])
labels = df['SeriousDlqin2yrs']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Chuyển đổi ngược lại thành DataFrame để dễ quan sát
df_processed = pd.DataFrame(X_scaled, columns=features.columns)
df_processed['SeriousDlqin2yrs'] = labels.values

# 5. Lưu dữ liệu đã xử lý
df_processed.to_csv('credit_processed.csv', index=False)

print(f"Số lượng bản ghi ban đầu: 150,000")
print(f"Số lượng bản ghi sau khi xử lý: {df_processed.shape[0]}")
print("\n5 dòng đầu tiên của dữ liệu đã chuẩn hóa:")
print(df_processed.head())