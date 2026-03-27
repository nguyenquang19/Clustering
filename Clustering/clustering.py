import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. TẢI DỮ LIỆU
df = pd.read_csv(r'D:/CODING_DATA/cs-training.csv')
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. CHUẨN HÓA TÊN CỘT
bang_doi_ten = {
    'SeriousDlqin2yrs': 'Vo_No',
    'RevolvingUtilizationOfUnsecuredLines': 'Ty_Le_Su_Dung',
    'age': 'Tuoi',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Tre_Han_30_59',
    'DebtRatio': 'Ty_Le_No',
    'MonthlyIncome': 'Thu_Nhap',
    'NumberOfOpenCreditLinesAndLoans': 'So_Khoan_Tin_Dung',
    'NumberOfTimes90DaysLate': 'Tre_Han_Tren_90',
    'NumberRealEstateLoansOrLines': 'Vay_BDS',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Tre_Han_60_89',
    'NumberOfDependents': 'Nguoi_Phu_Thuoc'
}
df = df.rename(columns=bang_doi_ten)

# 3. XỬ LÝ GIÁ TRỊ THIẾU (ADVANCED)
# A. Tạo biến cờ (Missing Indicator) - Lưu lại thông tin về việc dữ liệu bị thiếu
df['Thu_Nhap_Bi_Thieu'] = df['Thu_Nhap'].isnull().astype(int)
df['Phu_Thuoc_Bi_Thieu'] = df['Nguoi_Phu_Thuoc'].isnull().astype(int)

# B. Điền "Người phụ thuộc" theo Nhóm Tuổi (Groupby Imputation)
# Tạo nhóm tuổi: Thanh niên, Trung niên, Cao tuổi
df['Nhom_Tuoi'] = pd.cut(df['Tuoi'], bins=[0, 30, 45, 60, 150], labels=[1, 2, 3, 4])
df['Nguoi_Phu_Thuoc'] = df.groupby('Nhom_Tuoi')['Nguoi_Phu_Thuoc'].transform(lambda x: x.fillna(x.median()))
df = df.drop(columns=['Nhom_Tuoi'])

# C. Điền "Thu nhập" bằng Iterative Imputer (Machine Learning Imputation)
# Nó sẽ dùng Tuổi, Tỷ lệ nợ, và Số khoản vay để dự đoán mức thu nhập hợp lý nhất
imputer = IterativeImputer(random_state=42, max_iter=10)
cols_to_use = ['Tuoi', 'Ty_Le_No', 'Thu_Nhap', 'So_Khoan_Tin_Dung', 'Vay_BDS']
df[cols_to_use] = imputer.fit_transform(df[cols_to_use])

# 4. XỬ LÝ NGOẠI LAI (WINSORIZATION - CAPPING)
# Thay vì xóa dòng, ta "chặn" giá trị ở mức 1.5 * IQR để giữ lại dữ liệu
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return series.clip(lower, upper)

for col in ['Ty_Le_Su_Dung', 'Ty_Le_No', 'Thu_Nhap']:
    df[col] = cap_outliers(df[col])

# 5. CHUẨN HÓA (STANDARD SCALER)
features = df.drop(columns=['Vo_No'])
target = df['Vo_No']

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
df_scaled['Vo_No'] = target.values

# 6. LÀM TRÒN VÀ XỬ LÝ TRÙNG LẶP TRIỆT ĐỂ
# Làm tròn 3 chữ số giúp gom các dòng có sai số float cực nhỏ lại với nhau
df_scaled = df_scaled.round(3)
truoc_xoa = len(df_scaled)
df_scaled = df_scaled.drop_duplicates()
sau_xoa = len(df_scaled)

# 7. XUẤT FILE
df_scaled.to_excel('Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx', index=False)

print(f"Số lượng dòng ban đầu: {len(df)}")
print(f"Số lượng dòng cuối cùng: {sau_xoa}")
print(f"Số dòng trùng lặp thực sự đã xóa: {truoc_xoa - sau_xoa}")