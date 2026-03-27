import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. TẢI DỮ LIỆU
print("--- ĐANG TẢI VÀ XỬ LÝ DỮ LIỆU ---")
du_lieu = pd.read_csv(r'D:/CODING_DATA/cs-training.csv')

# Bỏ cột index thừa nếu có (Unnamed: 0)
if du_lieu.columns[0] == 'Unnamed: 0':
    du_lieu = du_lieu.drop(columns=['Unnamed: 0'])

# 2. CHUẨN HÓA TÊN CỘT VỀ TIẾNG VIỆT CÓ DẤU
bang_doi_ten = {
    'SeriousDlqin2yrs': 'Vỡ_Nợ_Trong_2_Năm',
    'RevolvingUtilizationOfUnsecuredLines': 'Tỷ_Lệ_Sử_Dụng_Hạn_Mức',
    'age': 'Tuổi',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Số_Lần_Trễ_Hạn_30_Đến_59_Ngày',
    'DebtRatio': 'Tỷ_Lệ_Nợ_Trên_Thu_Nhập',
    'MonthlyIncome': 'Thu_Nhập_Hàng_Tháng',
    'NumberOfOpenCreditLinesAndLoans': 'Số_Khoản_Tín_Dụng_Đang_Mở',
    'NumberOfTimes90DaysLate': 'Số_Lần_Trễ_Hạn_Trên_90_Ngày',
    'NumberRealEstateLoansOrLines': 'Số_Khoản_Vay_Bất_Động_Sản',
    'NumberOfTime60-89DaysPastDueNotWorse': 'Số_Lần_Trễ_Hạn_60_Đến_89_Ngày',
    'NumberOfDependents': 'Số_Người_Phụ_Thuộc'
}
du_lieu = du_lieu.rename(columns=bang_doi_ten)
print("-> Đã đổi tên cột sang Tiếng Việt có dấu.")

# 3. XÓA BẢN GHI TRÙNG LẶP (Lần 1 - Dọn rác dữ liệu thô)
so_dong_trung_lap_1 = du_lieu.duplicated().sum()
du_lieu = du_lieu.drop_duplicates(keep='first')
print(f"-> Đã xóa {so_dong_trung_lap_1} bản ghi trùng lặp ban đầu.")

# 4. XỬ LÝ GIÁ TRỊ THIẾU (Missing Values)
du_lieu['Thu_Nhập_Hàng_Tháng'] = du_lieu['Thu_Nhập_Hàng_Tháng'].fillna(du_lieu['Thu_Nhập_Hàng_Tháng'].median())
du_lieu['Số_Người_Phụ_Thuộc'] = du_lieu['Số_Người_Phụ_Thuộc'].fillna(du_lieu['Số_Người_Phụ_Thuộc'].median())

# XÓA BẢN GHI TRÙNG LẶP (Lần 2 - Xóa các dòng giống hệt nhau phát sinh sau khi điền Median)
so_dong_trung_lap_2 = du_lieu.duplicated().sum()
du_lieu = du_lieu.drop_duplicates(keep='first')
print(f"-> Đã dọn dẹp thêm {so_dong_trung_lap_2} bản ghi trùng lặp phát sinh do điền giá trị thiếu.")

# 5. LOẠI BỎ GIÁ TRỊ NGOẠI LAI (Outliers) BẰNG PHƯƠNG PHÁP IQR
cac_cot_can_loc = ['Tỷ_Lệ_Sử_Dụng_Hạn_Mức', 'Tỷ_Lệ_Nợ_Trên_Thu_Nhập', 'Thu_Nhập_Hàng_Tháng']
so_dong_truoc_khi_loc = du_lieu.shape[0]

for cot in cac_cot_can_loc:
    Q1 = du_lieu[cot].quantile(0.25)
    Q3 = du_lieu[cot].quantile(0.75)
    IQR = Q3 - Q1
    gioi_han_duoi = Q1 - 1.5 * IQR
    gioi_han_tren = Q3 + 1.5 * IQR
    du_lieu = du_lieu[(du_lieu[cot] >= gioi_han_duoi) & (du_lieu[cot] <= gioi_han_tren)]

so_dong_bi_loai_bo = so_dong_truoc_khi_loc - du_lieu.shape[0]
print(f"-> Đã xóa {so_dong_bi_loai_bo} bản ghi ngoại lai (Outliers).")

# 6. CHUẨN HÓA DỮ LIỆU BẰNG STANDARD SCALER
dac_trung = du_lieu.drop(columns=['Vỡ_Nợ_Trong_2_Năm'])
nhan = du_lieu['Vỡ_Nợ_Trong_2_Năm']

bo_chuan_hoa = StandardScaler()
du_lieu_da_chuan_hoa = bo_chuan_hoa.fit_transform(dac_trung)

bang_du_lieu_cuoi_cung = pd.DataFrame(du_lieu_da_chuan_hoa, columns=dac_trung.columns)

# 7. LÀM TRÒN TOÀN BỘ VỀ 3 CHỮ SỐ THẬP PHÂN
bang_du_lieu_cuoi_cung = bang_du_lieu_cuoi_cung.round(3)
print("-> Đã làm tròn toàn bộ dữ liệu về 3 chữ số thập phân.")

# Gắn lại nhãn vỡ nợ vào bảng (Nhãn không bị thay đổi vì chỉ là 0 và 1)
bang_du_lieu_cuoi_cung['Vỡ_Nợ_Trong_2_Năm'] = nhan.values

# 8. XUẤT RA FILE EXCEL
ten_file_xuat = 'Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx'
bang_du_lieu_cuoi_cung.to_excel(ten_file_xuat, index=False, engine='openpyxl')

print(f"\n--- KẾT QUẢ TỔNG KẾT ---")
print(f"Số lượng bản ghi cuối cùng đưa vào mô hình: {bang_du_lieu_cuoi_cung.shape[0]} dòng")
print(f"Đã xuất file thành công: {ten_file_xuat}")