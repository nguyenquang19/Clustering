import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=== BƯỚC 3: KỸ NGHỆ ĐẶC TRƯNG, CHUẨN HÓA VÀ PCA ===")

# ---------------------------------------------------------
# 1. KẾ THỪA DỮ LIỆU TỪ BƯỚC 2
# ---------------------------------------------------------
# Lưu ý: Kiểm tra kỹ đường dẫn file Excel của bạn
duong_dan_file = r'D:/Microsoft VS Code/Du_Lieu_Tin_Dung_Da_Xu_Ly_Full.xlsx'

try:
    du_lieu = pd.read_excel(duong_dan_file)
    print(f"-> Đã tải thành công dữ liệu sạch: {du_lieu.shape[0]} dòng.")
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file tại {duong_dan_file}. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# Hiển thị danh sách cột để chắc chắn máy tính đọc đúng
# print("Các cột đang có:", du_lieu.columns.tolist())

# ---------------------------------------------------------
# 2. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING) - DÙNG TÊN CỘT MỚI
# ---------------------------------------------------------
# Tạo biến 1: Tổng số lần trễ hạn (Sửa tên cột thành không dấu)
du_lieu['Tong_So_Lan_Tre_Han'] = (
    du_lieu['Tre_Han_30_59'] + 
    du_lieu['Tre_Han_60_89'] + 
    du_lieu['Tre_Han_90']
)

# Tạo biến 2: Thu nhập bình quân (Dùng Thu_Nhap và Nguoi_Phu_Thuoc)
du_lieu['Thu_Nhap_Binh_Quan'] = du_lieu['Thu_Nhap'] / (du_lieu['Nguoi_Phu_Thuoc'] + 1)

print("-> Đã tạo thêm 2 đặc trưng mới: Tong_So_Lan_Tre_Han, Thu_Nhap_Binh_Quan.")

# ---------------------------------------------------------
# 3. TẠO MA TRẬN ĐẶC TRƯNG & CHUẨN HÓA (STANDARD SCALER)
# ---------------------------------------------------------
# Tách riêng nhãn 'Vo_No'
nhan = du_lieu['Vo_No'].values
dac_trung = du_lieu.drop(columns=['Vo_No'])

bo_chuan_hoa = StandardScaler()
ma_tran_dac_trung_da_chuan_hoa = bo_chuan_hoa.fit_transform(dac_trung)

# Lưu bộ dữ liệu Full (đã scale)
bang_full_san_sang = pd.DataFrame(ma_tran_dac_trung_da_chuan_hoa, columns=dac_trung.columns).round(3)
bang_full_san_sang['Vo_No'] = nhan

# Xuất ra CSV để dùng cho bước Phân cụm (Clustering)
bang_full_san_sang.to_csv('Du_Lieu_Full_San_Sang_Clustering.csv', index=False, encoding="utf-8-sig")
print("-> Đã xuất file chuẩn hóa: Du_Lieu_Full_San_Sang_Clustering.csv")

# ---------------------------------------------------------
# 4. GIẢM CHIỀU DỮ LIỆU BẰNG PCA (ĐỂ VẼ ĐỒ THỊ)
# ---------------------------------------------------------
pca = PCA(n_components=2)
ma_tran_pca = pca.fit_transform(ma_tran_dac_trung_da_chuan_hoa)

ty_le_thong_tin = pca.explained_variance_ratio_
print(f"-> Khả năng bảo toàn thông tin của PCA: {sum(ty_le_thong_tin)*100:.2f}%")

# Lưu kết quả PCA 2 chiều để vẽ đồ thị
bang_pca = pd.DataFrame(ma_tran_pca, columns=['PC1', 'PC2']).round(3)
bang_pca['Vo_No'] = nhan

bang_pca.to_csv('Du_Lieu_PCA_2D.csv', index=False, encoding="utf-8-sig")
print("-> Đã xuất file PCA 2 chiều: Du_Lieu_PCA_2D.csv")

print("\n=== HOÀN TẤT BƯỚC 3 ===")