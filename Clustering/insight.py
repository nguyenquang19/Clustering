import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as shc
import warnings
warnings.filterwarnings('ignore') # Ẩn các cảnh báo đỏ cho code sạch đẹp

print("=== VẼ BIỂU ĐỒ HỌC THUẬT & PHÂN CỤM (TỪ FILE ĐÃ XỬ LÝ) ===")

# =========================================================
# PHẦN 1: TẢI DỮ LIỆU "SẴN SÀNG" & LẤY MẪU
# =========================================================
try:
    df_full = pd.read_csv(r'D:/CODING_DATA/Du_Lieu_Full_San_Sang_Clustering.csv')
except FileNotFoundError:
    print("LỖI: Không tìm thấy file! Bạn hãy kiểm tra lại xem file có nằm cùng thư mục không nhé.")
    exit()

# Trích xuất 5000 dòng để vẽ Dendrogram mượt mà và chạy DBSCAN không bị tràn RAM
df_sample = df_full.sample(n=5000, random_state=42)

# Tách nhãn và đặc trưng (Dữ liệu trong file này ĐÃ ĐƯỢC CHUẨN HÓA từ trước)
y_thuc_te = df_sample['Vỡ_Nợ_Trong_2_Năm'].values
X_scaled = df_sample.drop(columns=['Vỡ_Nợ_Trong_2_Năm']).values

# Chạy PCA 2 chiều để phục vụ cho việc vẽ biểu đồ lưới 4 thuật toán ở cuối
X_pca = PCA(n_components=2).fit_transform(X_scaled)


# =========================================================
# PHẦN 2: VẼ 3 BIỂU ĐỒ CHỨNG MINH THAM SỐ (HỌC THUẬT)
# =========================================================
print("\n[2/3] Đang vẽ các biểu đồ học thuật chứng minh tham số...")
print("LƯU Ý: Mỗi khi biểu đồ hiện lên, bạn cần TẮT (dấu X) cửa sổ ảnh đó đi thì code mới chạy tiếp!")

# --- Hình 1: Elbow Method (Tìm số cụm K cho K-Means) ---
inertias = []
khoang_K = range(2, 11)
for k in khoang_K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(khoang_K, inertias, marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)
plt.title('Phương pháp Elbow xác định số cụm K tối ưu', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Số lượng cụm (K)', fontsize=12)
plt.ylabel('Tổng bình phương khoảng cách nội cụm (Inertia)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Hinh1_Elbow.png', dpi=300)
plt.show() 

# --- Hình 2: k-NN Distance (Tìm tham số eps cho DBSCAN) ---
k_lang_gieng = 15
nn = NearestNeighbors(n_neighbors=k_lang_gieng).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
khoang_cach_k = np.sort(distances[:, k_lang_gieng-1])

plt.figure(figsize=(8, 5))
plt.plot(khoang_cach_k, color='#27ae60', linewidth=2)
plt.title(f'Biểu đồ k-NN Distance (Xác định tham số eps)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Các điểm dữ liệu (Đã sắp xếp)', fontsize=12)
plt.ylabel(f'Khoảng cách tới láng giềng thứ {k_lang_gieng}', fontsize=12)
plt.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Gợi ý eps ≈ 1.5')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Hinh2_kNN.png', dpi=300)
plt.show()

# --- Hình 3: Dendrogram (Sơ đồ cây Hierarchical Clustering) ---
plt.figure(figsize=(10, 6))
plt.title("Sơ đồ cây phân cấp - Dendrogram (Phương pháp Ward)", fontsize=14, fontweight='bold', pad=15)
shc.dendrogram(shc.linkage(X_scaled, method='ward'), truncate_mode='level', p=5, color_threshold=45)
plt.axhline(y=45, color='red', linestyle='--', linewidth=2, label='Đường cắt ngang (Sinh ra 3 cụm)')
plt.ylabel('Khoảng cách Euclidean', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('Hinh3_Dendrogram.png', dpi=300)
plt.show()


# =========================================================
# PHẦN 3: VẼ LƯỚI 4 BIỂU ĐỒ PHÂN CỤM (K-Means, GMM, Hierarchical, DBSCAN)
# =========================================================
print("\n[3/3] Đang chạy 4 thuật toán và vẽ phân cụm...")
so_cum = 3
cac_mo_hinh = {
    "K-Means Clustering": KMeans(n_clusters=so_cum, random_state=42, n_init=10),
    "Gaussian Mixture (GMM)": GaussianMixture(n_components=so_cum, random_state=42),
    "Hierarchical Clustering": AgglomerativeClustering(n_clusters=so_cum),
    "DBSCAN (Density-based)": DBSCAN(eps=1.5, min_samples=15)
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
fig.patch.set_facecolor('#ffffff')

for i, (ten_mo_hinh, mo_hinh) in enumerate(cac_mo_hinh.items()):
    # Huấn luyện mô hình ngay trên dữ liệu mẫu
    nhan_cum = mo_hinh.fit_predict(X_scaled)
    so_luong_cum = len(set(nhan_cum)) - (1 if -1 in nhan_cum else 0)
    
    # Vẽ Scatter Plot trên không gian 2D của PCA
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=nhan_cum, cmap='plasma', s=20, alpha=0.7)
    axes[i].set_title(f"{ten_mo_hinh}\n(Số cụm sinh ra: {so_luong_cum})", fontsize=15, fontweight='bold', pad=10, color='#2c3e50')
    axes[i].set_xlabel("Thành phần chính 1 (PC1)", fontsize=12)
    axes[i].set_ylabel("Thành phần chính 2 (PC2)", fontsize=12)
    axes[i].grid(True, linestyle='--', alpha=0.5)
    
    # Thanh màu Legend
    fig.colorbar(scatter, ax=axes[i], label='Cụm')

plt.tight_layout(pad=4.0)
plt.savefig('Hinh4_Luoi_4_Mo_Hinh.png', dpi=300)

print("\nĐANG HIỂN THỊ BỨC ẢNH CUỐI CÙNG (LƯỚI 4 MÔ HÌNH)...")
plt.show()

print("\n=== ĐÃ HOÀN TẤT TOÀN BỘ QUY TRÌNH! ===")
print("Các file ảnh đã được tự động lưu lại thành công: ")
print("- Hinh1_Elbow.png\n- Hinh2_kNN.png\n- Hinh3_Dendrogram.png\n- Hinh4_Luoi_4_Mo_Hinh.png")