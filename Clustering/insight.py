import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as shc
import os
import warnings
warnings.filterwarnings('ignore')

print("=== VẼ BIỂU ĐỒ HỌC THUẬT & PHÂN CỤM (TỐI ƯU) ===")

# =========================================================
# PHẦN 1: TẢI DỮ LIỆU & LẤY MẪU
# =========================================================
# Đảm bảo đường dẫn này khớp với nơi bạn lưu file ở Bước 3
duong_dan = 'Du_Lieu_Full_San_Sang_Clustering.csv' 

if not os.path.exists(duong_dan):
    print(f"LỖI: Không tìm thấy file {duong_dan}!")
    exit()

df_full = pd.read_csv(duong_dan)

# Lấy mẫu 5000 dòng để máy chạy nhanh và biểu đồ Dendrogram không bị rối
df_sample = df_full.sample(n=min(5000, len(df_full)), random_state=42)

# FIX: Đổi 'Vỡ_Nợ_Trong_2_Năm' thành 'Vo_No' cho đồng bộ
y_thuc_te = df_sample['Vo_No'].values
X_scaled = df_sample.drop(columns=['Vo_No']).values

# PCA 2 chiều để vẽ biểu đồ
X_pca = PCA(n_components=2).fit_transform(X_scaled)

# =========================================================
# PHẦN 2: CÁC BIỂU ĐỒ CHỨNG MINH THAM SỐ
# =========================================================
print("\n[1/2] Đang tạo các biểu đồ học thuật (Elbow, k-NN, Dendrogram)...")

fig_params, axes_params = plt.subplots(1, 3, figsize=(20, 6))

# --- 1. Elbow Method (K-Means) ---
inertias = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)
axes_params[0].plot(K_range, inertias, marker='o', color='#2c3e50', lw=2)
axes_params[0].set_title('1. Phương pháp Elbow (K-Means)', fontweight='bold')
axes_params[0].set_xlabel('Số cụm K')
axes_params[0].set_ylabel('Inertia')

# --- 2. k-NN Distance (DBSCAN) ---
nn = NearestNeighbors(n_neighbors=10).fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
dist_sorted = np.sort(distances[:, 9])
axes_params[1].plot(dist_sorted, color='#27ae60', lw=2)
axes_params[1].axhline(y=1.5, color='red', linestyle='--', label='eps ≈ 1.5')
axes_params[1].set_title('2. k-NN Distance (DBSCAN)', fontweight='bold')
axes_params[1].legend()

# --- 3. Dendrogram (Hierarchical) ---
axes_params[2].set_title('3. Dendrogram (Hierarchical)', fontweight='bold')
shc.dendrogram(shc.linkage(X_scaled, method='ward'), ax=axes_params[2], truncate_mode='level', p=3)
axes_params[2].axhline(y=45, color='red', linestyle='--')

plt.tight_layout()
plt.savefig('Bieu_Do_Tham_So.png', dpi=300)
print("-> Đã lưu: Bieu_Do_Tham_So.png")
plt.show()

# =========================================================
# PHẦN 3: SO SÁNH 4 THUẬT TOÁN PHÂN CỤM
# =========================================================
print("\n[2/2] Đang chạy phân cụm và trực quan hóa PCA...")

cac_mo_hinh = {
    "K-Means": KMeans(n_clusters=3, random_state=42, n_init=10),
    "GMM": GaussianMixture(n_components=3, random_state=42),
    "Hierarchical": AgglomerativeClustering(n_clusters=3),
    "DBSCAN": DBSCAN(eps=1.5, min_samples=15)
}

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(cac_mo_hinh.items()):
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    sc = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=15, alpha=0.6)
    axes[i].set_title(f"{name} (Số cụm: {n_clusters})", fontsize=14, fontweight='bold')
    plt.colorbar(sc, ax=axes[i])

plt.tight_layout()
plt.savefig('Ket_Qua_Phan_Cum.png', dpi=300)
print("-> Đã lưu: Ket_Qua_Phan_Cum.png")
plt.show()

print("\n=== HOÀN TẤT! Dữ liệu của bạn đã được phân cụm thành công. ===")