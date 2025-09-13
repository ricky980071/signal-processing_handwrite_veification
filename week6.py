import os  # 處理檔案與資料夾
import numpy as np  # 數值運算
import matplotlib.pyplot as plt  # 畫圖
from week5_FIN6_19 import extract_strokes, detect_edges, clock_wise_edge_numbering, compute_edge_angles, find_angle_points  # 匯入自訂特徵點與輪廓追蹤
from PIL import Image  # 影像處理


Y_STROKE_MAX = 220  # 筆劃判斷的亮度閾值

def stroke_extraction(image):
    """
    使用 week5_FIN6_19.py 的 extract_strokes 標準，將 PIL Image 轉為筆劃遮罩與亮度陣列。
    這裡 image 可以是 PIL Image 或 image_path。
    """
    from week5_FIN6_19 import extract_strokes
    # 若傳入的是路徑，先開啟
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    np_img = np.array(img)
    # RGB -> Grayscale
    Y = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    binary_img = (Y > 220).astype(np.uint8) * 255  # 字跡 = 黑 (0), 背景 = 白 (255)
    stroke_mask = (binary_img == 0).astype(np.uint8)
    stroke_image = Image.fromarray(stroke_mask * 255, mode='L')
    return stroke_image, Y

def binary_array_extraction(image):
    """
    取得二值化陣列（黑=1, 白=0），用 extract_strokes 標準。
    """
    stroke_image, _ = stroke_extraction(image)
    stroke_array = np.array(stroke_image)
    return (stroke_array > 0).astype(np.float32)

def phi(m, n, L):
    # 複數遮罩的單點值
    if m == 0 and n == 0:
        return 0
    else:
        denominator = np.sqrt(m**2 + n**2)
        return (n - 1j * m) / denominator

def phi_matrix(L):
    # 產生複數遮罩矩陣
    size = 2 * L + 1
    phi_matrix = np.zeros((size, size), dtype=complex)
    for i, m in enumerate(range(-L, L + 1)):
        for j, n in enumerate(range(-L, L + 1)):
            phi_matrix[i, j] = phi(m, n, L)
    return phi_matrix

def calculate_direction(L, image):
    # 卷積法計算方向特徵
    phi_mat = phi_matrix(L)
    binary_array = binary_array_extraction(image)
    convolved_mask = np.zeros_like(binary_array, dtype=complex)
    padded_binary_array = np.pad(binary_array, pad_width=L, mode='constant', constant_values=0)
    for i in range(binary_array.shape[0]):
        for j in range(binary_array.shape[1]):
            region = padded_binary_array[i:i + 2 * L + 1, j:j + 2 * L + 1]
            convolved_mask[i, j] = np.sum(region * phi_mat)
    angle_mask = np.angle(convolved_mask)
    return angle_mask

def coordinate_normalization_m(image, m):
    # x座標正規化
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    m_min = m_max = 0
    if len(stroke_indices[0]) > 0:
        m_min = np.min(stroke_indices[1])
        m_max = np.max(stroke_indices[1])
    m_hat = (m - m_min) / (m_max - m_min) * 100 if m_max > m_min else 0
    return m_hat

def coordinate_normalization_n(image, n):
    # y座標正規化
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    n_min = n_max = 0
    if len(stroke_indices[0]) > 0:
        n_min = np.min(stroke_indices[0])
        n_max = np.max(stroke_indices[0])
    n_hat = (n - n_min) / (n_max - n_min) * 100 if n_max > n_min else 0
    return n_hat

def stroke_ratio_m(image, m):
    # x累積比例
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0
    m_ratio = np.sum(stroke_indices[1] <= m) / total_stroke_pixels
    return m_ratio

def stroke_ratio_n(image, n):
    # y累積比例
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0
    n_ratio = np.sum(stroke_indices[0] <= n) / total_stroke_pixels
    return n_ratio

# 只針對一張圖做特徵分析
img_path = 'handwrite/1/database/base_1_1_1.bmp'  # 圖片路徑
image = Image.open(img_path)  # 讀取圖片
stroke_mask = extract_strokes(img_path)  # 筆劃遮罩
edges = detect_edges(stroke_mask)  # 邊緣偵測
numbered_edges, all_stroke_coords = clock_wise_edge_numbering(edges)  # 順時針編號與分群
theta_arrays = compute_edge_angles(all_stroke_coords, d=12)  # 計算每個部件的角度

# 收集所有特徵點（ending+turning）
feature_points = []
for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
    theta_deg = np.degrees(theta_arr)
    ending_pts = find_angle_points(theta_deg, case=1, window=12)
    turning_pts = find_angle_points(theta_deg, case=2, window=12)
    for idx, _ in ending_pts + turning_pts:
        feature_points.append(coords[idx])

# 卷積法方向特徵
angle_mask = calculate_direction(L=3, image=image)

# 取得所有 stroke pixel 的 (y, x) 及正規化
binary_array = binary_array_extraction(image)
stroke_indices = np.where(binary_array > 0)
all_y = stroke_indices[0]
all_x = stroke_indices[1]
m_min = np.min(all_x) if len(all_x) > 0 else 0
m_max = np.max(all_x) if len(all_x) > 0 else 0
n_min = np.min(all_y) if len(all_y) > 0 else 0
n_max = np.max(all_y) if len(all_y) > 0 else 0
# 所有像素的正規化
if m_max > m_min:
    all_m_hats = (all_x - m_min) / (m_max - m_min) * 100
else:
    all_m_hats = np.zeros_like(all_x)
if n_max > n_min:
    all_n_hats = (all_y - n_min) / (n_max - n_min) * 100
else:
    all_n_hats = np.zeros_like(all_y)
# 特徵點的正規化
m_hats = [(x - m_min) / (m_max - m_min) * 100 if m_max > m_min else 0 for x in [x for y, x in feature_points]]
n_hats = [(y - n_min) / (n_max - n_min) * 100 if n_max > n_min else 0 for y in [y for y, x in feature_points]]

# 可視化
plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 1)
plt.imshow(stroke_mask, cmap='gray')  # 原圖
plt.title('Original')

plt.subplot(2, 3, 2)
plt.imshow(angle_mask, cmap='hsv')  # 卷積方向特徵
plt.colorbar(label='Direction (radian)')
plt.title('Convolution Direction Feature')

plt.subplot(2, 3, 3)
plt.imshow(stroke_mask, cmap='gray')  # 特徵點位置
if feature_points:
    ys, xs = zip(*feature_points)
    plt.scatter(xs, ys, c='red', s=20, label='Feature Points')
plt.title('Feature Points')
plt.legend()

plt.subplot(2, 3, 4)
plt.scatter(all_m_hats, all_n_hats, c='gray', s=5, label='All Stroke Pixels')  # 所有像素正規化
plt.scatter(m_hats, n_hats, c='blue', s=20, label='Feature Points')  # 特徵點正規化
for i, (mx, ny) in enumerate(zip(m_hats, n_hats)):
    plt.text(mx, ny, str(i), color='red', fontsize=10, fontweight='bold')  # 標註順序
plt.xlabel('m_hat (normalized x)')
plt.ylabel('n_hat (normalized y)')
plt.title('Normalized Coordinates: All Stroke Pixels & Feature Points')
plt.legend()
plt.gca().invert_yaxis()  # y軸反轉

plt.subplot(2, 3, 5)
m_ratios = [(np.sum(all_x <= x) / len(all_x)) if len(all_x) > 0 else 0 for x in [x for y, x in feature_points]]
sorted_m = sorted((val, idx) for idx, val in enumerate(m_ratios))
sorted_vals, sorted_idx = zip(*sorted_m) if sorted_m else ([],[])
plt.plot(sorted_vals, np.linspace(0, 1, len(m_ratios)), marker='o')  # m_ratio累積曲線
for i, (val, idx) in enumerate(sorted_m):
    plt.text(val, i/(len(m_ratios)-1) if len(m_ratios)>1 else 0, str(idx), color='red', fontsize=10, fontweight='bold')  # 標註順序
plt.xlabel('m_ratio')
plt.ylabel('Cumulative Ratio')
plt.title('Stroke Ratio m (Feature Points)')

plt.subplot(2, 3, 6)
n_ratios = [(np.sum(all_y <= y) / len(all_y)) if len(all_y) > 0 else 0 for y in [y for y, x in feature_points]]
sorted_n = sorted((val, idx) for idx, val in enumerate(n_ratios))
sorted_vals, sorted_idx = zip(*sorted_n) if sorted_n else ([],[])
plt.plot(sorted_vals, np.linspace(0, 1, len(n_ratios)), marker='o')  # n_ratio累積曲線
for i, (val, idx) in enumerate(sorted_n):
    plt.text(val, i/(len(n_ratios)-1) if len(n_ratios)>1 else 0, str(idx), color='red', fontsize=10, fontweight='bold')  # 標註順序
plt.xlabel('n_ratio')
plt.ylabel('Cumulative Ratio')
plt.title('Stroke Ratio n (Feature Points)')

plt.tight_layout()
plt.show()