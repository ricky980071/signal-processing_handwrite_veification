from main import binary_array_extraction  # 匯入二值化函數
import numpy as np  # 匯入 numpy
from PIL import Image  # 匯入 PIL 影像處理
import csv  # 匯入 csv 處理

# 定義 phi 函數，產生複數遮罩
# m, n: 座標，L: 遮罩半徑
# m=n=0 時回傳 0，否則回傳 (n-jm)/sqrt(m^2+n^2)
def phi(m, n, L):
    if m == 0 and n == 0:
        return 0
    else:
        denominator = np.sqrt(m**2 + n**2)
        return (n - 1j * m) / denominator

# 產生 phi 遮罩矩陣，大小 (2L+1)x(2L+1)
def phi_matrix(L):
    size = 2 * L + 1
    phi_matrix = np.zeros((size, size), dtype=complex)
    for i, m in enumerate(range(-L, L + 1)):
        for j, n in enumerate(range(-L, L + 1)):
            phi_matrix[i, j] = phi(m, n, L)
    return phi_matrix

# 計算方向特徵（卷積法）
# L: 遮罩半徑，image: PIL 影像
# 回傳每個像素的方向角度陣列（弧度）
def calculate_direction(L, image):
    phi_mat = phi_matrix(L)  # 取得 phi 遮罩
    binary_array = binary_array_extraction(image)  # 取得二值化陣列
    convolved_mask = np.zeros_like(binary_array, dtype=complex)  # 建立複數陣列
    padded_binary_array = np.pad(binary_array, pad_width=L, mode='constant', constant_values=0)  # 邊界補零
    for i in range(binary_array.shape[0]):  # 對每個像素做卷積
        for j in range(binary_array.shape[1]):
            region = padded_binary_array[i:i + 2 * L + 1, j:j + 2 * L + 1]
            convolved_mask[i, j] = np.sum(region * phi_mat)
    angle_mask = np.angle(convolved_mask)  # 取角度
    return angle_mask

# x 座標正規化（0~100）
def coordinate_normalization_m(image, m):
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    m_min = m_max = 0
    if len(stroke_indices[0]) > 0:
        m_min = np.min(stroke_indices[1])  # 最小 x
        m_max = np.max(stroke_indices[1])  # 最大 x
    m_hat = (m - m_min) / (m_max - m_min) * 100
    return m_hat

# y 座標正規化（0~100）
def coordinate_normalization_n(image, n):
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    n_min = n_max = 0
    if len(stroke_indices[0]) > 0:
        n_min = np.min(stroke_indices[0])  # 最小 y
        n_max = np.max(stroke_indices[0])  # 最大 y
    n_hat = (n - n_min) / (n_max - n_min) * 100
    return n_hat

# 計算 x 累積比例（stroke_ratio_m）
def stroke_ratio_m(m):
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0, 0  # 避免除零
    m_ratio = np.sum(stroke_indices[1] <= m) / total_stroke_pixels
    return m_ratio

# 計算 y 累積比例（stroke_ratio_n）
def stroke_ratio_n(n):
    binary_array = binary_array_extraction(image)
    stroke_indices = np.where(binary_array > 0)
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0, 0  # 避免除零
    n_ratio = np.sum(stroke_indices[0] <= n) / total_stroke_pixels
    return n_ratio

# 主程式測試
if __name__ == "__main__":
    test_image_path = '../data/1/database/base_1_1_1.bmp'  # 測試圖像路徑
    image = Image.open(test_image_path)  # 讀取圖像
    image_width, image_height = image.size  # 取得圖像尺寸

    # 計算方向特徵（可選）
    # conv = calculate_direction(3, image)
    # # Save the conv array into a CSV file
    # output_csv_path = '../results/conv_angle_output.csv'

    # with open(output_csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(conv)

    # # print normalized coordinates csv
    # normalized_coordinates_m = []
    # for m in range(image_width):
    #     m_hat = coordinate_normalization_m(image, m)
    #     normalized_coordinates_m.append([m, m_hat])

    # normalized_coordinates_n = []
    # for n in range(image_height):
    #     n_hat = coordinate_normalization_n(image, n)
    #     normalized_coordinates_n.append([n, n_hat])

    # # Save the normalized coordinates into a CSV file
    # output_csv_path = '../results/normalized_coordinates_m_output.csv'

    # with open(output_csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['m', 'normalized_m'])
    #     writer.writerows(normalized_coordinates_m)
    
    # output_csv_path = '../results/normalized_coordinates_n_output.csv'

    # with open(output_csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['n', 'normalized_n'])
    #     writer.writerows(normalized_coordinates_n)

    # 計算 x 累積比例
    stroke_ratios_m = []
    for m in range(image_width):
        stroke_ratios_m.append([m, stroke_ratio_m(m)])
    # 計算 y 累積比例
    stroke_ratios_n = []
    for n in range(image_height):
        stroke_ratios_n.append([n, stroke_ratio_n(n)])

    # 儲存 x 累積比例到 CSV
    output_csv_path = '../results/stroke_ratios_m_output.csv'
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['m', 'stroke_ratio_m'])
        writer.writerows(stroke_ratios_m)

    # 儲存 y 累積比例到 CSV
    output_csv_path = '../results/stroke_ratios_n_output.csv'
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n', 'stroke_ratio_n'])
        writer.writerows(stroke_ratios_n)