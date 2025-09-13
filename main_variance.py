from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_features(image_path, verbose=False):
    # 開啟圖片並轉灰階
    img = Image.open(image_path).convert('RGB')
    np_img = np.array(img)

    # RGB -> Grayscale
    Y = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    binary_img = (Y > 220).astype(np.uint8) * 255  # 字跡 = 黑 (0), 背景 = 白 (255)

    # 把黑色 (0) 當作筆跡
    stroke_mask = (binary_img == 0).astype(np.uint8)

    # 垂直切割區段：38, 38, 37, 38, 38
    cols = [38, 38, 37, 38, 38]
    col_start = 0
    f1_to_f5 = []
    for w in cols:
        col_end = col_start + w
        count = np.sum(stroke_mask[:, col_start:col_end])
        f1_to_f5.append(count)
        col_start = col_end

    # 水平切割（橫向切五條）
    row_start = 0
    f6_to_f10 = []
    for h in cols:  # 寬高相同
        row_end = row_start + h
        count = np.sum(stroke_mask[row_start:row_end, :])
        f6_to_f10.append(count)
        row_start = row_end
        
    # 計算動差特徵 (Moments)
    total_strokes = np.sum(stroke_mask)
    
    if total_strokes > 0:  # 確保有筆劃
        # 創建所有像素的坐標網格
        y_indices, x_indices = np.indices(stroke_mask.shape)
        
        # 將坐標轉換為以中心點為原點的坐標
        center_x, center_y = 95, 95  # 圖像中心點(95, 95)
        m_coords = x_indices - center_x  # m對應x座標
        n_coords = y_indices - center_y  # n對應y座標
        
        # 按照公式計算質心 (公式中的m0和n0)
        m0 = np.sum(m_coords * stroke_mask) / total_strokes
        n0 = np.sum(n_coords * stroke_mask) / total_strokes
        
        # 計算二階中心化動差
        V_2_0 = np.sum(((m_coords - m0) ** 2) * stroke_mask) / total_strokes
        V_1_1 = np.sum((m_coords - m0) * (n_coords - n0) * stroke_mask) / total_strokes
        V_0_2 = np.sum(((n_coords - n0) ** 2) * stroke_mask) / total_strokes
    else:
        # 如果沒有筆劃，設定預設值
        V_2_0 = 0
        V_1_1 = 0
        V_0_2 = 0

    # 合併成一個 13 維向量 (原始 10 維 + 新增 3 維二階動差特徵)
    original_features = np.array(f1_to_f5 + f6_to_f10)
    moment_features = np.array([V_2_0, V_1_1, V_0_2])  # 只包含3個二階動差特徵
    feature_vector = np.concatenate((original_features, moment_features))

    # 只在需要時列印
    if verbose:
        print(f"f1 - f5 (垂直直條筆劃數): {f1_to_f5}")
        print(f"f6 - f10 (水平橫條筆劃數): {f6_to_f10}")
        print(f"二階動差 (V_2_0, V_1_1, V_0_2): ({V_2_0:.2f}, {V_1_1:.2f}, {V_0_2:.2f})")
        print(f"特徵向量: {feature_vector}")

    return feature_vector

def process_images(folder_path, limit, offset=0):
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

# 新增標準化功能 - 更新以支援新的特徵
def standardize_features(features, vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment_mean=None, moment_std=None):
    # 將 f1-f5 (垂直) 和 f6-f10 (水平) 分別標準化
    vertical_std = np.where(vertical_std == 0, 1, vertical_std)  # 避免除以零
    horizontal_std = np.where(horizontal_std == 0, 1, horizontal_std)  # 避免除以零

    vertical_features = (features[:5] - vertical_mean) / vertical_std
    horizontal_features = (features[5:10] - horizontal_mean) / horizontal_std
    
    # 如果有提供動差特徵的標準化參數，則標準化動差特徵
    if moment_mean is not None and moment_std is not None and len(features) > 10:
        moment_std = np.where(moment_std == 0, 1, moment_std)  # 避免除以零
        moment_features = (features[10:] - moment_mean) / moment_std
        return np.concatenate((vertical_features, horizontal_features, moment_features))
    else:
        return np.concatenate((vertical_features, horizontal_features))

def calculate_mean_and_std(folder_path, limit):
    # 計算資料夾中所有圖片的特徵的平均值與標準差
    features = process_images(folder_path, limit)
    if features.size == 0:
        return np.zeros(5), np.ones(5), np.zeros(5), np.ones(5), np.zeros(3), np.ones(3)
    
    vertical_features = features[:, :5]
    horizontal_features = features[:, 5:10]
    vertical_mean = np.mean(vertical_features, axis=0)
    vertical_std = np.std(vertical_features, axis=0)
    horizontal_mean = np.mean(horizontal_features, axis=0)
    horizontal_std = np.std(horizontal_features, axis=0)
    
    # 處理動差特徵（如果存在）
    if features.shape[1] > 10:
        moment_features = features[:, 10:]
        moment_mean = np.mean(moment_features, axis=0)
        moment_std = np.std(moment_features, axis=0)
        return vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment_mean, moment_std
    else:
        return vertical_mean, vertical_std, horizontal_mean, horizontal_std, np.zeros(3), np.ones(3)

# 是否啟用標準化
enable_standardization = True

# 設定每個資料夾處理的樣本數量
samples_per_class = 25  # 每個類別使用25筆資料訓練

# 存儲所有數字的總體結果
all_predictions_standardized = []
all_predictions_non_standardized = []
all_test_labels = []

# 處理所有數字(1-9)的資料夾
print("開始處理所有數字資料夾...")

for digit in range(1, 10):  # 從1到9
    print(f"\n===== 處理數字 {digit} =====")
    
    # 定義資料夾路徑
    database_path = f'handwrite/{digit}/database'
    testcase_path = f'handwrite/{digit}/testcase'
    
    # 檢查資料夾是否存在
    if not os.path.exists(database_path) or not os.path.exists(testcase_path):
        print(f"警告：數字 {digit} 的資料夾不存在，跳過處理")
        continue
    
    # 計算 database 的平均值與標準差 (更新為包含動差特徵)
    vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment_mean, moment_std = calculate_mean_and_std(database_path, 50)
    
    # 處理 database 和 testcase 資料夾
    train_features_raw = np.vstack((
        process_images(database_path, samples_per_class),
        process_images(testcase_path, samples_per_class)
    ))
    
    # 後25項作為答案集
    answer_features_raw = np.vstack((
        process_images(database_path, samples_per_class, offset=samples_per_class),
        process_images(testcase_path, samples_per_class, offset=samples_per_class)
    ))
    
    # 檢查是否有足夠的數據
    if train_features_raw.size == 0 or answer_features_raw.size == 0:
        print(f"警告：數字 {digit} 的資料不足，跳過處理")
        continue
    
    # 保存原始和標準化後的特徵
    train_features = train_features_raw.copy()
    answer_features = answer_features_raw.copy()
    
    # 如果啟用標準化，則對特徵進行標準化 (更新標準化函數呼叫)
    if enable_standardization:
        train_features = np.array([
            standardize_features(f, vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment_mean, moment_std)
            for f in train_features_raw
        ])
        answer_features = np.array([
            standardize_features(f, vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment_mean, moment_std)
            for f in answer_features_raw
        ])
    
    # 檢查是否有 NaN 值
    if np.isnan(train_features).any() or np.isnan(answer_features).any():
        print(f"警告：數字 {digit} 的資料中包含 NaN 值！請檢查標準化過程。")
        continue
    
    # 建立 SVM 模型
    svm_model_standardized = SVC(kernel='linear', random_state=42)
    svm_model_non_standardized = SVC(kernel='linear', random_state=42)
    
    # 訓練資料的標籤 (假設前 samples_per_class 筆為類別 0，後 samples_per_class 筆為類別 1)
    train_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
    
    # 測試資料的標籤 (假設前 samples_per_class 筆為類別 0，後 samples_per_class 筆為類別 1)
    test_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
    
    # 訓練 SVM 模型（有標準化）
    svm_model_standardized.fit(train_features, train_labels)
    
    # 使用測試集進行預測（有標準化）
    predictions_standardized = svm_model_standardized.predict(answer_features)
    
    # 計算正確率（有標準化）
    accuracy_standardized = accuracy_score(test_labels, predictions_standardized)
    
    # 訓練 SVM 模型（無標準化）
    svm_model_non_standardized.fit(train_features_raw, train_labels)
    
    # 使用測試集進行預測（無標準化）
    predictions_non_standardized = svm_model_non_standardized.predict(answer_features_raw)
    
    # 計算正確率（無標準化）
    accuracy_non_standardized = accuracy_score(test_labels, predictions_non_standardized)
    
    # 輸出當前數字的結果
    print(f"數字 {digit} 的 SVM 模型正確率（有標準化）: {accuracy_standardized * 100:.2f}%")
    print(f"數字 {digit} 的 SVM 模型正確率（無標準化）: {accuracy_non_standardized * 100:.2f}%")
    
    # 收集所有預測結果和實際標籤，用於計算總體正確率
    all_predictions_standardized.extend(predictions_standardized)
    all_predictions_non_standardized.extend(predictions_non_standardized)
    all_test_labels.extend(test_labels)

# 計算總體正確率
if all_test_labels:
    total_accuracy_standardized = accuracy_score(all_test_labels, all_predictions_standardized)
    total_accuracy_non_standardized = accuracy_score(all_test_labels, all_predictions_non_standardized)
    
    print("\n===== 總結 =====")
    print(f"總體 SVM 模型正確率（有標準化）: {total_accuracy_standardized * 100:.2f}%")
    print(f"總體 SVM 模型正確率（無標準化）: {total_accuracy_non_standardized * 100:.2f}%")
else:
    print("警告：沒有任何有效的測試資料")