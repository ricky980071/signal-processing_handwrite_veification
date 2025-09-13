from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def extract_features_basic(image_path, verbose=False):
    """只提取基本特徵（不包含動差特徵）"""
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

    # 合併成一個 10 維向量
    feature_vector = np.array(f1_to_f5 + f6_to_f10)

    # 只在需要時列印
    if verbose:
        print(f"f1 - f5 (垂直直條筆劃數): {f1_to_f5}")
        print(f"f6 - f10 (水平橫條筆劃數): {f6_to_f10}")
        print(f"特徵向量: {feature_vector}")

    return feature_vector

def extract_features_with_2nd_moments(image_path, verbose=False):
    """提取基本特徵和二階動差特徵"""
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

def extract_features_with_all_moments(image_path, verbose=False):
    """提取基本特徵、二階動差和三階動差特徵"""
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
        
        # 計算三階中心化動差
        V_3_0 = np.sum(((m_coords - m0) ** 3) * stroke_mask) / total_strokes
        V_2_1 = np.sum(((m_coords - m0) ** 2) * (n_coords - n0) * stroke_mask) / total_strokes
        V_1_2 = np.sum((m_coords - m0) * ((n_coords - n0) ** 2) * stroke_mask) / total_strokes
        V_0_3 = np.sum(((n_coords - n0) ** 3) * stroke_mask) / total_strokes
    else:
        # 如果沒有筆劃，設定預設值
        V_2_0 = V_1_1 = V_0_2 = 0
        V_3_0 = V_2_1 = V_1_2 = V_0_3 = 0

    # 合併成一個 17 維向量 (原始 10 維 + 3 維二階動差 + 4 維三階動差)
    original_features = np.array(f1_to_f5 + f6_to_f10)
    second_moments = np.array([V_2_0, V_1_1, V_0_2])
    third_moments = np.array([V_3_0, V_2_1, V_1_2, V_0_3])
    feature_vector = np.concatenate((original_features, second_moments, third_moments))

    # 只在需要時列印
    if verbose:
        print(f"f1 - f5 (垂直直條筆劃數): {f1_to_f5}")
        print(f"f6 - f10 (水平橫條筆劃數): {f6_to_f10}")
        print(f"二階動差 (V_2_0, V_1_1, V_0_2): ({V_2_0:.2f}, {V_1_1:.2f}, {V_0_2:.2f})")
        print(f"三階動差 (V_3_0, V_2_1, V_1_2, V_0_3): ({V_3_0:.2f}, {V_2_1:.2f}, {V_1_2:.2f}, {V_0_3:.2f})")
        print(f"特徵向量: {feature_vector}")

    return feature_vector

# 處理圖像函數 - 定義三個版本
def process_images_basic(folder_path, limit, offset=0):
    """處理圖像並提取基本特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_basic(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

def process_images_with_2nd_moments(folder_path, limit, offset=0):
    """處理圖像並提取基本特徵和二階動差特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_with_2nd_moments(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

def process_images_with_all_moments(folder_path, limit, offset=0):
    """處理圖像並提取所有特徵（包括三階動差）"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_with_all_moments(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

# 標準化函數 - 定義三個版本
def standardize_features_basic(features, vertical_mean, vertical_std, horizontal_mean, horizontal_std):
    """基本特徵標準化"""
    vertical_std = np.where(vertical_std == 0, 1, vertical_std)  # 避免除以零
    horizontal_std = np.where(horizontal_std == 0, 1, horizontal_std)  # 避免除以零

    vertical_features = (features[:5] - vertical_mean) / vertical_std
    horizontal_features = (features[5:] - horizontal_mean) / horizontal_std
    return np.concatenate((vertical_features, horizontal_features))

def standardize_features_with_2nd_moments(features, vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment2_mean, moment2_std):
    """基本特徵和二階動差標準化"""
    vertical_std = np.where(vertical_std == 0, 1, vertical_std)  # 避免除以零
    horizontal_std = np.where(horizontal_std == 0, 1, horizontal_std)  # 避免除以零
    moment2_std = np.where(moment2_std == 0, 1, moment2_std)  # 避免除以零

    vertical_features = (features[:5] - vertical_mean) / vertical_std
    horizontal_features = (features[5:10] - horizontal_mean) / horizontal_std
    moment2_features = (features[10:13] - moment2_mean) / moment2_std
    return np.concatenate((vertical_features, horizontal_features, moment2_features))

def standardize_features_with_all_moments(features, vertical_mean, vertical_std, horizontal_mean, horizontal_std, 
                                         moment2_mean, moment2_std, moment3_mean, moment3_std):
    """所有特徵標準化（包括三階動差）"""
    vertical_std = np.where(vertical_std == 0, 1, vertical_std)  # 避免除以零
    horizontal_std = np.where(horizontal_std == 0, 1, horizontal_std)  # 避免除以零
    moment2_std = np.where(moment2_std == 0, 1, moment2_std)  # 避免除以零
    moment3_std = np.where(moment3_std == 0, 1, moment3_std)  # 避免除以零

    vertical_features = (features[:5] - vertical_mean) / vertical_std
    horizontal_features = (features[5:10] - horizontal_mean) / horizontal_std
    moment2_features = (features[10:13] - moment2_mean) / moment2_std
    moment3_features = (features[13:] - moment3_mean) / moment3_std
    return np.concatenate((vertical_features, horizontal_features, moment2_features, moment3_features))

# 平均值和標準差計算函數 - 定義三個版本
def calculate_mean_and_std_basic(folder_path, limit):
    """計算基本特徵的平均值和標準差"""
    features = process_images_basic(folder_path, limit)
    if features.size == 0:
        return np.zeros(5), np.ones(5), np.zeros(5), np.ones(5)
    
    vertical_features = features[:, :5]
    horizontal_features = features[:, 5:]
    vertical_mean = np.mean(vertical_features, axis=0)
    vertical_std = np.std(vertical_features, axis=0)
    horizontal_mean = np.mean(horizontal_features, axis=0)
    horizontal_std = np.std(horizontal_features, axis=0)
    return vertical_mean, vertical_std, horizontal_mean, horizontal_std

def calculate_mean_and_std_with_2nd_moments(folder_path, limit):
    """計算基本特徵和二階動差的平均值和標準差"""
    features = process_images_with_2nd_moments(folder_path, limit)
    if features.size == 0:
        return np.zeros(5), np.ones(5), np.zeros(5), np.ones(5), np.zeros(3), np.ones(3)
    
    vertical_features = features[:, :5]
    horizontal_features = features[:, 5:10]
    moment2_features = features[:, 10:13]
    vertical_mean = np.mean(vertical_features, axis=0)
    vertical_std = np.std(vertical_features, axis=0)
    horizontal_mean = np.mean(horizontal_features, axis=0)
    horizontal_std = np.std(horizontal_features, axis=0)
    moment2_mean = np.mean(moment2_features, axis=0)
    moment2_std = np.std(moment2_features, axis=0)
    return vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment2_mean, moment2_std

def calculate_mean_and_std_with_all_moments(folder_path, limit):
    """計算所有特徵的平均值和標準差（包括三階動差）"""
    features = process_images_with_all_moments(folder_path, limit)
    if features.size == 0:
        return np.zeros(5), np.ones(5), np.zeros(5), np.ones(5), np.zeros(3), np.ones(3), np.zeros(4), np.ones(4)
    
    vertical_features = features[:, :5]
    horizontal_features = features[:, 5:10]
    moment2_features = features[:, 10:13]
    moment3_features = features[:, 13:]
    
    vertical_mean = np.mean(vertical_features, axis=0)
    vertical_std = np.std(vertical_features, axis=0)
    horizontal_mean = np.mean(horizontal_features, axis=0)
    horizontal_std = np.std(horizontal_features, axis=0)
    moment2_mean = np.mean(moment2_features, axis=0)
    moment2_std = np.std(moment2_features, axis=0)
    moment3_mean = np.mean(moment3_features, axis=0)
    moment3_std = np.std(moment3_features, axis=0)
    
    return vertical_mean, vertical_std, horizontal_mean, horizontal_std, moment2_mean, moment2_std, moment3_mean, moment3_std

# 主程式執行部分
def main():
    # 是否啟用標準化
    enable_standardization = True

    # 設定每個資料夾處理的樣本數量
    samples_per_class = 25  # 每個類別使用25筆資料訓練

    # 存儲所有數字的總體結果
    basic_predictions = []
    moment2_predictions = []
    moment3_predictions = []
    all_test_labels = []

    # 處理所有數字(1-9)的資料夾
    print("開始處理所有數字資料夾...")
    print("將比較三種特徵組合：「基本特徵」、「基本特徵+二階動差」、「基本特徵+二階動差+三階動差」")

    for digit in range(1, 10):  # 從1到9
        print(f"\n===== 處理數字 {digit} =====")
        
        # 定義資料夾路徑
        database_path = f'handwrite/{digit}/database'
        testcase_path = f'handwrite/{digit}/testcase'
        
        # 檢查資料夾是否存在
        if not os.path.exists(database_path) or not os.path.exists(testcase_path):
            print(f"警告：數字 {digit} 的資料夾不存在，跳過處理")
            continue
        
        # 計算各種特徵的平均值和標準差
        basic_vmean, basic_vstd, basic_hmean, basic_hstd = calculate_mean_and_std_basic(database_path, 50)
        
        moment2_vmean, moment2_vstd, moment2_hmean, moment2_hstd, moment2_mean, moment2_std = calculate_mean_and_std_with_2nd_moments(database_path, 50)
        
        moment3_vmean, moment3_vstd, moment3_hmean, moment3_hstd, moment3_2_mean, moment3_2_std, moment3_mean, moment3_std = calculate_mean_and_std_with_all_moments(database_path, 50)
        
        # 處理各種特徵的訓練和測試資料
        # 基本特徵
        basic_train_raw = np.vstack((
            process_images_basic(database_path, samples_per_class),
            process_images_basic(testcase_path, samples_per_class)
        ))
        
        basic_answer_raw = np.vstack((
            process_images_basic(database_path, samples_per_class, offset=samples_per_class),
            process_images_basic(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 二階動差特徵
        moment2_train_raw = np.vstack((
            process_images_with_2nd_moments(database_path, samples_per_class),
            process_images_with_2nd_moments(testcase_path, samples_per_class)
        ))
        
        moment2_answer_raw = np.vstack((
            process_images_with_2nd_moments(database_path, samples_per_class, offset=samples_per_class),
            process_images_with_2nd_moments(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 三階動差特徵
        moment3_train_raw = np.vstack((
            process_images_with_all_moments(database_path, samples_per_class),
            process_images_with_all_moments(testcase_path, samples_per_class)
        ))
        
        moment3_answer_raw = np.vstack((
            process_images_with_all_moments(database_path, samples_per_class, offset=samples_per_class),
            process_images_with_all_moments(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 檢查是否有足夠的數據
        if basic_train_raw.size == 0 or basic_answer_raw.size == 0:
            print(f"警告：數字 {digit} 的資料不足，跳過處理")
            continue
        
        # 保存原始和標準化後的特徵
        basic_train = basic_train_raw.copy()
        basic_answer = basic_answer_raw.copy()
        moment2_train = moment2_train_raw.copy()
        moment2_answer = moment2_answer_raw.copy()
        moment3_train = moment3_train_raw.copy()
        moment3_answer = moment3_answer_raw.copy()
        
        # 如果啟用標準化，則對特徵進行標準化
        if enable_standardization:
            basic_train = np.array([
                standardize_features_basic(f, basic_vmean, basic_vstd, basic_hmean, basic_hstd)
                for f in basic_train_raw
            ])
            basic_answer = np.array([
                standardize_features_basic(f, basic_vmean, basic_vstd, basic_hmean, basic_hstd)
                for f in basic_answer_raw
            ])
            
            moment2_train = np.array([
                standardize_features_with_2nd_moments(f, moment2_vmean, moment2_vstd, moment2_hmean, moment2_hstd, moment2_mean, moment2_std)
                for f in moment2_train_raw
            ])
            moment2_answer = np.array([
                standardize_features_with_2nd_moments(f, moment2_vmean, moment2_vstd, moment2_hmean, moment2_hstd, moment2_mean, moment2_std)
                for f in moment2_answer_raw
            ])
            
            moment3_train = np.array([
                standardize_features_with_all_moments(f, moment3_vmean, moment3_vstd, moment3_hmean, moment3_hstd, 
                                                     moment3_2_mean, moment3_2_std, moment3_mean, moment3_std)
                for f in moment3_train_raw
            ])
            moment3_answer = np.array([
                standardize_features_with_all_moments(f, moment3_vmean, moment3_vstd, moment3_hmean, moment3_hstd, 
                                                     moment3_2_mean, moment3_2_std, moment3_mean, moment3_std)
                for f in moment3_answer_raw
            ])
        
        # 檢查是否有 NaN 值
        has_nan = False
        if np.isnan(basic_train).any() or np.isnan(basic_answer).any():
            print(f"警告：數字 {digit} 的基本特徵資料中包含 NaN 值！")
            has_nan = True
        if np.isnan(moment2_train).any() or np.isnan(moment2_answer).any():
            print(f"警告：數字 {digit} 的二階動差特徵資料中包含 NaN 值！")
            has_nan = True
        if np.isnan(moment3_train).any() or np.isnan(moment3_answer).any():
            print(f"警告：數字 {digit} 的三階動差特徵資料中包含 NaN 值！")
            has_nan = True
        
        if has_nan:
            continue
        
        # 建立 SVM 模型
        basic_model = SVC(kernel='linear', random_state=42)
        moment2_model = SVC(kernel='linear', random_state=42)
        moment3_model = SVC(kernel='linear', random_state=42)
        
        # 訓練資料的標籤 (前 25 筆為類別 0，後 25 筆為類別 1)
        train_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
        
        # 測試資料的標籤 (前 25 筆為類別 0，後 25 筆為類別 1)
        test_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
        
        # 訓練和測試所有模型
        # 基本特徵
        basic_model.fit(basic_train, train_labels)
        basic_pred = basic_model.predict(basic_answer)
        basic_acc = accuracy_score(test_labels, basic_pred)
        
        # 二階動差特徵
        moment2_model.fit(moment2_train, train_labels)
        moment2_pred = moment2_model.predict(moment2_answer)
        moment2_acc = accuracy_score(test_labels, moment2_pred)
        
        # 三階動差特徵
        moment3_model.fit(moment3_train, train_labels)
        moment3_pred = moment3_model.predict(moment3_answer)
        moment3_acc = accuracy_score(test_labels, moment3_pred)
        
        # 輸出當前數字的結果
        print(f"數字 {digit} 結果比較:")
        print(f"基本特徵 SVM 模型正確率: {basic_acc * 100:.2f}%")
        print(f"基本特徵+二階動差 SVM 模型正確率: {moment2_acc * 100:.2f}%")
        print(f"基本特徵+二階動差+三階動差 SVM 模型正確率: {moment3_acc * 100:.2f}%")
        
        # 收集所有預測結果和實際標籤，用於計算總體正確率
        basic_predictions.extend(basic_pred)
        moment2_predictions.extend(moment2_pred)
        moment3_predictions.extend(moment3_pred)
        all_test_labels.extend(test_labels)

    # 計算總體正確率
    if all_test_labels:
        basic_total_acc = accuracy_score(all_test_labels, basic_predictions)
        moment2_total_acc = accuracy_score(all_test_labels, moment2_predictions)
        moment3_total_acc = accuracy_score(all_test_labels, moment3_predictions)
        
        print("\n===== 總體比較結果 =====")
        print(f"基本特徵 SVM 模型總體正確率: {basic_total_acc * 100:.2f}%")
        print(f"基本特徵+二階動差 SVM 模型總體正確率: {moment2_total_acc * 100:.2f}%")
        print(f"基本特徵+二階動差+三階動差 SVM 模型總體正確率: {moment3_total_acc * 100:.2f}%")
        
        # 計算動差特徵帶來的改進
        moment2_improvement = moment2_total_acc - basic_total_acc
        moment3_improvement = moment3_total_acc - basic_total_acc
        moment3_vs_moment2 = moment3_total_acc - moment2_total_acc
        
        print("\n===== 動差特徵改進效果 =====")
        print(f"二階動差特徵相對於基本特徵的正確率提升: {moment2_improvement * 100:.2f}%")
        print(f"二階+三階動差特徵相對於基本特徵的正確率提升: {moment3_improvement * 100:.2f}%")
        print(f"增加三階動差相對於只有二階動差的額外提升: {moment3_vs_moment2 * 100:.2f}%")
    else:
        print("警告：沒有任何有效的測試資料")

if __name__ == "__main__":
    main()