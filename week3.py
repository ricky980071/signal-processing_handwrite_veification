from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import ndimage

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

def extract_features_with_moments(image_path, verbose=False):
    """提取基本特徵和動差特徵（二階+三階）"""
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

    return feature_vector, Y, stroke_mask  # 同時返回灰度圖和筆劃遮罩，方便計算其他特徵

def calculate_intensity_features(Y, stroke_mask):
    """計算筆劃強度特徵 (平均值和標準差)"""
    total_strokes = np.sum(stroke_mask)
    
    if total_strokes > 0:
        # 反轉灰度值（使筆劃為高值，背景為低值）
        inverted_Y = 255 - Y
        
        # 計算筆劃強度平均值
        mu_I = np.sum(inverted_Y * stroke_mask) / total_strokes
        
        # 計算筆劃強度標準差
        sigma_I = np.sqrt(np.sum(((inverted_Y - mu_I) ** 2) * stroke_mask) / total_strokes)
    else:
        mu_I = 0
        sigma_I = 0
    
    return np.array([mu_I, sigma_I])

def calculate_erosion_features(stroke_mask):
    """計算侵蝕特徵 (三個不同級別的侵蝕比率)"""
    total_strokes = np.sum(stroke_mask)
    
    if total_strokes > 0:
        # 初始化侵蝕遮罩
        eroded_mask = stroke_mask.copy()
        
        # 計算三個不同級別的侵蝕比率
        erosion_ratios = []
        
        # 使用結構元素進行侵蝕
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        
        for k in range(1, 4):
            # 進行侵蝕操作
            eroded_mask = ndimage.binary_erosion(eroded_mask, structure=kernel).astype(np.uint8)
            
            # 計算侵蝕比率
            eroded_count = np.sum(eroded_mask)
            e_k = eroded_count / total_strokes if total_strokes > 0 else 0
            erosion_ratios.append(e_k)
    else:
        erosion_ratios = [0, 0, 0]
    
    return np.array(erosion_ratios)

def extract_features_with_intensity(image_path, verbose=False):
    """提取基本特徵、動差特徵和筆劃強度特徵"""
    moments_features, Y, stroke_mask = extract_features_with_moments(image_path, verbose)
    intensity_features = calculate_intensity_features(Y, stroke_mask)
    
    feature_vector = np.concatenate((moments_features, intensity_features))
    
    # 只在需要時列印
    if verbose:
        print(f"筆劃強度特徵 (mu_I, sigma_I): ({intensity_features[0]:.2f}, {intensity_features[1]:.2f})")
        print(f"特徵向量: {feature_vector}")
    
    return feature_vector

def extract_features_with_erosion(image_path, verbose=False):
    """提取基本特徵、動差特徵和侵蝕特徵"""
    moments_features, Y, stroke_mask = extract_features_with_moments(image_path, verbose)
    erosion_features = calculate_erosion_features(stroke_mask)
    
    feature_vector = np.concatenate((moments_features, erosion_features))
    
    # 只在需要時列印
    if verbose:
        print(f"侵蝕特徵 (e1, e2, e3): ({erosion_features[0]:.2f}, {erosion_features[1]:.2f}, {erosion_features[2]:.2f})")
        print(f"特徵向量: {feature_vector}")
    
    return feature_vector

def extract_features_all(image_path, verbose=False):
    """提取所有特徵：基本特徵、動差特徵、筆劃強度特徵和侵蝕特徵"""
    moments_features, Y, stroke_mask = extract_features_with_moments(image_path, verbose)
    intensity_features = calculate_intensity_features(Y, stroke_mask)
    erosion_features = calculate_erosion_features(stroke_mask)
    
    feature_vector = np.concatenate((moments_features, intensity_features, erosion_features))
    
    # 只在需要時列印
    if verbose:
        print(f"筆劃強度特徵 (mu_I, sigma_I): ({intensity_features[0]:.2f}, {intensity_features[1]:.2f})")
        print(f"侵蝕特徵 (e1, e2, e3): ({erosion_features[0]:.2f}, {erosion_features[1]:.2f}, {erosion_features[2]:.2f})")
        print(f"特徵向量: {feature_vector}")
    
    return feature_vector

# 對應的處理函數 - 五個版本
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

def process_images_with_moments(folder_path, limit, offset=0):
    """處理圖像並提取基本特徵和動差特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features, _, _ = extract_features_with_moments(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

def process_images_with_intensity(folder_path, limit, offset=0):
    """處理圖像並提取基本特徵、動差特徵和筆劃強度特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_with_intensity(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

def process_images_with_erosion(folder_path, limit, offset=0):
    """處理圖像並提取基本特徵、動差特徵和侵蝕特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_with_erosion(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

def process_images_all(folder_path, limit, offset=0):
    """處理圖像並提取所有特徵：基本特徵、動差特徵、筆劃強度特徵和侵蝕特徵"""
    features_list = []
    if not os.path.exists(folder_path):
        print(f"警告：資料夾 {folder_path} 不存在")
        return np.array([])
    
    image_files = sorted(os.listdir(folder_path))  # 按名稱排序
    selected_files = image_files[offset:offset + limit]  # 根據 offset 和 limit 選取檔案
    for image_file in selected_files:
        image_path = os.path.join(folder_path, image_file)
        features = extract_features_all(image_path)
        features_list.append(features)
    
    if not features_list:
        return np.array([])
    return np.array(features_list)

# 標準化函數 - 五個版本
def standardize_features_basic(features, mean_dict, std_dict):
    """基本特徵標準化"""
    vertical_std = np.where(std_dict['vertical'] == 0, 1, std_dict['vertical'])  # 避免除以零
    horizontal_std = np.where(std_dict['horizontal'] == 0, 1, std_dict['horizontal'])  # 避免除以零

    vertical_features = (features[:5] - mean_dict['vertical']) / vertical_std
    horizontal_features = (features[5:] - mean_dict['horizontal']) / horizontal_std
    return np.concatenate((vertical_features, horizontal_features))

def standardize_features_with_moments(features, mean_dict, std_dict):
    """基本特徵和動差標準化"""
    vertical_std = np.where(std_dict['vertical'] == 0, 1, std_dict['vertical'])  # 避免除以零
    horizontal_std = np.where(std_dict['horizontal'] == 0, 1, std_dict['horizontal'])  # 避免除以零
    second_moments_std = np.where(std_dict['second_moments'] == 0, 1, std_dict['second_moments'])  # 避免除以零
    third_moments_std = np.where(std_dict['third_moments'] == 0, 1, std_dict['third_moments'])  # 避免除以零

    vertical_features = (features[:5] - mean_dict['vertical']) / vertical_std
    horizontal_features = (features[5:10] - mean_dict['horizontal']) / horizontal_std
    second_moments_features = (features[10:13] - mean_dict['second_moments']) / second_moments_std
    third_moments_features = (features[13:17] - mean_dict['third_moments']) / third_moments_std
    
    return np.concatenate((vertical_features, horizontal_features, second_moments_features, third_moments_features))

def standardize_features_with_intensity(features, mean_dict, std_dict):
    """基本特徵、動差和筆劃強度標準化"""
    vertical_std = np.where(std_dict['vertical'] == 0, 1, std_dict['vertical'])  # 避免除以零
    horizontal_std = np.where(std_dict['horizontal'] == 0, 1, std_dict['horizontal'])  # 避免除以零
    second_moments_std = np.where(std_dict['second_moments'] == 0, 1, std_dict['second_moments'])  # 避免除以零
    third_moments_std = np.where(std_dict['third_moments'] == 0, 1, std_dict['third_moments'])  # 避免除以零
    intensity_std = np.where(std_dict['intensity'] == 0, 1, std_dict['intensity'])  # 避免除以零

    vertical_features = (features[:5] - mean_dict['vertical']) / vertical_std
    horizontal_features = (features[5:10] - mean_dict['horizontal']) / horizontal_std
    second_moments_features = (features[10:13] - mean_dict['second_moments']) / second_moments_std
    third_moments_features = (features[13:17] - mean_dict['third_moments']) / third_moments_std
    intensity_features = (features[17:19] - mean_dict['intensity']) / intensity_std
    
    return np.concatenate((vertical_features, horizontal_features, second_moments_features, 
                         third_moments_features, intensity_features))

def standardize_features_with_erosion(features, mean_dict, std_dict):
    """基本特徵、動差和侵蝕特徵標準化"""
    vertical_std = np.where(std_dict['vertical'] == 0, 1, std_dict['vertical'])  # 避免除以零
    horizontal_std = np.where(std_dict['horizontal'] == 0, 1, std_dict['horizontal'])  # 避免除以零
    second_moments_std = np.where(std_dict['second_moments'] == 0, 1, std_dict['second_moments'])  # 避免除以零
    third_moments_std = np.where(std_dict['third_moments'] == 0, 1, std_dict['third_moments'])  # 避免除以零
    erosion_std = np.where(std_dict['erosion'] == 0, 1, std_dict['erosion'])  # 避免除以零

    vertical_features = (features[:5] - mean_dict['vertical']) / vertical_std
    horizontal_features = (features[5:10] - mean_dict['horizontal']) / horizontal_std
    second_moments_features = (features[10:13] - mean_dict['second_moments']) / second_moments_std
    third_moments_features = (features[13:17] - mean_dict['third_moments']) / third_moments_std
    erosion_features = (features[17:20] - mean_dict['erosion']) / erosion_std
    
    return np.concatenate((vertical_features, horizontal_features, second_moments_features, 
                         third_moments_features, erosion_features))

def standardize_features_all(features, mean_dict, std_dict):
    """所有特徵標準化"""
    vertical_std = np.where(std_dict['vertical'] == 0, 1, std_dict['vertical'])  # 避免除以零
    horizontal_std = np.where(std_dict['horizontal'] == 0, 1, std_dict['horizontal'])  # 避免除以零
    second_moments_std = np.where(std_dict['second_moments'] == 0, 1, std_dict['second_moments'])  # 避免除以零
    third_moments_std = np.where(std_dict['third_moments'] == 0, 1, std_dict['third_moments'])  # 避免除以零
    intensity_std = np.where(std_dict['intensity'] == 0, 1, std_dict['intensity'])  # 避免除以零
    erosion_std = np.where(std_dict['erosion'] == 0, 1, std_dict['erosion'])  # 避免除以零

    vertical_features = (features[:5] - mean_dict['vertical']) / vertical_std
    horizontal_features = (features[5:10] - mean_dict['horizontal']) / horizontal_std
    second_moments_features = (features[10:13] - mean_dict['second_moments']) / second_moments_std
    third_moments_features = (features[13:17] - mean_dict['third_moments']) / third_moments_std
    intensity_features = (features[17:19] - mean_dict['intensity']) / intensity_std
    erosion_features = (features[19:22] - mean_dict['erosion']) / erosion_std
    
    return np.concatenate((vertical_features, horizontal_features, second_moments_features, 
                         third_moments_features, intensity_features, erosion_features))

# 平均值和標準差計算函數 - 根據特徵類型
def calculate_stats(features_list, feature_type):
    """計算指定特徵類型的平均值和標準差"""
    # 特徵維度映射
    dim_map = {
        'basic': {'vertical': (0, 5), 'horizontal': (5, 10)},
        'moments': {'vertical': (0, 5), 'horizontal': (5, 10), 
                  'second_moments': (10, 13), 'third_moments': (13, 17)},
        'intensity': {'vertical': (0, 5), 'horizontal': (5, 10), 
                    'second_moments': (10, 13), 'third_moments': (13, 17),
                    'intensity': (17, 19)},
        'erosion': {'vertical': (0, 5), 'horizontal': (5, 10), 
                  'second_moments': (10, 13), 'third_moments': (13, 17),
                  'erosion': (17, 20)},
        'all': {'vertical': (0, 5), 'horizontal': (5, 10), 
              'second_moments': (10, 13), 'third_moments': (13, 17),
              'intensity': (17, 19), 'erosion': (19, 22)}
    }
    
    # 確認特徵類型有效
    if feature_type not in dim_map:
        raise ValueError(f"無效的特徵類型: {feature_type}")
    
    # 初始化平均值和標準差字典
    mean_dict = {}
    std_dict = {}
    
    # 計算每種特徵的平均值和標準差
    for key, (start, end) in dim_map[feature_type].items():
        feature_segment = features_list[:, start:end]
        mean_dict[key] = np.mean(feature_segment, axis=0)
        std_dict[key] = np.std(feature_segment, axis=0)
    
    return mean_dict, std_dict

# 主程式執行部分
def main():
    # 設定每個資料夾處理的樣本數量
    samples_per_class = 25  # 每個類別使用25筆資料訓練

    # 存儲所有數字的總體結果
    basic_predictions = []
    moments_predictions = []
    intensity_predictions = []
    erosion_predictions = []
    all_predictions = []
    all_test_labels = []

    # 處理所有數字(1-9)的資料夾
    print("開始處理所有數字資料夾...")
    print("將比較五種特徵組合：")
    print("1. 基本特徵")
    print("2. 基本特徵+動差特徵")
    print("3. 基本特徵+動差特徵+筆劃強度特徵")
    print("4. 基本特徵+動差特徵+侵蝕特徵")
    print("5. 全部特徵")

    for digit in range(1, 10):  # 從1到9
        print(f"\n===== 處理數字 {digit} =====")
        
        # 定義資料夾路徑
        database_path = f'handwrite/{digit}/database'
        testcase_path = f'handwrite/{digit}/testcase'
        
        # 檢查資料夾是否存在
        if not os.path.exists(database_path) or not os.path.exists(testcase_path):
            print(f"警告：數字 {digit} 的資料夾不存在，跳過處理")
            continue
        
        # 處理不同特徵組合的訓練和測試資料
        # 基本特徵
        basic_train_raw = np.vstack((
            process_images_basic(database_path, samples_per_class),
            process_images_basic(testcase_path, samples_per_class)
        ))
        
        basic_answer_raw = np.vstack((
            process_images_basic(database_path, samples_per_class, offset=samples_per_class),
            process_images_basic(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 動差特徵
        moments_train_raw = np.vstack((
            process_images_with_moments(database_path, samples_per_class),
            process_images_with_moments(testcase_path, samples_per_class)
        ))
        
        moments_answer_raw = np.vstack((
            process_images_with_moments(database_path, samples_per_class, offset=samples_per_class),
            process_images_with_moments(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 筆劃強度特徵
        intensity_train_raw = np.vstack((
            process_images_with_intensity(database_path, samples_per_class),
            process_images_with_intensity(testcase_path, samples_per_class)
        ))
        
        intensity_answer_raw = np.vstack((
            process_images_with_intensity(database_path, samples_per_class, offset=samples_per_class),
            process_images_with_intensity(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 侵蝕特徵
        erosion_train_raw = np.vstack((
            process_images_with_erosion(database_path, samples_per_class),
            process_images_with_erosion(testcase_path, samples_per_class)
        ))
        
        erosion_answer_raw = np.vstack((
            process_images_with_erosion(database_path, samples_per_class, offset=samples_per_class),
            process_images_with_erosion(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 全部特徵
        all_train_raw = np.vstack((
            process_images_all(database_path, samples_per_class),
            process_images_all(testcase_path, samples_per_class)
        ))
        
        all_answer_raw = np.vstack((
            process_images_all(database_path, samples_per_class, offset=samples_per_class),
            process_images_all(testcase_path, samples_per_class, offset=samples_per_class)
        ))
        
        # 檢查是否有足夠的數據
        if basic_train_raw.size == 0 or basic_answer_raw.size == 0:
            print(f"警告：數字 {digit} 的資料不足，跳過處理")
            continue
        
        # 計算不同特徵組合的平均值和標準差
        basic_mean, basic_std = calculate_stats(basic_train_raw, 'basic')
        moments_mean, moments_std = calculate_stats(moments_train_raw, 'moments')
        intensity_mean, intensity_std = calculate_stats(intensity_train_raw, 'intensity')
        erosion_mean, erosion_std = calculate_stats(erosion_train_raw, 'erosion')
        all_mean, all_std = calculate_stats(all_train_raw, 'all')
        
        # 標準化特徵
        basic_train = np.array([
            standardize_features_basic(f, basic_mean, basic_std)
            for f in basic_train_raw
        ])
        basic_answer = np.array([
            standardize_features_basic(f, basic_mean, basic_std)
            for f in basic_answer_raw
        ])
        
        moments_train = np.array([
            standardize_features_with_moments(f, moments_mean, moments_std)
            for f in moments_train_raw
        ])
        moments_answer = np.array([
            standardize_features_with_moments(f, moments_mean, moments_std)
            for f in moments_answer_raw
        ])
        
        intensity_train = np.array([
            standardize_features_with_intensity(f, intensity_mean, intensity_std)
            for f in intensity_train_raw
        ])
        intensity_answer = np.array([
            standardize_features_with_intensity(f, intensity_mean, intensity_std)
            for f in intensity_answer_raw
        ])
        
        erosion_train = np.array([
            standardize_features_with_erosion(f, erosion_mean, erosion_std)
            for f in erosion_train_raw
        ])
        erosion_answer = np.array([
            standardize_features_with_erosion(f, erosion_mean, erosion_std)
            for f in erosion_answer_raw
        ])
        
        all_train = np.array([
            standardize_features_all(f, all_mean, all_std)
            for f in all_train_raw
        ])
        all_answer = np.array([
            standardize_features_all(f, all_mean, all_std)
            for f in all_answer_raw
        ])
        
        # 檢查是否有 NaN 值
        has_nan = False
        for data, name in [
            (basic_train, "基本特徵訓練集"),
            (basic_answer, "基本特徵測試集"),
            (moments_train, "動差特徵訓練集"),
            (moments_answer, "動差特徵測試集"),
            (intensity_train, "筆劃強度特徵訓練集"),
            (intensity_answer, "筆劃強度特徵測試集"),
            (erosion_train, "侵蝕特徵訓練集"),
            (erosion_answer, "侵蝕特徵測試集"),
            (all_train, "全部特徵訓練集"),
            (all_answer, "全部特徵測試集")
        ]:
            if np.isnan(data).any():
                print(f"警告：數字 {digit} 的 {name} 中包含 NaN 值！")
                has_nan = True
        
        if has_nan:
            continue
        
        # 建立 SVM 模型
        basic_model = SVC(kernel='linear', random_state=42)
        moments_model = SVC(kernel='linear', random_state=42)
        intensity_model = SVC(kernel='linear', random_state=42)
        erosion_model = SVC(kernel='linear', random_state=42)
        all_model = SVC(kernel='linear', random_state=42)
        
        # 訓練資料的標籤 (前 25 筆為類別 0，後 25 筆為類別 1)
        train_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
        
        # 測試資料的標籤 (前 25 筆為類別 0，後 25 筆為類別 1)
        test_labels = np.array([0] * samples_per_class + [1] * samples_per_class)
        
        # 訓練和測試所有模型
        # 基本特徵
        basic_model.fit(basic_train, train_labels)
        basic_pred = basic_model.predict(basic_answer)
        basic_acc = accuracy_score(test_labels, basic_pred)
        
        # 動差特徵
        moments_model.fit(moments_train, train_labels)
        moments_pred = moments_model.predict(moments_answer)
        moments_acc = accuracy_score(test_labels, moments_pred)
        
        # 筆劃強度特徵
        intensity_model.fit(intensity_train, train_labels)
        intensity_pred = intensity_model.predict(intensity_answer)
        intensity_acc = accuracy_score(test_labels, intensity_pred)
        
        # 侵蝕特徵
        erosion_model.fit(erosion_train, train_labels)
        erosion_pred = erosion_model.predict(erosion_answer)
        erosion_acc = accuracy_score(test_labels, erosion_pred)
        
        # 全部特徵
        all_model.fit(all_train, train_labels)
        all_pred = all_model.predict(all_answer)
        all_acc = accuracy_score(test_labels, all_pred)
        
        # 輸出當前數字的結果
        print(f"數字 {digit} 結果比較:")
        print(f"基本特徵 (原始): {basic_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵: {moments_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵 + 筆劃強度特徵: {intensity_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵 + 侵蝕特徵: {erosion_acc * 100:.2f}%")
        print(f"所有特徵: {all_acc * 100:.2f}%")
        
        # 收集所有預測結果和實際標籤，用於計算總體正確率
        basic_predictions.extend(basic_pred)
        moments_predictions.extend(moments_pred)
        intensity_predictions.extend(intensity_pred)
        erosion_predictions.extend(erosion_pred)
        all_predictions.extend(all_pred)
        all_test_labels.extend(test_labels)

    # 計算總體正確率
    if all_test_labels:
        basic_total_acc = accuracy_score(all_test_labels, basic_predictions)
        moments_total_acc = accuracy_score(all_test_labels, moments_predictions)
        intensity_total_acc = accuracy_score(all_test_labels, intensity_predictions)
        erosion_total_acc = accuracy_score(all_test_labels, erosion_predictions)
        all_total_acc = accuracy_score(all_test_labels, all_predictions)
        
        print("\n===== 總體比較結果 =====")
        print(f"基本特徵 (原始): {basic_total_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵: {moments_total_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵 + 筆劃強度特徵: {intensity_total_acc * 100:.2f}%")
        print(f"基本特徵 + 動差特徵 + 侵蝕特徵: {erosion_total_acc * 100:.2f}%")
        print(f"所有特徵: {all_total_acc * 100:.2f}%")
        
        # 計算各種特徵帶來的改進
        print("\n===== 特徵改進效果 =====")
        print(f"動差特徵提升: {(moments_total_acc - basic_total_acc) * 100:.2f}%")
        print(f"動差 + 筆劃強度特徵提升: {(intensity_total_acc - basic_total_acc) * 100:.2f}%")
        print(f"動差 + 侵蝕特徵提升: {(erosion_total_acc - basic_total_acc) * 100:.2f}%")
        print(f"所有特徵提升: {(all_total_acc - basic_total_acc) * 100:.2f}%")
        
        # 細分析每種新特徵的貢獻
        print("\n===== 各特徵獨立貢獻分析 =====")
        print(f"動差特徵單獨貢獻: {(moments_total_acc - basic_total_acc) * 100:.2f}%")
        print(f"筆劃強度特徵單獨貢獻: {(intensity_total_acc - moments_total_acc) * 100:.2f}%")
        print(f"侵蝕特徵單獨貢獻: {(erosion_total_acc - moments_total_acc) * 100:.2f}%")
        print(f"同時添加筆劃強度和侵蝕的額外貢獻: {(all_total_acc - moments_total_acc - (intensity_total_acc - moments_total_acc) - (erosion_total_acc - moments_total_acc)) * 100:.2f}%")
    else:
        print("警告：沒有任何有效的測試資料")

if __name__ == "__main__":
    main()