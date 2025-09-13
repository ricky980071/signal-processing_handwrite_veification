# -*- coding: utf-8 -*-
# week7.py
# This file主要功能：
# 1. 根據 choice.txt 找到每個數字的標準字跡檔
# 2. 對每個數字資料夾的 database 和 testcase 中的所有字跡找 end points
# 3. 比較標準字跡的 end points 與其他字跡的 end points 相似度
# 4. 可視化比對結果

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from week5_FIN6_19 import (
    extract_strokes, detect_edges, clock_wise_edge_numbering, 
    compute_edge_angles, find_angle_points
)
from week6 import (
    coordinate_normalization_m, coordinate_normalization_n,
    stroke_ratio_m, stroke_ratio_n, binary_array_extraction
)

# 讀取標準字跡檔案名稱
def read_standard_files():
    """從 choice.txt 讀取標準字跡檔案"""
    choice_path = 'handwrite/choice.txt'
    standard_files = {}
    
    with open(choice_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析每行的標準檔案名
    digit = 1
    for line in lines:
        line = line.strip()
        if line and not line.startswith('（'):
            standard_files[digit] = line + '.bmp'
            digit += 1
    
    return standard_files

def extract_endpoint_features(image_path):
    """
    提取圖像的 end points 特徵
    返回: [(theta, xn, yn, x_ratio, y_ratio), ...]
    """
    try:
        # 提取筆劃和邊緣
        stroke_mask = extract_strokes(image_path)
        edges = detect_edges(stroke_mask)
        numbered_edges, all_stroke_coords = clock_wise_edge_numbering(edges)
        theta_arrays = compute_edge_angles(all_stroke_coords, d=12)
        
        # 載入圖像用於正規化計算
        image = Image.open(image_path)
        
        # 收集所有 end points (角度 < 30度)
        endpoint_features = []
        
        for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
            if len(coords) == 0:
                continue
                
            theta_deg = np.degrees(theta_arr)
            # 只取 end points (角度 < 30度)
            ending_pts = find_angle_points(theta_deg, case=1, window=12)
            
            # 過濾角度小於30度的點
            for idx, angle in ending_pts:
                if angle < 30:
                    y, x = coords[idx]
                    
                    # 計算特徵
                    theta_rad = np.radians(angle)
                    xn = coordinate_normalization_m(image, x)
                    yn = coordinate_normalization_n(image, y)
                    x_ratio = stroke_ratio_m(image, x)
                    y_ratio = stroke_ratio_n(image, y)
                    
                    endpoint_features.append((theta_rad, xn, yn, x_ratio, y_ratio))
        
        return endpoint_features
    
    except Exception as e:
        print(f"處理圖像 {image_path} 時發生錯誤: {e}")
        return []

def compute_endpoint_distance(ep1, ep2, w1=8.0, w2=1.0/3000, w3=1.0/10):
    """
    計算兩個 end points 之間的距離
    ep1, ep2: (theta, xn, yn, x_ratio, y_ratio)
    """
    theta1, xn1, yn1, x_ratio1, y_ratio1 = ep1
    theta2, xn2, yn2, x_ratio2, y_ratio2 = ep2
    
    # 角度距離 (考慮週期性)
    theta_diff = abs(theta1 - theta2)
    theta_dist = min(theta_diff, 2 * np.pi - theta_diff)
    
    # 正規化座標距離
    coord_dist = (xn1 - xn2)**2 + (yn1 - yn2)**2
    
    # 累積比例距離
    ratio_dist = (x_ratio1 - x_ratio2)**2 + (y_ratio1 - y_ratio2)**2
    
    # 總距離
    distance_squared = w1 * theta_dist + w2 * coord_dist + w3 * ratio_dist
    
    return np.sqrt(distance_squared)

def compare_endpoints(standard_eps, input_eps, w1=8.0, w2=1.0/1000, w3=1.0/10):
    """
    比較標準字跡和輸入字跡的所有 end points
    返回所有 M×N 個距離值的列表
    """
    distances = []
    
    for std_ep in standard_eps:
        for inp_ep in input_eps:
            dist = compute_endpoint_distance(std_ep, inp_ep, w1, w2, w3)
            distances.append(dist)
    
    return distances

def find_corresponding_points(standard_image_path, test_image_path, L=3, d=12, 
                            direction_weight=8.0, normalized_coordinates_weight=1.0/3000, stroke_ratio_weight=1.0/10):
    """
    找到標準圖像和測試圖像之間的對應點
    參考 distance.py 的邏輯
    """
    # 提取特徵點
    standard_end_points, standard_turning_points = extract_characteristic_points(standard_image_path, L, d)
    test_end_points, test_turning_points = extract_characteristic_points(test_image_path, L, d)
    
    corresponding_points = []
    all_distances = []  # 儲存所有 M×N 個距離
    min_distances = []  # 儲存每個標準點的最小距離 (M個)
    
    def match_points(standard_points, test_points, point_type):
        for i, s_point in enumerate(standard_points):
            min_distance, best_match, best_match_index = float('inf'), None, None
            point_distances = []  # 當前標準點與所有測試點的距離
            
            for j, t_point in enumerate(test_points):
                dist, dir_dist, norm_dist, ratio_dist = distance(
                    s_point, t_point, direction_weight, normalized_coordinates_weight, stroke_ratio_weight
                )
                
                point_distances.append(dist)  # 記錄這個距離
                all_distances.append(dist)   # 加入所有距離列表
                
                # 只有在找到更好的匹配時才更新
                if dist < min_distance:
                    min_distance, best_match, best_match_index = dist, t_point['coordinate'], j
            
            # 記錄這個標準點的最小距離
            if point_distances and point_type == 'end_point':
                min_distances.append(min_distance)
            
            if best_match is not None:
                corresponding_points.append({
                    'type': point_type,
                    'standard_coord': s_point['coordinate'],
                    'test_coord': best_match,
                    'distance': min_distance,
                    'standard_index': i,
                    'test_index': best_match_index
                })
    
    # 匹配 end points 和 turning points
    match_points(standard_end_points, test_end_points, 'end_point')
    match_points(standard_turning_points, test_turning_points, 'turning_point')
    
    return corresponding_points, all_distances, min_distances

def process_all_characters():
    """處理所有字符的特徵點比對 - 包含 M×N 和 M 兩種距離統計"""
    standard_files = read_standard_files()
    results = {}
    
    for digit in range(1, 10):
        print(f"Processing digit {digit}...")
        
        # 取得標準字跡檔案
        if digit not in standard_files:
            print(f"Warning: Cannot find standard file for digit {digit}")
            continue
            
        standard_file = standard_files[digit]
        standard_path = f'handwrite/{digit}/database/{standard_file}'
        
        if not os.path.exists(standard_path):
            print(f"Warning: Standard file {standard_path} does not exist")
            continue
        
        # 提取標準字跡的特徵點
        print(f"  Extracting standard features: {standard_file}")
        standard_end_points, standard_turning_points = extract_characteristic_points(standard_path)
        
        if not standard_end_points:
            print(f"  Warning: No end points found in standard image")
            continue
        
        print(f"  Standard image has {len(standard_end_points)} end points, {len(standard_turning_points)} turning points")
        
        # 收集所有距離 - 兩種方式
        database_all_distances = []      # M×N 方式的所有距離
        database_min_distances = []      # M 方式的最小距離
        testcase_all_distances = []      # M×N 方式的所有距離
        testcase_min_distances = []      # M 方式的最小距離
        
        # 處理 database 中的其他檔案
        database_path = f'handwrite/{digit}/database'
        if os.path.exists(database_path):
            for filename in os.listdir(database_path):
                if filename.endswith('.bmp') and filename != standard_file:
                    file_path = os.path.join(database_path, filename)
                    corresponding_points, all_dists, min_dists = find_corresponding_points(standard_path, file_path)
                    
                    # M×N 方式：使用 all_dists（包含所有 M×N 個距離）
                    database_all_distances.extend(all_dists)
                    # M 方式：使用 min_dists（每個標準點的最小距離）
                    database_min_distances.extend(min_dists)
        
        # 處理 testcase 中的檔案
        testcase_path = f'handwrite/{digit}/testcase'
        if os.path.exists(testcase_path):
            for filename in os.listdir(testcase_path):
                if filename.endswith('.bmp'):
                    file_path = os.path.join(testcase_path, filename)
                    corresponding_points, all_dists, min_dists = find_corresponding_points(standard_path, file_path)
                    
                    # M×N 方式：使用 all_dists（包含所有 M×N 個距離）
                    testcase_all_distances.extend(all_dists)
                    # M 方式：使用 min_dists（每個標準點的最小距離）
                    testcase_min_distances.extend(min_dists)
        
        # 計算平均距離 - 兩種方式
        # M×N 方式 (原始方式)
        avg_database_all = np.mean(database_all_distances) if database_all_distances else 0
        avg_testcase_all = np.mean(testcase_all_distances) if testcase_all_distances else 0
        avg_all_all = np.mean(database_all_distances + testcase_all_distances) if (database_all_distances + testcase_all_distances) else 0
        
        # M 方式 (最小距離方式)
        avg_database_min = np.mean(database_min_distances) if database_min_distances else 0
        avg_testcase_min = np.mean(testcase_min_distances) if testcase_min_distances else 0
        avg_all_min = np.mean(database_min_distances + testcase_min_distances) if (database_min_distances + testcase_min_distances) else 0
        
        results[digit] = {
            # M×N 方式結果
            'database_avg_all': avg_database_all,
            'testcase_avg_all': avg_testcase_all,
            'all_avg_all': avg_all_all,
            'database_count_all': len(database_all_distances),
            'testcase_count_all': len(testcase_all_distances),
            
            # M 方式結果
            'database_avg_min': avg_database_min,
            'testcase_avg_min': avg_testcase_min,
            'all_avg_min': avg_all_min,
            'database_count_min': len(database_min_distances),
            'testcase_count_min': len(testcase_min_distances),
            
            # 標準字跡信息
            'standard_end_points_count': len(standard_end_points),
            'standard_turning_points_count': len(standard_turning_points)
        }
        
        print(f"  Database avg distance (M×N): {avg_database_all:.4f} ({len(database_all_distances)} comparisons)")
        print(f"  Database avg distance (M):   {avg_database_min:.4f} ({len(database_min_distances)} comparisons)")
        print(f"  Testcase avg distance (M×N): {avg_testcase_all:.4f} ({len(testcase_all_distances)} comparisons)")
        print(f"  Testcase avg distance (M):   {avg_testcase_min:.4f} ({len(testcase_min_distances)} comparisons)")
        print(f"  Overall avg distance (M×N):  {avg_all_all:.4f}")
        print(f"  Overall avg distance (M):    {avg_all_min:.4f}")
    
    return results

def extract_feature_vectors(standard_image_path, test_images_paths, L=3, d=12, 
                           direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
    """
    提取特徵向量：對於一個標準圖像，計算與多個測試圖像的 M 維特徵向量 - 支援自訂權重
    
    Parameters:
    - standard_image_path: 標準圖像路徑
    - test_images_paths: 測試圖像路徑列表
    - 其他參數同 find_corresponding_points
    
    Returns:
    - feature_vectors: list of M-dimensional vectors, 每個測試圖像對應一個 M 維向量
    - M: 標準圖像的 end points 數量
    """
    # 提取標準圖像的特徵點
    standard_end_points, _ = extract_characteristic_points(standard_image_path, L, d)
    M = len(standard_end_points)
    
    if M == 0:
        print(f"Warning: No end points found in standard image {standard_image_path}")
        return [], 0
    
    feature_vectors = []
    
    for test_path in test_images_paths:
        try:
            # 對每個測試圖像計算對應點（使用指定權重）
            _, _, min_distances = find_corresponding_points(standard_image_path, test_path, L, d,
                                                          direction_weight, normalized_coordinates_weight, stroke_ratio_weight)
            
            # min_distances 應該有 M 個元素（每個標準 end point 的最小距離）
            if len(min_distances) == M:
                feature_vectors.append(min_distances)
            else:
                print(f"Warning: Expected {M} distances but got {len(min_distances)} for {test_path}")
                # 補齊或截斷到 M 維
                if len(min_distances) < M:
                    padded_distances = min_distances + [float('inf')] * (M - len(min_distances))
                    feature_vectors.append(padded_distances)
                else:
                    feature_vectors.append(min_distances[:M])
        except Exception as e:
            print(f"Error processing {test_path}: {e}")
            # 添加一個異常向量
            feature_vectors.append([float('inf')] * M)
    
    return feature_vectors, M

def process_all_characters_with_vectors(direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
    """處理所有字符並生成特徵向量，同時保留原有的統計功能 - 支援自訂權重"""
    standard_files = read_standard_files()
    results = {}
    all_feature_vectors = {}  # 新增：儲存所有特徵向量
    
    for digit in range(1, 10):
        print(f"Processing digit {digit}...")
        
        # 取得標準字跡檔案
        if digit not in standard_files:
            print(f"Warning: Cannot find standard file for digit {digit}")
            continue
            
        standard_file = standard_files[digit]
        standard_path = f'handwrite/{digit}/database/{standard_file}'
        
        if not os.path.exists(standard_path):
            print(f"Warning: Standard file {standard_path} does not exist")
            continue
        
        # 提取標準字跡的特徵點
        print(f"  Extracting standard features: {standard_file}")
        standard_end_points, standard_turning_points = extract_characteristic_points(standard_path)
        
        if not standard_end_points:
            print(f"  Warning: No end points found in standard image")
            continue
        
        M = len(standard_end_points)
        print(f"  Standard image has {M} end points, {len(standard_turning_points)} turning points")
        
        # 收集所有測試圖像路徑
        database_images = []
        testcase_images = []
        
        # 收集 database 中的其他檔案路徑
        database_path = f'handwrite/{digit}/database'
        if os.path.exists(database_path):
            for filename in os.listdir(database_path):
                if filename.endswith('.bmp') and filename != standard_file:
                    database_images.append(os.path.join(database_path, filename))
        
        # 收集 testcase 中的檔案路徑
        testcase_path = f'handwrite/{digit}/testcase'
        if os.path.exists(testcase_path):
            for filename in os.listdir(testcase_path):
                if filename.endswith('.bmp'):
                    testcase_images.append(os.path.join(testcase_path, filename))
        
        # 提取特徵向量（使用指定權重）
        print(f"  Extracting feature vectors for {len(database_images)} database images...")
        database_vectors, M_db = extract_feature_vectors(standard_path, database_images, 
                                                        direction_weight=direction_weight,
                                                        normalized_coordinates_weight=normalized_coordinates_weight,
                                                        stroke_ratio_weight=stroke_ratio_weight)
        
        print(f"  Extracting feature vectors for {len(testcase_images)} testcase images...")
        testcase_vectors, M_tc = extract_feature_vectors(standard_path, testcase_images,
                                                        direction_weight=direction_weight,
                                                        normalized_coordinates_weight=normalized_coordinates_weight,
                                                        stroke_ratio_weight=stroke_ratio_weight)
        
        # 儲存特徵向量
        all_feature_vectors[digit] = {
            'standard_file': standard_file,
            'M': M,
            'database_vectors': database_vectors,
            'database_files': [os.path.basename(path) for path in database_images],
            'testcase_vectors': testcase_vectors,
            'testcase_files': [os.path.basename(path) for path in testcase_images],
            'all_vectors': database_vectors + testcase_vectors,
            'all_files': [os.path.basename(path) for path in database_images + testcase_images]
        }
        
        # 原有的統計計算（保持不變，使用指定權重）
        database_all_distances = []
        database_min_distances = []
        testcase_all_distances = []
        testcase_min_distances = []
        
        for img_path in database_images:
            corresponding_points, all_dists, min_dists = find_corresponding_points(standard_path, img_path,
                                                                                 direction_weight=direction_weight,
                                                                                 normalized_coordinates_weight=normalized_coordinates_weight,
                                                                                 stroke_ratio_weight=stroke_ratio_weight)
            database_all_distances.extend(all_dists)
            database_min_distances.extend(min_dists)
        
        for img_path in testcase_images:
            corresponding_points, all_dists, min_dists = find_corresponding_points(standard_path, img_path,
                                                                                 direction_weight=direction_weight,
                                                                                 normalized_coordinates_weight=normalized_coordinates_weight,
                                                                                 stroke_ratio_weight=stroke_ratio_weight)
            testcase_all_distances.extend(all_dists)
            testcase_min_distances.extend(min_dists)
        
        # 計算平均距離
        avg_database_all = np.mean(database_all_distances) if database_all_distances else 0
        avg_testcase_all = np.mean(testcase_all_distances) if testcase_all_distances else 0
        avg_all_all = np.mean(database_all_distances + testcase_all_distances) if (database_all_distances + testcase_all_distances) else 0
        
        avg_database_min = np.mean(database_min_distances) if database_min_distances else 0
        avg_testcase_min = np.mean(testcase_min_distances) if testcase_min_distances else 0
        avg_all_min = np.mean(database_min_distances + testcase_min_distances) if (database_min_distances + testcase_min_distances) else 0
        
        results[digit] = {
            # M×N 方式結果
            'database_avg_all': avg_database_all,
            'testcase_avg_all': avg_testcase_all,
            'all_avg_all': avg_all_all,
            'database_count_all': len(database_all_distances),
            'testcase_count_all': len(testcase_all_distances),
            
            # M 方式結果
            'database_avg_min': avg_database_min,
            'testcase_avg_min': avg_testcase_min,
            'all_avg_min': avg_all_min,
            'database_count_min': len(database_min_distances),
            'testcase_count_min': len(testcase_min_distances),
            
            # 標準字跡信息
            'standard_end_points_count': len(standard_end_points),
            'standard_turning_points_count': len(standard_turning_points),
            
            # 新增：特徵向量信息
            'vector_dimension': M,
            'database_vector_count': len(database_vectors),
            'testcase_vector_count': len(testcase_vectors),
            'total_vector_count': len(database_vectors) + len(testcase_vectors)
        }
        
        print(f"  Database avg distance (M×N): {avg_database_all:.4f} ({len(database_all_distances)} comparisons)")
        print(f"  Database avg distance (M):   {avg_database_min:.4f} ({len(database_min_distances)} comparisons)")
        print(f"  Testcase avg distance (M×N): {avg_testcase_all:.4f} ({len(testcase_all_distances)} comparisons)")
        print(f"  Testcase avg distance (M):   {avg_testcase_min:.4f} ({len(testcase_min_distances)} comparisons)")
        print(f"  Overall avg distance (M×N):  {avg_all_all:.4f}")
        print(f"  Overall avg distance (M):    {avg_all_min:.4f}")
        print(f"  Feature vectors: {M}-dimensional, {len(database_vectors)} database + {len(testcase_vectors)} testcase = {len(database_vectors) + len(testcase_vectors)} total")
    
    return results, all_feature_vectors

def save_feature_vectors(all_feature_vectors, output_dir='feature_vectors'):
    """
    將特徵向量保存到檔案
    
    Parameters:
    - all_feature_vectors: 從 process_all_characters_with_vectors 返回的特徵向量字典
    - output_dir: 輸出目錄
    """
    import os
    import csv
    
    # 創建輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 為每個數字保存特徵向量
    for digit, data in all_feature_vectors.items():
        # 保存為 CSV 格式
        csv_path = os.path.join(output_dir, f'digit_{digit}_feature_vectors.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 寫入標題行
            header = ['filename', 'type'] + [f'distance_{i+1}' for i in range(data['M'])]
            writer.writerow(header)
            
            # 寫入 database 向量
            for filename, vector in zip(data['database_files'], data['database_vectors']):
                row = [filename, 'database'] + vector
                writer.writerow(row)
            
            # 寫入 testcase 向量
            for filename, vector in zip(data['testcase_files'], data['testcase_vectors']):
                row = [filename, 'testcase'] + vector
                writer.writerow(row)
        
        print(f"Saved {len(data['all_vectors'])} feature vectors for digit {digit} to {csv_path}")
        
        # 保存為 NumPy 格式（便於 SVM 使用）
        npy_path = os.path.join(output_dir, f'digit_{digit}_feature_vectors.npy')
        vectors_array = np.array(data['all_vectors'])
        np.save(npy_path, vectors_array)
        
        # 保存對應的標籤和檔案名
        labels_path = os.path.join(output_dir, f'digit_{digit}_labels.npy')
        labels = ['database'] * len(data['database_vectors']) + ['testcase'] * len(data['testcase_vectors'])
        np.save(labels_path, labels)
        
        filenames_path = os.path.join(output_dir, f'digit_{digit}_filenames.npy')
        np.save(filenames_path, data['all_files'])
        
        print(f"Saved NumPy arrays: {npy_path}, {labels_path}, {filenames_path}")
    
    # 保存所有數字的彙總信息
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Feature Vectors Summary\n")
        f.write("======================\n\n")
        for digit, data in all_feature_vectors.items():
            f.write(f"Digit {digit}:\n")
            f.write(f"  Standard file: {data['standard_file']}\n")
            f.write(f"  Vector dimension (M): {data['M']}\n")
            f.write(f"  Database vectors: {len(data['database_vectors'])}\n")
            f.write(f"  Testcase vectors: {len(data['testcase_vectors'])}\n")
            f.write(f"  Total vectors: {len(data['all_vectors'])}\n\n")
    
    print(f"Saved summary to {summary_path}")

def visualize_feature_vectors(all_feature_vectors):
    """可視化特徵向量的分布"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (digit, data) in enumerate(all_feature_vectors.items()):
        if i >= 9:
            break
            
        ax = axes[i]
        
        # 計算每個向量的平均值（作為一個簡單的統計量）
        db_means = [np.mean(vec) for vec in data['database_vectors']]
        tc_means = [np.mean(vec) for vec in data['testcase_vectors']]
        
        # 散佈圖
        if db_means:
            ax.scatter(range(len(db_means)), db_means, alpha=0.7, label='Database', color='blue', s=30)
        if tc_means:
            ax.scatter(range(len(db_means), len(db_means) + len(tc_means)), tc_means, 
                      alpha=0.7, label='Testcase', color='red', s=30)
        
        ax.set_title(f'Digit {digit} Feature Vector Means\n(M={data["M"]}, N={len(data["all_vectors"])})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Average Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 維度分布統計
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    digits = list(all_feature_vectors.keys())
    dimensions = [all_feature_vectors[d]['M'] for d in digits]
    plt.bar(digits, dimensions, alpha=0.7, color='green')
    plt.xlabel('Digit')
    plt.ylabel('Vector Dimension (M)')
    plt.title('Feature Vector Dimensions by Digit')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    sample_counts = [len(all_feature_vectors[d]['all_vectors']) for d in digits]
    plt.bar(digits, sample_counts, alpha=0.7, color='orange')
    plt.xlabel('Digit')
    plt.ylabel('Number of Samples')
    plt.title('Number of Feature Vectors by Digit')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def direction_distance(angle1, angle2):
    """角度距離計算 - 考慮週期性"""
    angle_diff = np.abs(angle1 - angle2)
    angle_diff_complement = 2 * np.pi - angle_diff
    return min(angle_diff, angle_diff_complement)

def normalized_coordinates_distance(m_hat1, n_hat1, m_hat2, n_hat2):
    """正規化座標距離"""
    return ((m_hat1 - m_hat2) ** 2 + (n_hat1 - n_hat2) ** 2)

def stroke_ratio_distance(stroke_ratio_m1, stroke_ratio_n1, stroke_ratio_m2, stroke_ratio_n2):
    """累積比例距離"""
    return ((stroke_ratio_m1 - stroke_ratio_m2) ** 2 + (stroke_ratio_n1 - stroke_ratio_n2) ** 2)

def distance(point1, point2, direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
    """
    計算兩個特徵點之間的距離
    使用指定的權重設定
    """
    dir_dist = direction_distance(point1['angle'], point2['angle'])
    norm_coord_dist = normalized_coordinates_distance(
        point1['m_hat'], point1['n_hat'], 
        point2['m_hat'], point2['n_hat']
    )
    stroke_ratio_dist = stroke_ratio_distance(
        point1['m_ratio'], point1['n_ratio'],
        point2['m_ratio'], point2['n_ratio']
    )
    
    total_distance = (direction_weight * dir_dist +
                     normalized_coordinates_weight * norm_coord_dist +
                     stroke_ratio_weight * stroke_ratio_dist)
    
    return total_distance, direction_weight * dir_dist, normalized_coordinates_weight * norm_coord_dist, stroke_ratio_weight * stroke_ratio_dist

def extract_characteristic_points(image_path, L=3, d=12):
    """
    提取特徵點 (end points 和 turning points)
    參考 distance.py 的 characteristic_point_extraction 邏輯
    但使用 week5_FIN6_19 的函數
    """
    try:
        # 使用 week5_FIN6_19 提取筆劃和特徵點
        stroke_mask = extract_strokes(image_path)
        edges = detect_edges(stroke_mask)
        numbered_edges, all_stroke_coords = clock_wise_edge_numbering(edges)
        theta_arrays = compute_edge_angles(all_stroke_coords, d=d)
        
        # 載入圖像並取得二值化陣列
        image = Image.open(image_path)
        binary_array = binary_array_extraction(image)
        stroke_indices = np.where(binary_array > 0)
        
        end_points = []
        turning_points = []
        
        # 處理每個部件
        for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
            if len(coords) == 0:
                continue
                
            theta_deg = np.degrees(theta_arr)
            
            # 找 end points (角度 < 35度)
            ending_pts = find_angle_points(theta_deg, case=1, window=12)
            for idx, angle in ending_pts:
                if angle < 35:  # 使用 distance.py 的閾值
                    n, m = coords[idx]  # coords 是 (y, x) 格式
                    
                    # 計算方向特徵 (使用 week6 的卷積方法)
                    phi_mat = phi_matrix(L)
                    h, w = binary_array.shape
                    
                    # 確保座標在有效範圍內
                    if n < L or n >= h - L or m < L or m >= w - L:
                        angle_at_point = 0  # 邊界點返回 0
                    else:
                        # 提取區域
                        region = binary_array[n-L:n+L+1, m-L:m+L+1]
                        # 卷積計算
                        convolved_value = np.sum(region * phi_mat)
                        # 返回角度
                        angle_at_point = np.angle(convolved_value)
                    
                    # 計算正規化特徵
                    m_hat = coordinate_normalization_m_custom(m, stroke_indices)
                    n_hat = coordinate_normalization_n_custom(n, stroke_indices)
                    m_ratio = stroke_ratio_m_custom(m, stroke_indices)
                    n_ratio = stroke_ratio_n_custom(n, stroke_indices)
                    
                    end_points.append({
                        'coordinate': (n, m),  # (row, col) 格式
                        'angle': angle_at_point,
                        'm_hat': m_hat,
                        'n_hat': n_hat,
                        'm_ratio': m_ratio,
                        'n_ratio': n_ratio
                    })
            
            # 找 turning points (35-150度)
            turning_pts = find_angle_points(theta_deg, case=2, window=12)
            for idx, angle in turning_pts:
                if 35 <= angle <= 150:  # 使用 distance.py 的閾值
                    n, m = coords[idx]  # coords 是 (y, x) 格式
                    
                    # 計算方向特徵
                    phi_mat = phi_matrix(L)
                    h, w = binary_array.shape
                    
                    # 確保座標在有效範圍內
                    if n < L or n >= h - L or m < L or m >= w - L:
                        angle_at_point = 0  # 邊界點返回 0
                    else:
                        # 提取區域
                        region = binary_array[n-L:n+L+1, m-L:m+L+1]
                        # 卷積計算
                        convolved_value = np.sum(region * phi_mat)
                        # 返回角度
                        angle_at_point = np.angle(convolved_value)
                    
                    # 計算正規化特徵
                    m_hat = coordinate_normalization_m_custom(m, stroke_indices)
                    n_hat = coordinate_normalization_n_custom(n, stroke_indices)
                    m_ratio = stroke_ratio_m_custom(m, stroke_indices)
                    n_ratio = stroke_ratio_n_custom(n, stroke_indices)
                    
                    turning_points.append({
                        'coordinate': (n, m),  # (row, col) 格式
                        'angle': angle_at_point,
                        'm_hat': m_hat,
                        'n_hat': n_hat,
                        'm_ratio': m_ratio,
                        'n_ratio': n_ratio
                    })
        
        return end_points, turning_points
    
    except Exception as e:
        print(f"處理圖像 {image_path} 時發生錯誤: {e}")
        return [], []

def phi(m, n, L):
    """複數遮罩的單點值"""
    if m == 0 and n == 0:
        return 0
    else:
        denominator = np.sqrt(m**2 + n**2)
        return (n - 1j * m) / denominator

def phi_matrix(L):
    """產生複數遮罩矩陣"""
    size = 2 * L + 1
    phi_matrix = np.zeros((size, size), dtype=complex)
    for i, m in enumerate(range(-L, L + 1)):
        for j, n in enumerate(range(-L, L + 1)):
            phi_matrix[i, j] = phi(m, n, L)
    return phi_matrix

def coordinate_normalization_m_custom(m, stroke_indices):
    """x座標正規化 - 使用與 distance.py 相同的邏輯"""
    m_min = m_max = 0
    if len(stroke_indices[0]) > 0:
        m_min = np.min(stroke_indices[1])  # x座標 (column)
        m_max = np.max(stroke_indices[1])
    if m_max > m_min:
        m_hat = (m - m_min) / (m_max - m_min) * 100
    else:
        m_hat = 0
    return m_hat

def coordinate_normalization_n_custom(n, stroke_indices):
    """y座標正規化 - 使用與 distance.py 相同的邏輯"""
    n_min = n_max = 0
    if len(stroke_indices[0]) > 0:
        n_min = np.min(stroke_indices[0])  # y座標 (row)
        n_max = np.max(stroke_indices[0])
    if n_max > n_min:
        n_hat = (n - n_min) / (n_max - n_min) * 100
    else:
        n_hat = 0
    return n_hat

def stroke_ratio_m_custom(m, stroke_indices):
    """x累積比例 - 使用與 distance.py 相同的邏輯"""
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0
    m_ratio = np.sum(stroke_indices[1] <= m) / total_stroke_pixels
    return m_ratio

def stroke_ratio_n_custom(n, stroke_indices):
    """y累積比例 - 使用與 distance.py 相同的邏輯"""
    total_stroke_pixels = len(stroke_indices[0])
    if total_stroke_pixels == 0:
        return 0
    n_ratio = np.sum(stroke_indices[0] <= n) / total_stroke_pixels
    return n_ratio

def main():
    """主函數 - 增強版，可調整權重"""
    print("Starting End Points Comparison with Feature Vector Extraction...")
    print("Using distance.py logic with week5_FIN6_19 and week6 functions")
    print("Comparing M×N method vs M method (minimum distances)")
    print("Extracting M-dimensional feature vectors for SVM...")
    
    # ========== 權重設定區域 - 可以隨時修改 ==========
    DIRECTION_WEIGHT = 10.0
    NORMALIZED_COORDINATES_WEIGHT = 1.0/1000  # 修改為 1/1000
    STROKE_RATIO_WEIGHT = 1.0/10
    
    print(f"\nCurrent weights:")
    print(f"  Direction weight: {DIRECTION_WEIGHT}")
    print(f"  Normalized coordinates weight: {NORMALIZED_COORDINATES_WEIGHT}")
    print(f"  Stroke ratio weight: {STROKE_RATIO_WEIGHT}")
    print("=" * 50)
    # ============================================
    
    # 確認必要的資料夾存在
    if not os.path.exists('handwrite'):
        print("Error: Cannot find handwrite folder")
        return
    
    if not os.path.exists('handwrite/choice.txt'):
        print("Error: Cannot find choice.txt file")
        return
    
    # 處理所有字符並提取特徵向量（使用指定權重）
    results, all_feature_vectors = process_all_characters_with_vectors(
        direction_weight=DIRECTION_WEIGHT,
        normalized_coordinates_weight=NORMALIZED_COORDINATES_WEIGHT,
        stroke_ratio_weight=STROKE_RATIO_WEIGHT
    )
    
    if results:
        # 原有的可視化
        visualize_results(results)
        
        # 新增：特徵向量可視化
        visualize_feature_vectors(all_feature_vectors)
        
        # 保存特徵向量
        save_feature_vectors(all_feature_vectors, output_dir=f'feature_vectors_w{DIRECTION_WEIGHT}_{NORMALIZED_COORDINATES_WEIGHT}_{STROKE_RATIO_WEIGHT}')
        
        # 打印特徵向量統計
        print("\n=== Feature Vector Statistics ===")
        print(f"{'Digit':<5} {'Dimension(M)':<12} {'DB_Vectors':<10} {'TC_Vectors':<10} {'Total':<8}")
        print("-" * 50)
        
        for digit in sorted(all_feature_vectors.keys()):
            data = all_feature_vectors[digit]
            print(f"{digit:<5} {data['M']:<12} {len(data['database_vectors']):<10} {len(data['testcase_vectors']):<10} {len(data['all_vectors']):<8}")
    else:
        print("No characters processed successfully")

def visualize_results(results):
    """可視化比對結果 - 比較 M×N 和 M 兩種方式"""
    digits = list(results.keys())
    
    # M×N 方式的結果
    database_avgs_all = [results[d]['database_avg_all'] for d in digits]
    testcase_avgs_all = [results[d]['testcase_avg_all'] for d in digits]
    all_avgs_all = [results[d]['all_avg_all'] for d in digits]
    
    # M 方式的結果
    database_avgs_min = [results[d]['database_avg_min'] for d in digits]
    testcase_avgs_min = [results[d]['testcase_avg_min'] for d in digits]
    all_avgs_min = [results[d]['all_avg_min'] for d in digits]
    
    plt.figure(figsize=(20, 12))
    
    # 1. M×N vs M 方式比較 - Database
    plt.subplot(3, 3, 1)
    x = np.arange(len(digits))
    width = 0.35
    
    plt.bar(x - width/2, database_avgs_all, width, label='M×N Method', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, database_avgs_min, width, label='M Method (Min)', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Average Distance')
    plt.title('Database: M×N vs M Method Comparison')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. M×N vs M 方式比較 - Testcase
    plt.subplot(3, 3, 2)
    plt.bar(x - width/2, testcase_avgs_all, width, label='M×N Method', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, testcase_avgs_min, width, label='M Method (Min)', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Average Distance')
    plt.title('Testcase: M×N vs M Method Comparison')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. M×N vs M 方式比較 - Overall
    plt.subplot(3, 3, 3)
    plt.bar(x - width/2, all_avgs_all, width, label='M×N Method', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, all_avgs_min, width, label='M Method (Min)', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Average Distance')
    plt.title('Overall: M×N vs M Method Comparison')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. M×N 方式的詳細結果
    plt.subplot(3, 3, 4)
    width = 0.25
    plt.bar(x - width, database_avgs_all, width, label='Database Avg', alpha=0.8, color='green')
    plt.bar(x, testcase_avgs_all, width, label='Testcase Avg', alpha=0.8, color='orange')
    plt.bar(x + width, all_avgs_all, width, label='Overall Avg', alpha=0.8, color='purple')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Average Distance')
    plt.title('M×N Method: Detailed Results')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. M 方式的詳細結果
    plt.subplot(3, 3, 5)
    plt.bar(x - width, database_avgs_min, width, label='Database Avg', alpha=0.8, color='green')
    plt.bar(x, testcase_avgs_min, width, label='Testcase Avg', alpha=0.8, color='orange')
    plt.bar(x + width, all_avgs_min, width, label='Overall Avg', alpha=0.8, color='purple')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Average Distance')
    plt.title('M Method: Detailed Results')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Database vs Testcase 散佈圖 - M×N 方式
    plt.subplot(3, 3, 6)
    plt.scatter(database_avgs_all, testcase_avgs_all, alpha=0.7, s=100, color='blue', label='M×N Method')
    for i, digit in enumerate(digits):
        plt.annotate(str(digit), (database_avgs_all[i], testcase_avgs_all[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Database Average Distance')
    plt.ylabel('Testcase Average Distance')
    plt.title('Database vs Testcase Scatter (M×N Method)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 7. Database vs Testcase 散佈圖 - M 方式
    plt.subplot(3, 3, 7)
    plt.scatter(database_avgs_min, testcase_avgs_min, alpha=0.7, s=100, color='red', label='M Method')
    for i, digit in enumerate(digits):
        plt.annotate(str(digit), (database_avgs_min[i], testcase_avgs_min[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Database Average Distance')
    plt.ylabel('Testcase Average Distance')
    plt.title('Database vs Testcase Scatter (M Method)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 8. 比對數量統計
    plt.subplot(3, 3, 8)
    database_counts_all = [results[d]['database_count_all'] for d in digits]
    database_counts_min = [results[d]['database_count_min'] for d in digits]
    
    plt.bar(x - width/2, database_counts_all, width, label='M×N Method', alpha=0.8)
    plt.bar(x + width/2, database_counts_min, width, label='M Method', alpha=0.8)
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Number of Comparisons')
    plt.title('Comparison Count Statistics')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. 標準字跡特徵點數量
    plt.subplot(3, 3, 9)
    std_end_counts = [results[d]['standard_end_points_count'] for d in digits]
    std_turning_counts = [results[d]['standard_turning_points_count'] for d in digits]
    
    plt.bar(x - width/2, std_end_counts, width, label='End Points', alpha=0.8, color='blue')
    plt.bar(x + width/2, std_turning_counts, width, label='Turning Points', alpha=0.8, color='orange')
    
    plt.xlabel('Digit (Character)')
    plt.ylabel('Feature Point Count')
    plt.title('Standard Image Feature Point Count')
    plt.xticks(x, digits)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 詳細結果表格
    print("\n=== Detailed Results Statistics ===")
    print(f"{'Digit':<5} {'DB_Avg(M×N)':<12} {'DB_Avg(M)':<10} {'TC_Avg(M×N)':<12} {'TC_Avg(M)':<10} {'All_Avg(M×N)':<12} {'All_Avg(M)':<10} {'Std_EP':<7} {'Std_TP':<7}")
    print("-" * 100)
    
    for digit in digits:
        r = results[digit]
        print(f"{digit:<5} {r['database_avg_all']:<12.4f} {r['database_avg_min']:<10.4f} {r['testcase_avg_all']:<12.4f} {r['testcase_avg_min']:<10.4f} "
              f"{r['all_avg_all']:<12.4f} {r['all_avg_min']:<10.4f} {r['standard_end_points_count']:<7} {r['standard_turning_points_count']:<7}")
    
    # 方法比較摘要
    print("\n=== Method Comparison Summary ===")
    avg_reduction_db = np.mean([(results[d]['database_avg_all'] - results[d]['database_avg_min']) / results[d]['database_avg_all'] * 100 
                               for d in digits if results[d]['database_avg_all'] > 0])
    avg_reduction_tc = np.mean([(results[d]['testcase_avg_all'] - results[d]['testcase_avg_min']) / results[d]['testcase_avg_all'] * 100 
                               for d in digits if results[d]['testcase_avg_all'] > 0])
    
    print(f"Average distance reduction by M method:")
    print(f"  Database: {avg_reduction_db:.2f}%")
    print(f"  Testcase: {avg_reduction_tc:.2f}%")

if __name__ == "__main__":
    main()

