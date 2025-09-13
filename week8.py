# -*- coding: utf-8 -*-
# week8.py
# 這個檔案主要功能：
# 1. 使用 week7 的方法，根據標準字篩選可靠的特徵點
# 2. 提取座標特徵和相對位置特徵
# 3. 結合 week3 的所有特徵進行 SVM 訓練
# 4. 對每個數字獨立訓練 SVM 模型

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from week7 import (
    read_standard_files, extract_characteristic_points, distance,
    phi_matrix, coordinate_normalization_m_custom, coordinate_normalization_n_custom,
    stroke_ratio_m_custom, stroke_ratio_n_custom
)
from week3 import extract_features_all

def find_reliable_feature_points(standard_image_path, database_images, L=3, d=12,
                                direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
    """
    根據標準字與 database 中前5個字的比對結果，篩選出可靠的特徵點
    
    Parameters:
    - standard_image_path: 標準字跡路徑
    - database_images: database 中的圖像路徑列表（至少5個）
    - 其他參數：距離計算相關參數
    
    Returns:
    - reliable_end_points: 可靠的 end points 列表
    - reliable_turning_points: 可靠的 turning points 列表
    """
    print(f"  Analyzing reliable feature points for standard image...")
    
    # 提取標準圖像的特徵點
    standard_end_points, standard_turning_points = extract_characteristic_points(standard_image_path, L, d)
    
    if not standard_end_points:
        print(f"  Warning: No end points found in standard image")
        return [], []
    
    print(f"  Standard image has {len(standard_end_points)} end points, {len(standard_turning_points)} turning points")
    
    # 只使用前5個 database 圖像進行分析
    analysis_images = database_images[:5]
    print(f"  Using {len(analysis_images)} images for reliability analysis")
    
    # 分析 end points 的可靠性
    reliable_end_points = analyze_point_reliability(
        standard_end_points, analysis_images, standard_image_path, 'end_point',
        L, d, direction_weight, normalized_coordinates_weight, stroke_ratio_weight
    )
    
    # 分析 turning points 的可靠性
    reliable_turning_points = analyze_point_reliability(
        standard_turning_points, analysis_images, standard_image_path, 'turning_point',
        L, d, direction_weight, normalized_coordinates_weight, stroke_ratio_weight
    )
    
    print(f"  Reliable points: {len(reliable_end_points)} end points, {len(reliable_turning_points)} turning points")
    
    return reliable_end_points, reliable_turning_points

def analyze_point_reliability(standard_points, analysis_images, standard_image_path, point_type,
                            L, d, direction_weight, normalized_coordinates_weight, stroke_ratio_weight):
    """
    分析特徵點的可靠性
    """
    reliable_points = []
    
    for i, std_point in enumerate(standard_points):
        distances = []
        matched_points = []
        
        # 對每個分析圖像找到最佳匹配點
        for img_path in analysis_images:
            try:
                # 提取測試圖像的特徵點
                test_end_points, test_turning_points = extract_characteristic_points(img_path, L, d)
                
                if point_type == 'end_point':
                    test_points = test_end_points
                else:
                    test_points = test_turning_points
                
                if not test_points:
                    continue
                
                # 找到最佳匹配點
                min_distance = float('inf')
                best_match = None
                
                for test_point in test_points:
                    dist, _, _, _ = distance(
                        std_point, test_point, direction_weight, 
                        normalized_coordinates_weight, stroke_ratio_weight
                    )
                    
                    if dist < min_distance:
                        min_distance = dist
                        best_match = test_point
                
                if best_match is not None:
                    distances.append(min_distance)
                    matched_points.append(best_match)
                    
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue
        
        # 檢查這個標準點是否可靠
        if len(distances) >= 3:  # 至少要有3個成功匹配
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # 檢查是否有距離超過平均值 + 1個標準差
            outliers = [d for d in distances if d > mean_dist + std_dist]
            
            # 如果異常值少於總數的40%，認為這個點是可靠的
            if len(outliers) / len(distances) < 0.4:
                reliable_points.append({
                    'point': std_point,
                    'index': i,
                    'mean_distance': mean_dist,
                    'std_distance': std_dist,
                    'match_count': len(distances)
                })
                print(f"    {point_type} {i+1}: reliable (mean_dist={mean_dist:.4f}, std_dist={std_dist:.4f}, matches={len(distances)})")
            else:
                print(f"    {point_type} {i+1}: unreliable (too many outliers: {len(outliers)}/{len(distances)})")
        else:
            print(f"    {point_type} {i+1}: unreliable (insufficient matches: {len(distances)})")
    
    return reliable_points

def extract_coordinate_features(reliable_end_points, reliable_turning_points):
    """
    提取座標特徵 (2K 個特徵)
    """
    coordinate_features = []
    
    # 提取 end points 的座標
    for point_info in reliable_end_points:
        coord = point_info['point']['coordinate']
        coordinate_features.extend([coord[1], coord[0]])  # (x, y) = (col, row)
    
    # 提取 turning points 的座標
    for point_info in reliable_turning_points:
        coord = point_info['point']['coordinate']
        coordinate_features.extend([coord[1], coord[0]])  # (x, y) = (col, row)
    
    return np.array(coordinate_features)

def extract_relative_position_features(reliable_end_points, reliable_turning_points):
    """
    提取相對位置特徵 K(K-1) 個特徵
    """
    relative_features = []
    
    # 收集所有可靠的特徵點
    all_points = []
    for point_info in reliable_end_points:
        coord = point_info['point']['coordinate']
        all_points.append((coord[1], coord[0]))  # (x, y)
    
    for point_info in reliable_turning_points:
        coord = point_info['point']['coordinate']
        all_points.append((coord[1], coord[0]))  # (x, y)
    
    # 計算所有點對之間的相對位置
    K = len(all_points)
    for i in range(K):
        for j in range(K):
            if i != j:
                x_diff = all_points[i][0] - all_points[j][0]  # xm - xn
                y_diff = all_points[i][1] - all_points[j][1]  # ym - yn
                relative_features.extend([x_diff, y_diff])
    
    return np.array(relative_features)

def extract_combined_features(image_path, reliable_end_points, reliable_turning_points):
    """
    提取組合特徵：座標特徵 + 相對位置特徵 + week3 的所有特徵
    """
    try:
        # 提取座標特徵 (2K)
        coord_features = extract_coordinate_features(reliable_end_points, reliable_turning_points)
        
        # 提取相對位置特徵 (K(K-1))
        relative_features = extract_relative_position_features(reliable_end_points, reliable_turning_points)
        
        # 提取 week3 的所有特徵
        week3_features = extract_features_all(image_path, verbose=False)
        
        # 組合所有特徵
        combined_features = np.concatenate([coord_features, relative_features, week3_features])
        
        return combined_features
        
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        # 返回零向量作為默認值
        K = len(reliable_end_points) + len(reliable_turning_points)
        coord_dim = 2 * K
        relative_dim = 2 * K * (K - 1)
        week3_dim = 22  # week3 的特徵維度
        return np.zeros(coord_dim + relative_dim + week3_dim)

def process_digit_svm(digit, direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
    """
    處理單個數字的 SVM 訓練和測試
    """
    print(f"\n===== Processing digit {digit} =====")
    
    # 讀取標準字跡檔案
    standard_files = read_standard_files()
    if digit not in standard_files:
        print(f"Warning: Cannot find standard file for digit {digit}")
        return None
    
    standard_file = standard_files[digit]
    standard_path = f'handwrite/{digit}/database/{standard_file}'
    database_path = f'handwrite/{digit}/database'
    testcase_path = f'handwrite/{digit}/testcase'
    
    # 檢查路徑是否存在
    if not all(os.path.exists(path) for path in [standard_path, database_path, testcase_path]):
        print(f"Warning: Required paths do not exist for digit {digit}")
        return None
    
    print(f"Standard file: {standard_file}")
    
    # 收集 database 中的其他圖像（排除標準字跡）
    database_images = []
    for filename in sorted(os.listdir(database_path)):
        if filename.endswith('.bmp') and filename != standard_file:
            database_images.append(os.path.join(database_path, filename))
    
    # 收集 testcase 中的圖像
    testcase_images = []
    for filename in sorted(os.listdir(testcase_path)):
        if filename.endswith('.bmp'):
            testcase_images.append(os.path.join(testcase_path, filename))
    
    if len(database_images) < 5:
        print(f"Warning: Insufficient database images for digit {digit} (need at least 5)")
        return None
    
    print(f"Available images: {len(database_images)} database, {len(testcase_images)} testcase")
    
    # 步驟1：找出可靠的特徵點
    reliable_end_points, reliable_turning_points = find_reliable_feature_points(
        standard_path, database_images, 
        direction_weight=direction_weight,
        normalized_coordinates_weight=normalized_coordinates_weight,
        stroke_ratio_weight=stroke_ratio_weight
    )
    
    if not reliable_end_points and not reliable_turning_points:
        print(f"Warning: No reliable feature points found for digit {digit}")
        return None
    
    K = len(reliable_end_points) + len(reliable_turning_points)
    coord_dim = 2 * K
    relative_dim = 2 * K * (K - 1)
    week3_dim = 22
    total_dim = coord_dim + relative_dim + week3_dim
    
    print(f"Feature dimensions: {K} reliable points -> {coord_dim} coord + {relative_dim} relative + {week3_dim} week3 = {total_dim} total")
    
    # 步驟2：提取訓練資料特徵
    print("Extracting training features...")
    train_features = []
    train_labels = []
    
    # 訓練資料：前25個 database（正確字跡，標籤=1）+ 前25個 testcase（錯誤字跡，標籤=0）
    train_database = database_images[:25]
    train_testcase = testcase_images[:25]
    
    # 處理 database 訓練資料（正確字跡）
    for img_path in train_database:
        features = extract_combined_features(img_path, reliable_end_points, reliable_turning_points)
        train_features.append(features)
        train_labels.append(1)  # 正確字跡
    
    # 處理 testcase 訓練資料（錯誤字跡）
    for img_path in train_testcase:
        features = extract_combined_features(img_path, reliable_end_points, reliable_turning_points)
        train_features.append(features)
        train_labels.append(0)  # 錯誤字跡
    
    # 步驟3：提取測試資料特徵
    print("Extracting test features...")
    test_features = []
    test_labels = []
    
    # 測試資料：剩下的25個 database + 剩下的25個 testcase
    test_database = database_images[25:50] if len(database_images) >= 50 else database_images[25:]
    test_testcase = testcase_images[25:50] if len(testcase_images) >= 50 else testcase_images[25:]
    
    # 處理 database 測試資料（正確字跡）
    for img_path in test_database:
        features = extract_combined_features(img_path, reliable_end_points, reliable_turning_points)
        test_features.append(features)
        test_labels.append(1)  # 正確字跡
    
    # 處理 testcase 測試資料（錯誤字跡）
    for img_path in test_testcase:
        features = extract_combined_features(img_path, reliable_end_points, reliable_turning_points)
        test_features.append(features)
        test_labels.append(0)  # 錯誤字跡
    
    # 轉換為 numpy 陣列
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    print(f"Training data: {train_features.shape[0]} samples, {train_features.shape[1]} features")
    print(f"Test data: {test_features.shape[0]} samples, {test_features.shape[1]} features")
    
    # 檢查是否有足夠的訓練和測試資料
    if len(train_features) == 0 or len(test_features) == 0:
        print(f"Warning: Insufficient data for digit {digit}")
        return None
    
    # 特徵標準化
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std == 0, 1, std)  # 避免除以零
    
    train_features_normalized = (train_features - mean) / std
    test_features_normalized = (test_features - mean) / std
    
    # 步驟4：訓練 SVM
    print("Training SVM...")
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(train_features_normalized, train_labels)
    
    # 步驟5：測試和評估
    train_predictions = svm_model.predict(train_features_normalized)
    test_predictions = svm_model.predict(test_features_normalized)
    
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    
    print(f"Results for digit {digit}:")
    print(f"  Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"  Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"  Feature dimensions: {train_features.shape[1]}")
    print(f"  Reliable feature points: {K}")
    
    return {
        'digit': digit,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_dim': train_features.shape[1],
        'reliable_points': K,
        'train_samples': len(train_features),
        'test_samples': len(test_features)
    }

def main():
    """主函數"""
    print("Starting Week8 Enhanced Feature Extraction and SVM Training...")
    print("Combining reliable feature points, coordinate features, relative position features, and week3 features")
    
    # 權重設定
    DIRECTION_WEIGHT = 8.0
    NORMALIZED_COORDINATES_WEIGHT = 1.0/1000
    STROKE_RATIO_WEIGHT = 1.0/10
    
    print(f"\nUsing weights:")
    print(f"  Direction weight: {DIRECTION_WEIGHT}")
    print(f"  Normalized coordinates weight: {NORMALIZED_COORDINATES_WEIGHT}")
    print(f"  Stroke ratio weight: {STROKE_RATIO_WEIGHT}")
    
    # 檢查必要資料夾
    if not os.path.exists('handwrite'):
        print("Error: Cannot find handwrite folder")
        return
    
    if not os.path.exists('handwrite/choice.txt'):
        print("Error: Cannot find choice.txt file")
        return
    
    # 處理所有數字
    results = []
    for digit in range(1, 2):
        result = process_digit_svm(
            digit, 
            direction_weight=DIRECTION_WEIGHT,
            normalized_coordinates_weight=NORMALIZED_COORDINATES_WEIGHT,
            stroke_ratio_weight=STROKE_RATIO_WEIGHT
        )
        if result:
            results.append(result)
    
    # 總結結果
    if results:
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        
        print(f"{'Digit':<6} {'Train Acc':<10} {'Test Acc':<10} {'Features':<10} {'Points':<8} {'Train/Test':<12}")
        print("-" * 60)
        
        total_train_acc = 0
        total_test_acc = 0
        
        for result in results:
            print(f"{result['digit']:<6} {result['train_accuracy']*100:<10.2f} {result['test_accuracy']*100:<10.2f} "
                  f"{result['feature_dim']:<10} {result['reliable_points']:<8} "
                  f"{result['train_samples']}/{result['test_samples']:<12}")
            total_train_acc += result['train_accuracy']
            total_test_acc += result['test_accuracy']
        
        print("-" * 60)
        avg_train_acc = total_train_acc / len(results)
        avg_test_acc = total_test_acc / len(results)
        print(f"{'AVG':<6} {avg_train_acc*100:<10.2f} {avg_test_acc*100:<10.2f}")
        
        print(f"\nOverall Performance:")
        print(f"  Average Training Accuracy: {avg_train_acc * 100:.2f}%")
        print(f"  Average Test Accuracy: {avg_test_acc * 100:.2f}%")
        print(f"  Successfully processed: {len(results)}/9 digits")
        
    else:
        print("No successful results obtained.")

if __name__ == "__main__":
    main()