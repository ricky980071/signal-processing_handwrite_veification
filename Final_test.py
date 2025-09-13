# -*- coding: utf-8 -*-
# Final_test.py
# 這個檔案主要功能：
# 1. 提供可選擇的特徵組合（座標特徵、相對位置特徵、week3特徵、week7特徵向量）
# 2. 支援 SVM 和 Neural Network 兩種機器學習模型
# 3. 對每個數字獨立訓練模型
# 4. 比較不同特徵組合和模型的效果

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from week7 import (
    read_standard_files, extract_characteristic_points, distance,
    phi_matrix, coordinate_normalization_m_custom, coordinate_normalization_n_custom,
    stroke_ratio_m_custom, stroke_ratio_n_custom
)
from week3 import extract_features_all

class FeatureExtractor:
    """特徵提取器類別，負責管理不同類型的特徵提取"""
    
    def __init__(self, direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
        self.direction_weight = direction_weight
        self.normalized_coordinates_weight = normalized_coordinates_weight
        self.stroke_ratio_weight = stroke_ratio_weight
        
    def find_reliable_feature_points(self, standard_image_path, database_images, L=3, d=12):
        """找出可靠的特徵點（從 week8 複製）"""
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
        reliable_end_points = self._analyze_point_reliability(
            standard_end_points, analysis_images, standard_image_path, 'end_point',
            L, d
        )
        
        # 分析 turning points 的可靠性
        reliable_turning_points = self._analyze_point_reliability(
            standard_turning_points, analysis_images, standard_image_path, 'turning_point',
            L, d
        )
        
        print(f"  Reliable points: {len(reliable_end_points)} end points, {len(reliable_turning_points)} turning points")
        
        return reliable_end_points, reliable_turning_points
    
    def _analyze_point_reliability(self, standard_points, analysis_images, standard_image_path, point_type, L, d):
        """分析特徵點的可靠性"""
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
                            std_point, test_point, self.direction_weight, 
                            self.normalized_coordinates_weight, self.stroke_ratio_weight
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
        
        return reliable_points
    
    def extract_coordinate_features(self, reliable_end_points, reliable_turning_points):
        """提取座標特徵 (2K 個特徵)"""
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
    
    def extract_relative_position_features(self, reliable_end_points, reliable_turning_points):
        """提取相對位置特徵 K(K-1) 個特徵"""
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
    
    def extract_week7_features(self, image_path, reliable_end_points, reliable_turning_points):
        """提取 week7 的 M 維特徵向量"""
        try:
            # 這裡需要實現從可靠特徵點提取 week7 特徵向量的邏輯
            # 暫時返回空陣列，可以根據需要擴展
            M = len(reliable_end_points) + len(reliable_turning_points)
            return np.zeros(M)  # 預設返回零向量
        except Exception as e:
            print(f"Error extracting week7 features from {image_path}: {e}")
            return np.array([])

class ModelTrainer:
    """模型訓練器類別，負責訓練 SVM 和 Neural Network"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def train_svm(self, X_train, y_train, digit, kernel='linear', C=1.0, random_state=42):
        """訓練 SVM 模型"""
        model = SVC(kernel=kernel, C=C, random_state=random_state)
        model.fit(X_train, y_train)
        self.models[f'svm_{digit}'] = model
        return model
    
    def train_neural_network(self, X_train, y_train, digit, hidden_layer_sizes=(100, 50), 
                           max_iter=1000, random_state=42, learning_rate_init=0.001):
        """訓練 Neural Network 模型"""
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        self.models[f'nn_{digit}'] = model
        return model
    
    def predict(self, model, X_test):
        """進行預測"""
        return model.predict(X_test)
    
    def evaluate(self, model, X_test, y_test):
        """評估模型"""
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, predictions

class FeatureSelector:
    """特徵選擇器，讓使用者選擇要使用的特徵"""
    
    @staticmethod
    def get_feature_options():
        """取得所有可用的特徵選項"""
        return {
            '1': ('coordinate', '座標特徵 (2K)'),
            '2': ('relative_position', '相對位置特徵 (K(K-1))'),
            '3': ('week3_basic', 'Week3 基本特徵 (10維)'),
            '4': ('week3_moments', 'Week3 動差特徵 (7維)'),
            '5': ('week3_intensity', 'Week3 筆劃強度特徵 (2維)'),
            '6': ('week3_erosion', 'Week3 侵蝕特徵 (3維)'),
            '7': ('week3_all', 'Week3 所有特徵 (22維)'),
            '8': ('week7_vector', 'Week7 特徵向量 (M維)'),
            '9': ('all_features', '所有特徵組合')
        }
    
    @staticmethod
    def display_feature_options():
        """顯示特徵選項"""
        options = FeatureSelector.get_feature_options()
        print("\n=== 可用特徵選項 ===")
        for key, (feature_key, description) in options.items():
            print(f"{key}: {description}")
        print("===================")
    
    @staticmethod
    def select_features():
        """互動式選擇特徵"""
        FeatureSelector.display_feature_options()
        
        print("\n請選擇要使用的特徵 (輸入數字，用逗號分隔):")
        print("例如: 1,7 (座標特徵+Week3所有特徵)")
        print("或輸入: 9 (使用所有特徵)")
        
        user_input = input("您的選擇: ").strip()
        
        # 解析使用者輸入
        selected_numbers = [num.strip() for num in user_input.split(',')]
        options = FeatureSelector.get_feature_options()
        
        # 驗證選擇並轉換為特徵鍵
        selected_features = []
        for num in selected_numbers:
            if num in options:
                feature_key, description = options[num]
                if feature_key == 'all_features':
                    # 如果選擇了所有特徵，返回除了'all_features'之外的所有特徵
                    all_feature_keys = []
                    for key, (fkey, _) in options.items():
                        if fkey != 'all_features':
                            all_feature_keys.append(fkey)
                    return all_feature_keys
                else:
                    selected_features.append(feature_key)
                    print(f"已選擇: {description}")
            else:
                print(f"警告: '{num}' 不是有效的選項")
        
        if not selected_features:
            print("未選擇任何有效特徵，使用所有特徵")
            all_feature_keys = []
            for key, (fkey, _) in options.items():
                if fkey != 'all_features':
                    all_feature_keys.append(fkey)
            return all_feature_keys
        
        return selected_features

def extract_combined_features(image_path, reliable_end_points, reliable_turning_points, 
                            feature_list, feature_extractor):
    """根據選擇的特徵列表提取組合特徵"""
    combined_features = []
    
    for feature_type in feature_list:
        try:
            if feature_type == 'coordinate':
                features = feature_extractor.extract_coordinate_features(
                    reliable_end_points, reliable_turning_points
                )
            elif feature_type == 'relative_position':
                features = feature_extractor.extract_relative_position_features(
                    reliable_end_points, reliable_turning_points
                )
            elif feature_type == 'week3_basic':
                from week3 import extract_features_basic
                features = extract_features_basic(image_path)
            elif feature_type == 'week3_moments':
                from week3 import extract_features_with_moments
                features, _, _ = extract_features_with_moments(image_path)
                features = features[10:17]  # 只取動差部分
            elif feature_type == 'week3_intensity':
                from week3 import extract_features_with_intensity
                features = extract_features_with_intensity(image_path)
                features = features[17:19]  # 只取強度部分
            elif feature_type == 'week3_erosion':
                from week3 import extract_features_with_erosion
                features = extract_features_with_erosion(image_path)
                features = features[17:20]  # 只取侵蝕部分
            elif feature_type == 'week3_all':
                features = extract_features_all(image_path, verbose=False)
            elif feature_type == 'week7_vector':
                features = feature_extractor.extract_week7_features(
                    image_path, reliable_end_points, reliable_turning_points
                )
            else:
                print(f"警告: 未知的特徵類型 '{feature_type}'")
                continue
            
            if len(features) > 0:
                combined_features.extend(features)
                
        except Exception as e:
            print(f"提取特徵 '{feature_type}' 時發生錯誤: {e}")
            continue
    
    return np.array(combined_features)

def process_digit_with_features(digit, selected_features, model_types, 
                               direction_weight=8.0, normalized_coordinates_weight=1.0/1000, 
                               stroke_ratio_weight=1.0/10):
    """處理單個數字的特徵提取和模型訓練"""
    print(f"\n===== Processing digit {digit} =====")
    
    # 初始化特徵提取器和模型訓練器
    feature_extractor = FeatureExtractor(direction_weight, normalized_coordinates_weight, stroke_ratio_weight)
    model_trainer = ModelTrainer()
    
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
    print(f"Selected features: {selected_features}")
    print(f"Selected models: {model_types}")
    
    # 收集圖像路徑
    database_images = []
    for filename in sorted(os.listdir(database_path)):
        if filename.endswith('.bmp') and filename != standard_file:
            database_images.append(os.path.join(database_path, filename))
    
    testcase_images = []
    for filename in sorted(os.listdir(testcase_path)):
        if filename.endswith('.bmp'):
            testcase_images.append(os.path.join(testcase_path, filename))
    
    if len(database_images) < 5:
        print(f"Warning: Insufficient database images for digit {digit} (need at least 5)")
        return None
    
    print(f"Available images: {len(database_images)} database, {len(testcase_images)} testcase")
    
    # 步驟1：找出可靠的特徵點
    reliable_end_points, reliable_turning_points = feature_extractor.find_reliable_feature_points(
        standard_path, database_images
    )
    
    if not reliable_end_points and not reliable_turning_points:
        print(f"Warning: No reliable feature points found for digit {digit}")
        return None
    
    K = len(reliable_end_points) + len(reliable_turning_points)
    print(f"Found {K} reliable feature points")
    
    # 步驟2：提取訓練資料特徵
    print("Extracting training features...")
    train_features = []
    train_labels = []
    
    # 訓練資料：前25個 database（正確字跡，標籤=1）+ 前25個 testcase（錯誤字跡，標籤=0）
    train_database = database_images[:25]
    train_testcase = testcase_images[:25]
    
    # 處理 database 訓練資料（正確字跡）
    for img_path in train_database:
        features = extract_combined_features(
            img_path, reliable_end_points, reliable_turning_points, 
            selected_features, feature_extractor
        )
        if len(features) > 0:
            train_features.append(features)
            train_labels.append(1)  # 正確字跡
    
    # 處理 testcase 訓練資料（錯誤字跡）
    for img_path in train_testcase:
        features = extract_combined_features(
            img_path, reliable_end_points, reliable_turning_points, 
            selected_features, feature_extractor
        )
        if len(features) > 0:
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
        features = extract_combined_features(
            img_path, reliable_end_points, reliable_turning_points, 
            selected_features, feature_extractor
        )
        if len(features) > 0:
            test_features.append(features)
            test_labels.append(1)  # 正確字跡
    
    # 處理 testcase 測試資料（錯誤字跡）
    for img_path in test_testcase:
        features = extract_combined_features(
            img_path, reliable_end_points, reliable_turning_points, 
            selected_features, feature_extractor
        )
        if len(features) > 0:
            test_features.append(features)
            test_labels.append(0)  # 錯誤字跡
    
    # 轉換為 numpy 陣列
    if not train_features or not test_features:
        print(f"Warning: Insufficient feature data for digit {digit}")
        return None
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    print(f"Training data: {train_features.shape[0]} samples, {train_features.shape[1]} features")
    print(f"Test data: {test_features.shape[0]} samples, {test_features.shape[1]} features")
    
    # 特徵標準化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 步驟4：訓練和測試模型
    results = {}
    
    if 'svm' in model_types:
        print("Training SVM...")
        svm_model = model_trainer.train_svm(train_features_scaled, train_labels, digit)
        svm_train_acc, svm_train_pred = model_trainer.evaluate(svm_model, train_features_scaled, train_labels)
        svm_test_acc, svm_test_pred = model_trainer.evaluate(svm_model, test_features_scaled, test_labels)
        
        results['svm'] = {
            'train_accuracy': svm_train_acc,
            'test_accuracy': svm_test_acc,
            'train_predictions': svm_train_pred,
            'test_predictions': svm_test_pred
        }
        
        print(f"SVM Results - Train: {svm_train_acc*100:.2f}%, Test: {svm_test_acc*100:.2f}%")
    
    if 'neural_network' in model_types:
        print("Training Neural Network...")
        nn_model = model_trainer.train_neural_network(train_features_scaled, train_labels, digit)
        nn_train_acc, nn_train_pred = model_trainer.evaluate(nn_model, train_features_scaled, train_labels)
        nn_test_acc, nn_test_pred = model_trainer.evaluate(nn_model, test_features_scaled, test_labels)
        
        results['neural_network'] = {
            'train_accuracy': nn_train_acc,
            'test_accuracy': nn_test_acc,
            'train_predictions': nn_train_pred,
            'test_predictions': nn_test_pred
        }
        
        print(f"Neural Network Results - Train: {nn_train_acc*100:.2f}%, Test: {nn_test_acc*100:.2f}%")
    
    return {
        'digit': digit,
        'feature_count': train_features.shape[1],
        'reliable_points': K,
        'train_samples': len(train_features),
        'test_samples': len(test_features),
        'selected_features': selected_features,
        'results': results,
        'train_labels': train_labels,
        'test_labels': test_labels
    }

def visualize_results(all_results, selected_features, model_types):
    """可視化結果"""
    if not all_results:
        print("No results to visualize")
        return
    
    digits = [result['digit'] for result in all_results]
    
    # 準備數據
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Model Performance Comparison\nFeatures: {", ".join(selected_features)}', fontsize=14)
    
    # 1. 訓練準確率比較
    ax1 = axes[0, 0]
    if 'svm' in model_types:
        svm_train_accs = [result['results']['svm']['train_accuracy']*100 for result in all_results]
        ax1.bar([d-0.2 for d in digits], svm_train_accs, width=0.4, label='SVM', alpha=0.7)
    
    if 'neural_network' in model_types:
        nn_train_accs = [result['results']['neural_network']['train_accuracy']*100 for result in all_results]
        ax1.bar([d+0.2 for d in digits], nn_train_accs, width=0.4, label='Neural Network', alpha=0.7)
    
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.set_title('Training Accuracy Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 測試準確率比較
    ax2 = axes[0, 1]
    if 'svm' in model_types:
        svm_test_accs = [result['results']['svm']['test_accuracy']*100 for result in all_results]
        ax2.bar([d-0.2 for d in digits], svm_test_accs, width=0.4, label='SVM', alpha=0.7)
    
    if 'neural_network' in model_types:
        nn_test_accs = [result['results']['neural_network']['test_accuracy']*100 for result in all_results]
        ax2.bar([d+0.2 for d in digits], nn_test_accs, width=0.4, label='Neural Network', alpha=0.7)
    
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 特徵維度統計
    ax3 = axes[1, 0]
    feature_counts = [result['feature_count'] for result in all_results]
    reliable_points = [result['reliable_points'] for result in all_results]
    
    ax3.bar([d-0.2 for d in digits], feature_counts, width=0.4, label='Total Features', alpha=0.7)
    ax3.bar([d+0.2 for d in digits], reliable_points, width=0.4, label='Reliable Points', alpha=0.7)
    
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Dimensions and Reliable Points')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 樣本數量統計
    ax4 = axes[1, 1]
    train_samples = [result['train_samples'] for result in all_results]
    test_samples = [result['test_samples'] for result in all_results]
    
    ax4.bar([d-0.2 for d in digits], train_samples, width=0.4, label='Training Samples', alpha=0.7)
    ax4.bar([d+0.2 for d in digits], test_samples, width=0.4, label='Test Samples', alpha=0.7)
    
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Sample Count')
    ax4.set_title('Training and Test Sample Counts')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函數"""
    print("=== Final Test: Feature Selection and Model Comparison ===")
    print("This program allows you to:")
    print("1. Select specific feature combinations")
    print("2. Choose between SVM and Neural Network models")
    print("3. Compare performance across different digits")
    
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
    
    # 選擇特徵
    selected_features = FeatureSelector.select_features()
    if not selected_features:
        print("No features selected. Exiting.")
        return
    
    print(f"\n選擇的特徵: {selected_features}")
    
    # 選擇模型
    print("\n=== 模型選擇 ===")
    print("可用模型:")
    print("1. SVM - Support Vector Machine")
    print("2. Neural Network - Multi-layer Perceptron")
    print("3. 兩個模型都使用")
    
    model_input = input("請選擇模型 (輸入數字，例如: 1 或 1,2 或 3): ").strip()
    
    # 解析模型選擇
    if model_input == '3':
        model_types = ['svm', 'neural_network']
    else:
        model_numbers = [num.strip() for num in model_input.split(',')]
        model_types = []
        for num in model_numbers:
            if num == '1':
                model_types.append('svm')
                print("已選擇: SVM")
            elif num == '2':
                model_types.append('neural_network')
                print("已選擇: Neural Network")
            else:
                print(f"警告: '{num}' 不是有效的模型選項")
    
    if not model_types:
        print("未選擇任何有效模型，使用兩個模型")
        model_types = ['svm', 'neural_network']
    
    print(f"選擇的模型: {model_types}")
    
    # 處理所有數字
    all_results = []
    for digit in range(1, 10):
        result = process_digit_with_features(
            digit, selected_features, model_types,
            direction_weight=DIRECTION_WEIGHT,
            normalized_coordinates_weight=NORMALIZED_COORDINATES_WEIGHT,
            stroke_ratio_weight=STROKE_RATIO_WEIGHT
        )
        if result:
            all_results.append(result)
    
    # 總結結果
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        
        # 創建結果表格
        header = f"{'Digit':<6} {'Features':<10} {'Points':<8} {'Train/Test':<12}"
        
        for model_type in model_types:
            header += f" {model_type.upper()}_Train {model_type.upper()}_Test"
        
        print(header)
        print("-" * len(header))
        
        # 計算平均準確率
        avg_results = {model_type: {'train': 0, 'test': 0} for model_type in model_types}
        
        for result in all_results:
            line = f"{result['digit']:<6} {result['feature_count']:<10} {result['reliable_points']:<8} {result['train_samples']}/{result['test_samples']:<11}"
            
            for model_type in model_types:
                if model_type in result['results']:
                    train_acc = result['results'][model_type]['train_accuracy'] * 100
                    test_acc = result['results'][model_type]['test_accuracy'] * 100
                    line += f" {train_acc:>11.2f} {test_acc:>9.2f}"
                    
                    avg_results[model_type]['train'] += train_acc
                    avg_results[model_type]['test'] += test_acc
                else:
                    line += f" {'N/A':>11} {'N/A':>9}"
            
            print(line)
        
        # 計算並顯示平均值
        print("-" * len(header))
        avg_line = f"{'AVG':<6} {'-':<10} {'-':<8} {'-':<12}"
        
        for model_type in model_types:
            avg_train = avg_results[model_type]['train'] / len(all_results)
            avg_test = avg_results[model_type]['test'] / len(all_results)
            avg_line += f" {avg_train:>11.2f} {avg_test:>9.2f}"
        
        print(avg_line)
        
        print(f"\nOverall Performance:")
        for model_type in model_types:
            avg_train = avg_results[model_type]['train'] / len(all_results)
            avg_test = avg_results[model_type]['test'] / len(all_results)
            print(f"  {model_type.upper()}: Train={avg_train:.2f}%, Test={avg_test:.2f}%")
        
        print(f"  Successfully processed: {len(all_results)}/9 digits")
        print(f"  Selected features: {', '.join(selected_features)}")
        
        # 可視化結果
        visualize_results(all_results, selected_features, model_types)
        
        # 詳細分析報告
        print("\n" + "="*60)
        print("DETAILED ANALYSIS")
        print("="*60)
        
        for result in all_results:
            print(f"\nDigit {result['digit']}:")
            print(f"  Feature dimensions: {result['feature_count']}")
            print(f"  Reliable feature points: {result['reliable_points']}")
            
            for model_type in model_types:
                if model_type in result['results']:
                    model_result = result['results'][model_type]
                    print(f"  {model_type.upper()}:")
                    print(f"    Training accuracy: {model_result['train_accuracy']*100:.2f}%")
                    print(f"    Test accuracy: {model_result['test_accuracy']*100:.2f}%")
    
    else:
        print("No successful results obtained.")

if __name__ == "__main__":
    main()