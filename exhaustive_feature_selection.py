# -*- coding: utf-8 -*-
# exhaustive_feature_selection.py
# 這個檔案主要功能：
# 1. 固定選擇基本特徵，窮舉剩餘 6 種特徵的所有組合
# 2. 測試 2^6 = 64 種不同的特徵組合
# 3. 對每個數字獨立進行特徵選擇和 SVM 訓練
# 4. 找出最佳的特徵組合並分析效果

import os
import numpy as np
import time
import itertools
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from week7 import (
    read_standard_files, extract_characteristic_points, distance,
    phi_matrix, coordinate_normalization_m_custom, coordinate_normalization_n_custom,
    stroke_ratio_m_custom, stroke_ratio_n_custom
)
from week3 import (
    extract_features_basic, extract_features_with_moments, 
    calculate_intensity_features, calculate_erosion_features
)

class FeatureExtractor:
    """特徵提取器，支援選擇性提取不同類型的特徵"""
    
    def __init__(self, direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
        self.direction_weight = direction_weight
        self.normalized_coordinates_weight = normalized_coordinates_weight
        self.stroke_ratio_weight = stroke_ratio_weight
        self.reliable_points_cache = {}  # 快取可靠特徵點
        
    def find_reliable_feature_points(self, standard_image_path, database_images, L=3, d=12):
        """尋找可靠的特徵點"""
        cache_key = (standard_image_path, tuple(database_images[:5]))
        if cache_key in self.reliable_points_cache:
            return self.reliable_points_cache[cache_key]
            
        # 提取標準圖像的特徵點
        standard_end_points, standard_turning_points = extract_characteristic_points(standard_image_path, L, d)
        
        if not standard_end_points:
            return [], []
        
        # 只使用前5個 database 圖像進行分析
        analysis_images = database_images[:5]
        
        # 分析 end points 的可靠性
        reliable_end_points = self._analyze_point_reliability(
            standard_end_points, analysis_images, 'end_point', L, d
        )
        
        # 分析 turning points 的可靠性
        reliable_turning_points = self._analyze_point_reliability(
            standard_turning_points, analysis_images, 'turning_point', L, d
        )
        
        result = (reliable_end_points, reliable_turning_points)
        self.reliable_points_cache[cache_key] = result
        return result
    
    def _analyze_point_reliability(self, standard_points, analysis_images, point_type, L, d):
        """分析特徵點的可靠性"""
        reliable_points = []
        
        for i, std_point in enumerate(standard_points):
            distances = []
            
            # 對每個分析圖像找到最佳匹配點
            for img_path in analysis_images:
                try:
                    test_end_points, test_turning_points = extract_characteristic_points(img_path, L, d)
                    
                    if point_type == 'end_point':
                        test_points = test_end_points
                    else:
                        test_points = test_turning_points
                    
                    if not test_points:
                        continue
                    
                    # 找到最佳匹配點
                    min_distance = float('inf')
                    for test_point in test_points:
                        dist, _, _, _ = distance(
                            std_point, test_point, self.direction_weight, 
                            self.normalized_coordinates_weight, self.stroke_ratio_weight
                        )
                        if dist < min_distance:
                            min_distance = dist
                    
                    if min_distance != float('inf'):
                        distances.append(min_distance)
                        
                except Exception:
                    continue
            
            # 檢查這個標準點是否可靠
            if len(distances) >= 3:  # 至少要有3個成功匹配
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
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
    
    def extract_features_by_groups(self, image_path, feature_groups, reliable_end_points=None, reliable_turning_points=None):
        """
        根據選擇的特徵組提取特徵
        
        feature_groups: list of strings, 可選值：
        - 'basic': 基本特徵 (10個) - 總是包含
        - 'second_moments': 二階動差特徵 (3個)
        - 'third_moments': 三階動差特徵 (4個)
        - 'intensity': 筆劃強度特徵 (2個)
        - 'erosion': 侵蝕特徵 (3個)
        - 'coordinate': 座標特徵 (2K個)
        - 'relative': 相對位置特徵 (2K(K-1)個)
        """
        features = []
        
        # 總是包含基本特徵
        if not any(group in feature_groups for group in ['basic', 'second_moments', 'third_moments', 'intensity', 'erosion']):
            # 如果沒有 week3 相關特徵，只提取基本特徵
            basic_features = extract_features_basic(image_path, verbose=False)
            features.extend(basic_features)
        else:
            # 提取基本特徵和動差相關特徵
            from week3 import extract_features_with_moments
            moments_features, Y, stroke_mask = extract_features_with_moments(image_path, verbose=False)
            
            # 總是包含基本特徵
            features.extend(moments_features[:10])  # 前10個是基本特徵
            
            if 'second_moments' in feature_groups:
                features.extend(moments_features[10:13])  # 二階動差
            
            if 'third_moments' in feature_groups:
                features.extend(moments_features[13:17])  # 三階動差
            
            if 'intensity' in feature_groups:
                intensity_features = calculate_intensity_features(Y, stroke_mask)
                features.extend(intensity_features)
            
            if 'erosion' in feature_groups:
                erosion_features = calculate_erosion_features(stroke_mask)
                features.extend(erosion_features)
        
        # 提取座標和相對位置特徵
        if 'coordinate' in feature_groups or 'relative' in feature_groups:
            if reliable_end_points is None or reliable_turning_points is None:
                # 如果沒有提供可靠特徵點，跳過這些特徵
                pass
            else:
                if 'coordinate' in feature_groups:
                    coord_features = self._extract_coordinate_features(reliable_end_points, reliable_turning_points)
                    features.extend(coord_features)
                
                if 'relative' in feature_groups:
                    relative_features = self._extract_relative_position_features(reliable_end_points, reliable_turning_points)
                    features.extend(relative_features)
        
        return np.array(features)
    
    def _extract_coordinate_features(self, reliable_end_points, reliable_turning_points):
        """提取座標特徵"""
        coordinate_features = []
        
        # 提取 end points 的座標
        for point_info in reliable_end_points:
            coord = point_info['point']['coordinate']
            coordinate_features.extend([coord[1], coord[0]])  # (x, y)
        
        # 提取 turning points 的座標
        for point_info in reliable_turning_points:
            coord = point_info['point']['coordinate']
            coordinate_features.extend([coord[1], coord[0]])  # (x, y)
        
        return coordinate_features
    
    def _extract_relative_position_features(self, reliable_end_points, reliable_turning_points):
        """提取相對位置特徵"""
        # 收集所有可靠的特徵點
        all_points = []
        for point_info in reliable_end_points:
            coord = point_info['point']['coordinate']
            all_points.append((coord[1], coord[0]))  # (x, y)
        
        for point_info in reliable_turning_points:
            coord = point_info['point']['coordinate']
            all_points.append((coord[1], coord[0]))  # (x, y)
        
        # 計算所有點對之間的相對位置
        relative_features = []
        K = len(all_points)
        for i in range(K):
            for j in range(K):
                if i != j:
                    x_diff = all_points[i][0] - all_points[j][0]
                    y_diff = all_points[i][1] - all_points[j][1]
                    relative_features.extend([x_diff, y_diff])
        
        return relative_features


class ExhaustiveFeatureSelector:
    """窮舉法特徵選擇器，固定基本特徵，測試其他特徵組合"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        # 固定基本特徵，其他6個特徵可選
        self.optional_features = [
            'second_moments',  # 二階動差特徵 (3個)
            'third_moments',   # 三階動差特徵 (4個)
            'intensity',       # 筆劃強度特徵 (2個)
            'erosion',         # 侵蝕特徵 (3個)
            'coordinate',      # 座標特徵 (2K個)
            'relative'         # 相對位置特徵 (2K(K-1)個)
        ]
        self.feature_names = {
            'basic': '基本特徵',
            'second_moments': '二階動差特徵',
            'third_moments': '三階動差特徵',
            'intensity': '筆劃強度特徵',
            'erosion': '侵蝕特徵',
            'coordinate': '座標特徵',
            'relative': '相對位置特徵'
        }
    
    def generate_all_combinations(self):
        """生成所有特徵組合（2^6 = 64種）"""
        combinations = []
        
        # 生成所有可能的子集
        for r in range(len(self.optional_features) + 1):  # 0 到 6 個特徵
            for combo in itertools.combinations(self.optional_features, r):
                # 總是包含基本特徵
                feature_set = ['basic'] + list(combo)
                combinations.append(feature_set)
        
        return combinations
    
    def evaluate_feature_combination(self, feature_groups, digit):
        """評估特定特徵組合的效果"""
        try:
            # 設定路徑
            standard_files = read_standard_files()
            if digit not in standard_files:
                return 0.0, {}
            
            standard_file = standard_files[digit]
            standard_path = f'handwrite/{digit}/database/{standard_file}'
            database_path = f'handwrite/{digit}/database'
            testcase_path = f'handwrite/{digit}/testcase'
            
            # 檢查路徑
            if not all(os.path.exists(path) for path in [standard_path, database_path, testcase_path]):
                return 0.0, {}
            
            # 收集圖像
            database_images = []
            for filename in sorted(os.listdir(database_path)):
                if filename.endswith('.bmp') and filename != standard_file:
                    database_images.append(os.path.join(database_path, filename))
            
            testcase_images = []
            for filename in sorted(os.listdir(testcase_path)):
                if filename.endswith('.bmp'):
                    testcase_images.append(os.path.join(testcase_path, filename))
            
            if len(database_images) < 5 or len(testcase_images) < 25:
                return 0.0, {}
            
            # 找出可靠特徵點（如果需要）
            reliable_end_points, reliable_turning_points = None, None
            if 'coordinate' in feature_groups or 'relative' in feature_groups:
                reliable_end_points, reliable_turning_points = self.feature_extractor.find_reliable_feature_points(
                    standard_path, database_images
                )
                if not reliable_end_points and not reliable_turning_points:
                    # 如果沒有可靠特徵點，移除相關特徵組
                    feature_groups = [g for g in feature_groups if g not in ['coordinate', 'relative']]
            
            # 提取訓練特徵
            train_features = []
            train_labels = []
            
            # database 訓練資料（正確字跡）
            for img_path in database_images[:25]:
                features = self.feature_extractor.extract_features_by_groups(
                    img_path, feature_groups, reliable_end_points, reliable_turning_points
                )
                if len(features) > 0:
                    train_features.append(features)
                    train_labels.append(1)
            
            # testcase 訓練資料（錯誤字跡）
            for img_path in testcase_images[:25]:
                features = self.feature_extractor.extract_features_by_groups(
                    img_path, feature_groups, reliable_end_points, reliable_turning_points
                )
                if len(features) > 0:
                    train_features.append(features)
                    train_labels.append(0)
            
            # 提取測試特徵
            test_features = []
            test_labels = []
            
            # database 測試資料（正確字跡）
            test_database = database_images[25:50] if len(database_images) >= 50 else database_images[25:]
            for img_path in test_database:
                features = self.feature_extractor.extract_features_by_groups(
                    img_path, feature_groups, reliable_end_points, reliable_turning_points
                )
                if len(features) > 0:
                    test_features.append(features)
                    test_labels.append(1)
            
            # testcase 測試資料（錯誤字跡）
            test_testcase = testcase_images[25:50] if len(testcase_images) >= 50 else testcase_images[25:]
            for img_path in test_testcase:
                features = self.feature_extractor.extract_features_by_groups(
                    img_path, feature_groups, reliable_end_points, reliable_turning_points
                )
                if len(features) > 0:
                    test_features.append(features)
                    test_labels.append(0)
            
            if len(train_features) == 0 or len(test_features) == 0:
                return 0.0, {}
            
            # 轉換為 numpy 陣列並標準化
            train_features = np.array(train_features)
            test_features = np.array(test_features)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)
            
            # 檢查特徵維度一致性
            if train_features.shape[1] != test_features.shape[1]:
                return 0.0, {}
            
            # 標準化
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            # 訓練 SVM
            svm_model = SVC(kernel='linear', random_state=42)
            svm_model.fit(train_features_scaled, train_labels)
            
            # 評估
            train_pred = svm_model.predict(train_features_scaled)
            test_pred = svm_model.predict(test_features_scaled)
            
            train_acc = accuracy_score(train_labels, train_pred)
            test_acc = accuracy_score(test_labels, test_pred)
            
            info = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'feature_dim': train_features.shape[1],
                'train_samples': len(train_features),
                'test_samples': len(test_features),
                'reliable_points': len(reliable_end_points) + len(reliable_turning_points) if reliable_end_points else 0
            }
            
            return test_acc, info
            
        except Exception as e:
            print(f"Error evaluating feature combination {feature_groups} for digit {digit}: {e}")
            return 0.0, {}
    
    def exhaustive_search(self, digit):
        """
        對單個數字執行窮舉搜尋
        
        Returns:
        - all_results: 所有組合的結果
        - best_combination: 最佳特徵組合
        - best_score: 最佳分數
        """
        print(f"\n===== Exhaustive Search for Digit {digit} =====")
        
        all_combinations = self.generate_all_combinations()
        print(f"Testing {len(all_combinations)} feature combinations...")
        
        all_results = []
        best_score = 0.0
        best_combination = None
        best_info = {}
        
        for i, feature_combo in enumerate(all_combinations):
            print(f"  [{i+1:2d}/{len(all_combinations)}] Testing: {[self.feature_names[f] for f in feature_combo if f != 'basic']}")
            
            start_time = time.time()
            score, info = self.evaluate_feature_combination(feature_combo, digit)
            end_time = time.time()
            
            result = {
                'combination': feature_combo,
                'score': score,
                'info': info,
                'time': end_time - start_time
            }
            all_results.append(result)
            
            print(f"    Score: {score:.4f}, Dims: {info.get('feature_dim', 0)}, Time: {result['time']:.2f}s")
            
            if score > best_score:
                best_score = score
                best_combination = feature_combo
                best_info = info
        
        print(f"\nBest combination for digit {digit}: {[self.feature_names[f] for f in best_combination]}")
        print(f"Best score: {best_score:.4f}")
        print(f"Feature dimensions: {best_info.get('feature_dim', 0)}")
        
        return all_results, best_combination, best_score
    
    def run_exhaustive_search_for_all_digits(self):
        """對所有數字執行窮舉搜尋"""
        print("Starting Exhaustive Feature Selection...")
        print("Fixed feature: 基本特徵 (always included)")
        print("Optional features: 二階動差、三階動差、筆劃強度、侵蝕、座標、相對位置")
        print(f"Total combinations to test: 2^6 = 64 per digit")
        
        all_digit_results = {}
        summary_results = []
        
        for digit in range(1, 10):
            try:
                all_results, best_combo, best_score = self.exhaustive_search(digit)
                
                all_digit_results[digit] = {
                    'all_results': all_results,
                    'best_combination': best_combo,
                    'best_score': best_score
                }
                
                if best_combo:
                    # 找到最佳結果的詳細資訊
                    best_result = next(r for r in all_results if r['combination'] == best_combo)
                    best_info = best_result['info']
                    
                    summary_results.append({
                        'digit': digit,
                        'best_score': best_score,
                        'best_combination': best_combo,
                        'feature_count': len(best_combo),
                        'feature_dim': best_info.get('feature_dim', 0),
                        'train_accuracy': best_info.get('train_accuracy', 0),
                        'test_accuracy': best_info.get('test_accuracy', 0)
                    })
                
            except Exception as e:
                print(f"Error processing digit {digit}: {e}")
                continue
        
        return all_digit_results, summary_results


def print_exhaustive_results(all_digit_results, summary_results):
    """列印窮舉搜尋結果"""
    print("\n" + "="*100)
    print("EXHAUSTIVE FEATURE SELECTION RESULTS")
    print("="*100)
    
    # 總結表格
    print(f"\n{'Digit':<6} {'Test Acc':<10} {'Train Acc':<11} {'Features':<10} {'Dims':<8} {'Selected Additional Features':<40}")
    print("-" * 100)
    
    total_test_acc = 0
    total_train_acc = 0
    
    for result in summary_results:
        # 移除基本特徵，只顯示額外選擇的特徵
        additional_features = [f for f in result['best_combination'] if f != 'basic']
        feature_names = []
        for f in additional_features:
            if f == 'second_moments':
                feature_names.append('二階動差')
            elif f == 'third_moments':
                feature_names.append('三階動差')
            elif f == 'intensity':
                feature_names.append('筆劃強度')
            elif f == 'erosion':
                feature_names.append('侵蝕')
            elif f == 'coordinate':
                feature_names.append('座標')
            elif f == 'relative':
                feature_names.append('相對位置')
        
        feature_str = ', '.join(feature_names) if feature_names else '無額外特徵'
        
        print(f"{result['digit']:<6} {result['test_accuracy']*100:<10.2f} {result['train_accuracy']*100:<11.2f} "
              f"{result['feature_count']:<10} {result['feature_dim']:<8} {feature_str:<40}")
        
        total_test_acc += result['test_accuracy']
        total_train_acc += result['train_accuracy']
    
    if summary_results:
        avg_test_acc = total_test_acc / len(summary_results)
        avg_train_acc = total_train_acc / len(summary_results)
        print("-" * 100)
        print(f"{'AVG':<6} {avg_test_acc*100:<10.2f} {avg_train_acc*100:<11.2f}")
    
    # 特徵選擇統計
    print("\n" + "="*80)
    print("FEATURE SELECTION STATISTICS")
    print("="*80)
    
    feature_selection_count = {}
    for result in summary_results:
        for feature in result['best_combination']:
            if feature != 'basic':  # 排除固定的基本特徵
                if feature not in feature_selection_count:
                    feature_selection_count[feature] = 0
                feature_selection_count[feature] += 1
    
    # 按選擇頻率排序
    sorted_features = sorted(feature_selection_count.items(), key=lambda x: x[1], reverse=True)
    
    feature_display_names = {
        'second_moments': '二階動差特徵',
        'third_moments': '三階動差特徵',
        'intensity': '筆劃強度特徵',
        'erosion': '侵蝕特徵',
        'coordinate': '座標特徵',
        'relative': '相對位置特徵'
    }
    
    print(f"{'Feature':<20} {'Selected Count':<15} {'Success Rate':<15}")
    print("-" * 50)
    
    total_digits = len(summary_results)
    for feature, count in sorted_features:
        success_rate = count / total_digits * 100
        display_name = feature_display_names.get(feature, feature)
        print(f"{display_name:<20} {count:<15} {success_rate:<15.1f}%")
    
    # 最佳組合分析
    print("\n" + "="*80)
    print("TOP 5 FEATURE COMBINATIONS ACROSS ALL DIGITS")
    print("="*80)
    
    # 統計所有組合的出現次數
    combination_count = {}
    for result in summary_results:
        # 排除基本特徵，只看額外特徵的組合
        additional_combo = tuple(sorted([f for f in result['best_combination'] if f != 'basic']))
        if additional_combo not in combination_count:
            combination_count[additional_combo] = 0
        combination_count[additional_combo] += 1
    
    # 按出現次數排序，取前5名
    top_combinations = sorted(combination_count.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print(f"{'Rank':<6} {'Count':<8} {'Additional Features':<50}")
    print("-" * 64)
    
    for rank, (combo, count) in enumerate(top_combinations, 1):
        if combo:
            combo_names = [feature_display_names.get(f, f) for f in combo]
            combo_str = ', '.join(combo_names)
        else:
            combo_str = '無額外特徵'
        
        print(f"{rank:<6} {count:<8} {combo_str:<50}")
    
    # 每個數字的詳細結果（前3個數字）
    print("\n" + "="*80)
    print("DETAILED RESULTS FOR FIRST 3 DIGITS")
    print("="*80)
    
    for digit in list(all_digit_results.keys())[:3]:
        if digit in all_digit_results:
            print(f"\nDigit {digit} - Top 5 combinations:")
            all_results = all_digit_results[digit]['all_results']
            
            # 按分數排序，取前5名
            sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:5]
            
            for rank, result in enumerate(sorted_results, 1):
                additional_features = [f for f in result['combination'] if f != 'basic']
                if additional_features:
                    feature_names = [feature_display_names.get(f, f) for f in additional_features]
                    feature_str = ', '.join(feature_names)
                else:
                    feature_str = '無額外特徵'
                
                print(f"  {rank}. Score: {result['score']:.4f}, "
                      f"Dims: {result['info'].get('feature_dim', 0)}, "
                      f"Features: {feature_str}")


def main():
    """主函數"""
    print("Starting Exhaustive Feature Selection...")
    print("Fixed Feature: 基本特徵 (always included)")
    print("Testing all combinations of the remaining 6 features")
    
    # 檢查必要資料夾
    if not os.path.exists('handwrite') or not os.path.exists('handwrite/choice.txt'):
        print("Error: Cannot find required handwrite data")
        return
    
    # 初始化特徵提取器和選擇器
    feature_extractor = FeatureExtractor()
    selector = ExhaustiveFeatureSelector(feature_extractor)
    
    # 執行窮舉搜尋
    start_time = time.time()
    all_digit_results, summary_results = selector.run_exhaustive_search_for_all_digits()
    end_time = time.time()
    
    # 列印結果
    print_exhaustive_results(all_digit_results, summary_results)
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed: {len(summary_results)}/9 digits")
    print(f"Total combinations tested: {len(summary_results) * 64}")


if __name__ == "__main__":
    main()