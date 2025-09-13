# -*- coding: utf-8 -*-
# iterate_feature.py
# 這個檔案主要功能：
# 1. 基於 week3 架構實現 SVM 功能
# 2. 測試 9 種不同特徵的重要性
# 3. 使用遞增特徵組合法 + 個別特徵評估法
# 4. 找出最佳特徵組合和特徵重要性排序

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import itertools
from PIL import Image
from scipy import ndimage
from week3 import (
    extract_features_basic, extract_features_with_moments, 
    extract_features_with_intensity, extract_features_with_erosion,
    extract_features_all, calculate_intensity_features, calculate_erosion_features,
    process_images_basic, process_images_with_moments, process_images_with_intensity,
    process_images_with_erosion, process_images_all
)

class FeatureExtractor:
    """特徵提取器類別，負責管理所有不同類型的特徵提取"""
    
    def __init__(self):
        self.feature_names = [
            'vertical_segments',      # F1-F5: 垂直切割筆劃數
            'horizontal_segments',    # F6-F10: 水平切割筆劃數
            'second_moments',         # 二階動差特徵
            'third_moments',          # 三階動差特徵
            'intensity_mean',         # 筆劃強度平均值
            'intensity_std',          # 筆劃強度標準差
            'erosion_1',              # 第一級侵蝕特徵
            'erosion_2',              # 第二級侵蝕特徵
            'erosion_3',              # 第三級侵蝕特徵
            'centroid_x',             # 質心 x 座標
            'centroid_y',             # 質心 y 座標
            'bbox_width',             # 邊界框寬度
            'bbox_height',            # 邊界框高度
            'bbox_area',              # 邊界框面積
            'relative_centroid_x',    # 相對質心 x 座標 (相對於圖像中心)
            'relative_centroid_y',    # 相對質心 y 座標 (相對於圖像中心)
            'top_density',            # 上半部筆劃密度
            'bottom_density',         # 下半部筆劃密度
            'left_density',           # 左半部筆劃密度
            'right_density',          # 右半部筆劃密度
            'center_density'          # 中心區域筆劃密度
        ]
        
    def extract_coordinate_features(self, stroke_mask, Y):
        """提取座標和相對位置特徵"""
        total_strokes = np.sum(stroke_mask)
        
        if total_strokes == 0:
            return {
                'centroid_x': 0,
                'centroid_y': 0,
                'bbox_width': 0,
                'bbox_height': 0,
                'bbox_area': 0,
                'relative_centroid_x': 0,
                'relative_centroid_y': 0,
                'top_density': 0,
                'bottom_density': 0,
                'left_density': 0,
                'right_density': 0,
                'center_density': 0
            }
        
        # 找到筆劃像素的座標
        stroke_indices = np.where(stroke_mask > 0)
        y_coords = stroke_indices[0]
        x_coords = stroke_indices[1]
        
        # 1. 質心座標
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        # 2. 邊界框特徵
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        bbox_area = bbox_width * bbox_height
        
        # 3. 相對位置特徵（相對於圖像中心 95, 95）
        img_center_x, img_center_y = 95, 95
        relative_centroid_x = centroid_x - img_center_x
        relative_centroid_y = centroid_y - img_center_y
        
        # 4. 分佈特徵
        img_height, img_width = stroke_mask.shape
        
        # 上半部和下半部密度
        top_half = stroke_mask[:img_height//2, :]
        bottom_half = stroke_mask[img_height//2:, :]
        top_density = np.sum(top_half) / (img_height//2 * img_width)
        bottom_density = np.sum(bottom_half) / ((img_height - img_height//2) * img_width)
        
        # 左半部和右半部密度
        left_half = stroke_mask[:, :img_width//2]
        right_half = stroke_mask[:, img_width//2:]
        left_density = np.sum(left_half) / (img_height * img_width//2)
        right_density = np.sum(right_half) / (img_height * (img_width - img_width//2))
        
        # 中心區域密度（中心 1/4 區域）
        center_y_start = img_height // 4
        center_y_end = 3 * img_height // 4
        center_x_start = img_width // 4
        center_x_end = 3 * img_width // 4
        center_region = stroke_mask[center_y_start:center_y_end, center_x_start:center_x_end]
        center_area = (center_y_end - center_y_start) * (center_x_end - center_x_start)
        center_density = np.sum(center_region) / center_area if center_area > 0 else 0
        
        return {
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'bbox_area': bbox_area,
            'relative_centroid_x': relative_centroid_x,
            'relative_centroid_y': relative_centroid_y,
            'top_density': top_density,
            'bottom_density': bottom_density,
            'left_density': left_density,
            'right_density': right_density,
            'center_density': center_density
        }
        
    def extract_image_features(self, image_path, verbose=False):
        """提取單張圖像的所有特徵"""
        # 開啟圖片並轉灰階
        img = Image.open(image_path).convert('RGB')
        np_img = np.array(img)

        # RGB -> Grayscale
        Y = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
        binary_img = (Y > 220).astype(np.uint8) * 255
        stroke_mask = (binary_img == 0).astype(np.uint8)

        # 1. 垂直切割特徵 (F1-F5)
        cols = [38, 38, 37, 38, 38]
        col_start = 0
        vertical_features = []
        for w in cols:
            col_end = col_start + w
            count = np.sum(stroke_mask[:, col_start:col_end])
            vertical_features.append(count)
            col_start = col_end

        # 2. 水平切割特徵 (F6-F10)
        row_start = 0
        horizontal_features = []
        for h in cols:
            row_end = row_start + h
            count = np.sum(stroke_mask[row_start:row_end, :])
            horizontal_features.append(count)
            row_start = row_end

        # 3. 動差特徵計算
        total_strokes = np.sum(stroke_mask)
        
        if total_strokes > 0:
            # 創建坐標網格
            y_indices, x_indices = np.indices(stroke_mask.shape)
            center_x, center_y = 95, 95
            m_coords = x_indices - center_x
            n_coords = y_indices - center_y
            
            # 計算質心
            m0 = np.sum(m_coords * stroke_mask) / total_strokes
            n0 = np.sum(n_coords * stroke_mask) / total_strokes
            
            # 二階動差
            V_2_0 = np.sum(((m_coords - m0) ** 2) * stroke_mask) / total_strokes
            V_1_1 = np.sum((m_coords - m0) * (n_coords - n0) * stroke_mask) / total_strokes
            V_0_2 = np.sum(((n_coords - n0) ** 2) * stroke_mask) / total_strokes
            
            # 三階動差
            V_3_0 = np.sum(((m_coords - m0) ** 3) * stroke_mask) / total_strokes
            V_2_1 = np.sum(((m_coords - m0) ** 2) * (n_coords - n0) * stroke_mask) / total_strokes
            V_1_2 = np.sum((m_coords - m0) * ((n_coords - n0) ** 2) * stroke_mask) / total_strokes
            V_0_3 = np.sum(((n_coords - n0) ** 3) * stroke_mask) / total_strokes
            
            second_moments = [V_2_0, V_1_1, V_0_2]
            third_moments = [V_3_0, V_2_1, V_1_2, V_0_3]
        else:
            second_moments = [0, 0, 0]
            third_moments = [0, 0, 0, 0]

        # 4. 筆劃強度特徵
        intensity_features = calculate_intensity_features(Y, stroke_mask)
        
        # 5. 侵蝕特徵
        erosion_features = calculate_erosion_features(stroke_mask)
        
        # 6. 座標和相對位置特徵
        coordinate_features = self.extract_coordinate_features(stroke_mask, Y)
        
        # 組合成所有特徵
        features = {
            'vertical_segments': np.array(vertical_features),
            'horizontal_segments': np.array(horizontal_features),
            'second_moments': np.array(second_moments),
            'third_moments': np.array(third_moments),
            'intensity_mean': np.array([intensity_features[0]]),
            'intensity_std': np.array([intensity_features[1]]),
            'erosion_1': np.array([erosion_features[0]]),
            'erosion_2': np.array([erosion_features[1]]),
            'erosion_3': np.array([erosion_features[2]]),
            'centroid_x': np.array([coordinate_features['centroid_x']]),
            'centroid_y': np.array([coordinate_features['centroid_y']]),
            'bbox_width': np.array([coordinate_features['bbox_width']]),
            'bbox_height': np.array([coordinate_features['bbox_height']]),
            'bbox_area': np.array([coordinate_features['bbox_area']]),
            'relative_centroid_x': np.array([coordinate_features['relative_centroid_x']]),
            'relative_centroid_y': np.array([coordinate_features['relative_centroid_y']]),
            'top_density': np.array([coordinate_features['top_density']]),
            'bottom_density': np.array([coordinate_features['bottom_density']]),
            'left_density': np.array([coordinate_features['left_density']]),
            'right_density': np.array([coordinate_features['right_density']]),
            'center_density': np.array([coordinate_features['center_density']])
        }
        
        if verbose:
            print(f"提取特徵完成:")
            for name, feat in features.items():
                print(f"  {name}: {feat}")
        
        return features

    def combine_features(self, features_dict, selected_features):
        """組合選定的特徵"""
        combined = []
        for feat_name in selected_features:
            if feat_name in features_dict:
                combined.extend(features_dict[feat_name])
        return np.array(combined)

    def process_folder(self, folder_path, limit, offset=0, selected_features=None):
        """處理資料夾中的圖像並提取選定特徵"""
        if selected_features is None:
            selected_features = self.feature_names
        
        features_list = []
        if not os.path.exists(folder_path):
            print(f"警告：資料夾 {folder_path} 不存在")
            return np.array([])
        
        image_files = sorted(os.listdir(folder_path))
        selected_files = image_files[offset:offset + limit]
        
        for image_file in selected_files:
            image_path = os.path.join(folder_path, image_file)
            features_dict = self.extract_image_features(image_path)
            combined_features = self.combine_features(features_dict, selected_features)
            features_list.append(combined_features)
        
        if not features_list:
            return np.array([])
        return np.array(features_list)

class FeatureImportanceTester:
    """特徵重要性測試器"""
    
    def __init__(self, samples_per_class=25):
        self.samples_per_class = samples_per_class
        self.feature_extractor = FeatureExtractor()
        
    def test_single_feature_combination(self, selected_features, digits=range(1, 10)):
        """測試單一特徵組合的效果"""
        all_predictions = []
        all_test_labels = []
        
        for digit in digits:
            database_path = f'handwrite/{digit}/database'
            testcase_path = f'handwrite/{digit}/testcase'
            
            if not os.path.exists(database_path) or not os.path.exists(testcase_path):
                continue
            
            # 提取特徵
            train_features = np.vstack((
                self.feature_extractor.process_folder(database_path, self.samples_per_class, 
                                                    selected_features=selected_features),
                self.feature_extractor.process_folder(testcase_path, self.samples_per_class, 
                                                    selected_features=selected_features)
            ))
            
            answer_features = np.vstack((
                self.feature_extractor.process_folder(database_path, self.samples_per_class, 
                                                    offset=self.samples_per_class, 
                                                    selected_features=selected_features),
                self.feature_extractor.process_folder(testcase_path, self.samples_per_class, 
                                                    offset=self.samples_per_class, 
                                                    selected_features=selected_features)
            ))
            
            if train_features.size == 0 or answer_features.size == 0:
                continue
            
            # 標準化特徵
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            answer_features_scaled = scaler.transform(answer_features)
            
            # 檢查 NaN 值
            if np.isnan(train_features_scaled).any() or np.isnan(answer_features_scaled).any():
                continue
            
            # 訓練和測試 SVM
            train_labels = np.array([0] * self.samples_per_class + [1] * self.samples_per_class)
            test_labels = np.array([0] * self.samples_per_class + [1] * self.samples_per_class)
            
            model = SVC(kernel='linear', random_state=42)
            model.fit(train_features_scaled, train_labels)
            predictions = model.predict(answer_features_scaled)
            
            all_predictions.extend(predictions)
            all_test_labels.extend(test_labels)
        
        if all_test_labels:
            accuracy = accuracy_score(all_test_labels, all_predictions)
            return accuracy
        else:
            return 0.0

    def test_incremental_features(self):
        """遞增特徵組合測試"""
        print("開始遞增特徵組合測試...")
        
        results = []
        current_features = []
        remaining_features = self.feature_extractor.feature_names.copy()
        
        # 基準測試：只使用垂直和水平切割
        baseline_features = ['vertical_segments', 'horizontal_segments']
        baseline_acc = self.test_single_feature_combination(baseline_features)
        results.append({
            'features': baseline_features.copy(),
            'accuracy': baseline_acc,
            'improvement': 0.0
        })
        
        current_features = baseline_features.copy()
        for feat in baseline_features:
            if feat in remaining_features:
                remaining_features.remove(feat)
        
        print(f"基準準確率 (垂直+水平切割): {baseline_acc:.4f}")
        
        # 遞增添加特徵
        while remaining_features:
            best_feature = None
            best_accuracy = 0.0
            
            print(f"\n測試添加剩餘特徵: {remaining_features}")
            
            for feature in remaining_features:
                test_features = current_features + [feature]
                accuracy = self.test_single_feature_combination(test_features)
                print(f"  {feature}: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature
            
            if best_feature:
                current_features.append(best_feature)
                remaining_features.remove(best_feature)
                improvement = best_accuracy - results[-1]['accuracy']
                
                results.append({
                    'features': current_features.copy(),
                    'accuracy': best_accuracy,
                    'improvement': improvement
                })
                
                print(f"選擇特徵: {best_feature}, 準確率: {best_accuracy:.4f}, 提升: {improvement:.4f}")
            else:
                print("沒有找到能提升準確率的特徵")
                break
        
        return results

    def test_individual_feature_importance(self):
        """個別特徵重要性測試"""
        print("\n開始個別特徵重要性測試...")
        
        # 測試每個特徵的獨立貢獻
        individual_results = []
        
        for feature in self.feature_extractor.feature_names:
            accuracy = self.test_single_feature_combination([feature])
            individual_results.append({
                'feature': feature,
                'accuracy': accuracy
            })
            print(f"{feature}: {accuracy:.4f}")
        
        # 排序
        individual_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        return individual_results

    def test_feature_combinations(self):
        """測試不同特徵組合"""
        print("\n開始特徵組合測試...")
        
        # 測試一些預定義的組合
        predefined_combinations = [
            ['vertical_segments', 'horizontal_segments'],
            ['vertical_segments', 'horizontal_segments', 'second_moments'],
            ['vertical_segments', 'horizontal_segments', 'second_moments', 'third_moments'],
            ['vertical_segments', 'horizontal_segments', 'second_moments', 'third_moments', 'intensity_mean', 'intensity_std'],
            ['vertical_segments', 'horizontal_segments', 'second_moments', 'third_moments', 'erosion_1', 'erosion_2', 'erosion_3'],
            self.feature_extractor.feature_names  # 所有特徵
        ]
        
        combination_results = []
        
        for i, features in enumerate(predefined_combinations):
            accuracy = self.test_single_feature_combination(features)
            combination_results.append({
                'combination': f"組合{i+1}",
                'features': features,
                'accuracy': accuracy,
                'feature_count': len(features)
            })
            print(f"組合{i+1} ({len(features)}特徵): {accuracy:.4f}")
            print(f"  特徵: {features}")
        
        return combination_results

    def visualize_results(self, incremental_results, individual_results, combination_results):
        """可視化結果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 遞增特徵組合結果
        steps = range(len(incremental_results))
        accuracies = [r['accuracy'] for r in incremental_results]
        improvements = [r['improvement'] for r in incremental_results]
        
        ax1.plot(steps, accuracies, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel('添加特徵步驟')
        ax1.set_ylabel('準確率')
        ax1.set_title('遞增特徵組合效果')
        ax1.grid(True, alpha=0.3)
        
        # 標記每個點的特徵數量
        for i, (step, acc) in enumerate(zip(steps, accuracies)):
            ax1.annotate(f'{len(incremental_results[i]["features"])}', 
                        (step, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        # 2. 個別特徵重要性
        features = [r['feature'] for r in individual_results]
        individual_accs = [r['accuracy'] for r in individual_results]
        
        ax2.barh(features, individual_accs, color='skyblue')
        ax2.set_xlabel('準確率')
        ax2.set_title('個別特徵重要性')
        ax2.grid(True, alpha=0.3)
        
        # 3. 特徵組合比較
        combo_names = [r['combination'] for r in combination_results]
        combo_accs = [r['accuracy'] for r in combination_results]
        combo_counts = [r['feature_count'] for r in combination_results]
        
        bars = ax3.bar(combo_names, combo_accs, color='lightgreen')
        ax3.set_ylabel('準確率')
        ax3.set_title('預定義特徵組合比較')
        ax3.grid(True, alpha=0.3)
        
        # 標記特徵數量
        for bar, count in zip(bars, combo_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{count}', ha='center', va='bottom')
        
        # 4. 特徵提升效果
        if len(incremental_results) > 1:
            improvement_steps = range(1, len(incremental_results))
            improvement_values = [r['improvement'] for r in incremental_results[1:]]
            
            ax4.bar(improvement_steps, improvement_values, color='orange')
            ax4.set_xlabel('添加特徵步驟')
            ax4.set_ylabel('準確率提升')
            ax4.set_title('每步特徵提升效果')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self):
        """執行完整分析"""
        print("=== 特徵重要性完整分析 ===")
        
        # 1. 遞增特徵組合測試
        incremental_results = self.test_incremental_features()
        
        # 2. 個別特徵重要性測試
        individual_results = self.test_individual_feature_importance()
        
        # 3. 特徵組合測試
        combination_results = self.test_feature_combinations()
        
        # 4. 結果總結
        print("\n=== 分析結果總結 ===")
        
        print("\n最佳遞增特徵組合:")
        best_incremental = max(incremental_results, key=lambda x: x['accuracy'])
        print(f"  準確率: {best_incremental['accuracy']:.4f}")
        print(f"  特徵: {best_incremental['features']}")
        
        print("\n個別特徵重要性排序:")
        for i, result in enumerate(individual_results[:5]):
            print(f"  {i+1}. {result['feature']}: {result['accuracy']:.4f}")
        
        print("\n最佳特徵組合:")
        best_combination = max(combination_results, key=lambda x: x['accuracy'])
        print(f"  準確率: {best_combination['accuracy']:.4f}")
        print(f"  特徵數量: {best_combination['feature_count']}")
        
        # 5. 可視化結果
        self.visualize_results(incremental_results, individual_results, combination_results)
        
        return incremental_results, individual_results, combination_results

def main():
    """主函數"""
    print("開始特徵重要性分析...")
    
    # 創建測試器
    tester = FeatureImportanceTester(samples_per_class=25)
    
    # 執行完整分析
    incremental_results, individual_results, combination_results = tester.run_full_analysis()
    
    # 保存結果
    print("\n保存結果到文件...")
    with open('feature_analysis_results.txt', 'w', encoding='utf-8') as f:
        f.write("=== 特徵重要性分析結果 ===\n\n")
        
        f.write("遞增特徵組合結果:\n")
        for i, result in enumerate(incremental_results):
            f.write(f"步驟{i+1}: 準確率={result['accuracy']:.4f}, 特徵={result['features']}\n")
        
        f.write("\n個別特徵重要性排序:\n")
        for i, result in enumerate(individual_results):
            f.write(f"{i+1}. {result['feature']}: {result['accuracy']:.4f}\n")
        
        f.write("\n特徵組合比較:\n")
        for result in combination_results:
            f.write(f"{result['combination']}: 準確率={result['accuracy']:.4f}, 特徵數量={result['feature_count']}\n")
    
    print("分析完成！結果已保存到 feature_analysis_results.txt")

if __name__ == "__main__":
    main()