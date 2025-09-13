# -*- coding: utf-8 -*-
# random_sampling_svm.py
# Main functions:
# 1. Use selected feature combination: Basic + Second Moments + Erosion + Coordinate
# 2. Randomly sample 25 images from each category for training/testing
# 3. Repeat 30 times and analyze statistical results
# 4. Visualize results with charts

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Import required modules
from week7 import (
    read_standard_files, extract_characteristic_points, distance,
    phi_matrix, coordinate_normalization_m_custom, coordinate_normalization_n_custom,
    stroke_ratio_m_custom, stroke_ratio_n_custom
)
from week3 import (
    extract_features_basic, extract_features_with_moments, 
    calculate_erosion_features
)

# Set font for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class SelectedFeatureExtractor:
    """Feature extractor for selected feature combination"""
    
    def __init__(self, direction_weight=8.0, normalized_coordinates_weight=1.0/1000, stroke_ratio_weight=1.0/10):
        self.direction_weight = direction_weight
        self.normalized_coordinates_weight = normalized_coordinates_weight
        self.stroke_ratio_weight = stroke_ratio_weight
        self.reliable_points_cache = {}
        
    def find_reliable_feature_points(self, standard_image_path, database_images, L=3, d=12):
        """Find reliable feature points"""
        cache_key = (standard_image_path, tuple(database_images[:5]))
        if cache_key in self.reliable_points_cache:
            return self.reliable_points_cache[cache_key]
            
        # Extract features from standard image
        standard_end_points, standard_turning_points = extract_characteristic_points(standard_image_path, L, d)
        
        if not standard_end_points:
            return [], []
        
        # Use first 5 database images for analysis
        analysis_images = database_images[:5]
        
        # Analyze reliability of end points
        reliable_end_points = self._analyze_point_reliability(
            standard_end_points, analysis_images, 'end_point', L, d
        )
        
        # Analyze reliability of turning points
        reliable_turning_points = self._analyze_point_reliability(
            standard_turning_points, analysis_images, 'turning_point', L, d
        )
        
        result = (reliable_end_points, reliable_turning_points)
        self.reliable_points_cache[cache_key] = result
        return result
    
    def _analyze_point_reliability(self, standard_points, analysis_images, point_type, L, d):
        """Analyze reliability of feature points"""
        reliable_points = []
        
        for i, std_point in enumerate(standard_points):
            distances = []
            
            for img_path in analysis_images:
                try:
                    test_end_points, test_turning_points = extract_characteristic_points(img_path, L, d)
                    
                    if point_type == 'end_point':
                        test_points = test_end_points
                    else:
                        test_points = test_turning_points
                    
                    if not test_points:
                        continue
                    
                    # Find best matching point
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
            
            # Check if this standard point is reliable
            if len(distances) >= 3:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                outliers = [d for d in distances if d > mean_dist + std_dist]
                
                # If outliers are less than 40% of total, consider this point reliable
                if len(outliers) / len(distances) < 0.4:
                    reliable_points.append({
                        'point': std_point,
                        'index': i,
                        'mean_distance': mean_dist,
                        'std_distance': std_dist,
                        'match_count': len(distances)
                    })
        
        return reliable_points
    
    def extract_selected_features(self, image_path, reliable_end_points=None, reliable_turning_points=None, debug=False):
        """
        Extract selected feature combination:
        - Basic features (10)
        - Second moments (3) 
        - Erosion features (3)
        - Coordinate features (2K)
        - Phi matrix features (direction features)
        - Normalized coordinate features
        - Stroke ratio features
        """
        features = []
        feature_breakdown = {}
        
        # Extract basic features and moments
        from week3 import extract_features_with_moments
        moments_features, Y, stroke_mask = extract_features_with_moments(image_path, verbose=False)
        
        # Basic features (first 10)
        basic_features = moments_features[:10]
        features.extend(basic_features)
        feature_breakdown['basic'] = len(basic_features)
        
        # Second moments (positions 10-13)
        second_moments = moments_features[10:13]
        features.extend(second_moments)
        feature_breakdown['second_moments'] = len(second_moments)
        
        # Erosion features
        erosion_features = calculate_erosion_features(stroke_mask)
        features.extend(erosion_features)
        feature_breakdown['erosion'] = len(erosion_features)
        
        # Coordinate features
        coord_features = []
        week7_features = []
        if reliable_end_points is not None and reliable_turning_points is not None:
            coord_features = self._extract_coordinate_features(reliable_end_points, reliable_turning_points)
            features.extend(coord_features)
            feature_breakdown['coordinate'] = len(coord_features)
            
            # Week7 additional features - only if we have reliable points
            week7_features = self._extract_week7_features(reliable_end_points, reliable_turning_points)
            features.extend(week7_features)
            feature_breakdown['week7'] = len(week7_features)
        else:
            feature_breakdown['coordinate'] = 0
            feature_breakdown['week7'] = 0
        
        if debug:
            print(f"Feature breakdown for {image_path}:")
            print(f"  Basic: {feature_breakdown['basic']}")
            print(f"  Second moments: {feature_breakdown['second_moments']}")
            print(f"  Erosion: {feature_breakdown['erosion']}")
            print(f"  Coordinate: {feature_breakdown['coordinate']}")
            print(f"  Week7: {feature_breakdown['week7']}")
            print(f"  Total: {len(features)}")
            if reliable_end_points is not None and reliable_turning_points is not None:
                print(f"  Reliable end points: {len(reliable_end_points)}")
                print(f"  Reliable turning points: {len(reliable_turning_points)}")
            print()
        
        return np.array(features), feature_breakdown
    
    def _extract_coordinate_features(self, reliable_end_points, reliable_turning_points):
        """Extract coordinate features"""
        coordinate_features = []
        
        # Extract end points coordinates
        for point_info in reliable_end_points:
            coord = point_info['point']['coordinate']
            coordinate_features.extend([coord[1], coord[0]])  # (x, y)
        
        # Extract turning points coordinates
        for point_info in reliable_turning_points:
            coord = point_info['point']['coordinate']
            coordinate_features.extend([coord[1], coord[0]])  # (x, y)
        
        return coordinate_features
    
    def _extract_week7_features(self, reliable_end_points, reliable_turning_points):
        """Extract week7 additional features using week7 functions directly"""
        week7_features = []
        
        # Collect all reliable points
        all_reliable_points = []
        for point_info in reliable_end_points:
            all_reliable_points.append(point_info['point'])
        for point_info in reliable_turning_points:
            all_reliable_points.append(point_info['point'])
        
        if len(all_reliable_points) < 1:
            return []
        
        try:
            # Extract phi_matrix features using week7's phi_matrix function
            phi_features = []
            for point in all_reliable_points:
                if 'coordinate' in point:
                    n, m = point['coordinate']  # (row, col) format
                    # Use week7's phi_matrix function with L=3
                    phi_mat = phi_matrix(3)  # This returns a 7x7 complex matrix
                    # Calculate the angle using the phi matrix approach
                    # We can use the angle already computed in the point
                    if 'angle' in point:
                        angle_val = point['angle']
                        phi_features.append(np.cos(angle_val))
                        phi_features.append(np.sin(angle_val))
            
            if phi_features:
                week7_features.extend(phi_features)
            
            # Extract normalized coordinate features using week7's functions
            normalized_coords = []
            for point in all_reliable_points:
                if 'coordinate' in point:
                    n, m = point['coordinate']  # (row, col) format
                    # We need stroke_indices for the custom functions
                    # Since we don't have access to the original image here,
                    # we'll use the pre-computed values in the point
                    if 'm_hat' in point and 'n_hat' in point:
                        normalized_coords.append(point['m_hat'])
                        normalized_coords.append(point['n_hat'])
            
            if normalized_coords:
                week7_features.extend(normalized_coords)
            
            # Extract stroke ratio features using week7's functions
            stroke_ratios = []
            for point in all_reliable_points:
                if 'm_ratio' in point and 'n_ratio' in point:
                    stroke_ratios.append(point['m_ratio'])
                    stroke_ratios.append(point['n_ratio'])
            
            if stroke_ratios:
                week7_features.extend(stroke_ratios)
                
        except Exception as e:
            print(f"Warning: Error extracting week7 features: {e}")
            import traceback
            traceback.print_exc()
            pass
        
        return week7_features


class RandomSamplingSVM:
    """Random sampling SVM trainer and evaluator"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.results = defaultdict(list)
        
    def random_sample_images(self, database_images, testcase_images, sample_size=25, random_seed=None):
        """Randomly sample images for training and testing"""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Ensure we have enough images
        if len(database_images) < sample_size * 2 or len(testcase_images) < sample_size * 2:
            raise ValueError(f"Not enough images for sampling. Need at least {sample_size * 2} each.")
        
        # Random sampling without replacement
        train_database = random.sample(database_images, sample_size)
        remaining_database = [img for img in database_images if img not in train_database]
        test_database = random.sample(remaining_database, min(sample_size, len(remaining_database)))
        
        train_testcase = random.sample(testcase_images, sample_size)
        remaining_testcase = [img for img in testcase_images if img not in train_testcase]
        test_testcase = random.sample(remaining_testcase, min(sample_size, len(remaining_testcase)))
        
        return train_database, train_testcase, test_database, test_testcase
    
    def train_and_evaluate_single_run(self, digit, run_id):
        """Train and evaluate SVM for a single run"""
        try:
            # Setup paths
            standard_files = read_standard_files()
            if digit not in standard_files:
                print(f"Error: digit {digit} not in standard_files")
                return None
            
            standard_file = standard_files[digit]
            standard_path = f'handwrite/{digit}/database/{standard_file}'
            database_path = f'handwrite/{digit}/database'
            testcase_path = f'handwrite/{digit}/testcase'
            
            # Check paths
            if not all(os.path.exists(path) for path in [standard_path, database_path, testcase_path]):
                print(f"Error: Missing paths for digit {digit}")
                print(f"  standard_path exists: {os.path.exists(standard_path)}")
                print(f"  database_path exists: {os.path.exists(database_path)}")
                print(f"  testcase_path exists: {os.path.exists(testcase_path)}")
                return None
            
            # Collect images (including standard image)
            database_images = []
            for filename in sorted(os.listdir(database_path)):
                if filename.endswith('.bmp'):  # Include all .bmp files including standard
                    database_images.append(os.path.join(database_path, filename))
            
            testcase_images = []
            for filename in sorted(os.listdir(testcase_path)):
                if filename.endswith('.bmp'):
                    testcase_images.append(os.path.join(testcase_path, filename))
            
            # Adjust minimum requirement since we now have 50 database images
            min_required = 20  # Lower requirement for meaningful sampling
            if len(database_images) < min_required or len(testcase_images) < min_required:
                print(f"Error: Not enough images for digit {digit}")
                print(f"  database_images: {len(database_images)} (need >={min_required})")
                print(f"  testcase_images: {len(testcase_images)} (need >={min_required})")
                return None
            
            # Random sampling
            try:
                train_db, train_tc, test_db, test_tc = self.random_sample_images(
                    database_images, testcase_images, sample_size=25, random_seed=run_id
                )
            except Exception as e:
                print(f"Error in random sampling for digit {digit}: {e}")
                return None
            
            # Find reliable feature points
            try:
                reliable_end_points, reliable_turning_points = self.feature_extractor.find_reliable_feature_points(
                    standard_path, database_images
                )
            except Exception as e:
                print(f"Error finding reliable points for digit {digit}: {e}")
                return None
            
            # Extract training features
            train_features = []
            train_labels = []
            
            # Database training data (correct handwriting)
            for img_path in train_db:
                try:
                    features, breakdown = self.feature_extractor.extract_selected_features(
                        img_path, reliable_end_points, reliable_turning_points
                    )
                    if len(features) > 0:
                        train_features.append(features)
                        train_labels.append(1)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    continue
            
            # Testcase training data (incorrect handwriting)
            for img_path in train_tc:
                try:
                    features, breakdown = self.feature_extractor.extract_selected_features(
                        img_path, reliable_end_points, reliable_turning_points
                    )
                    if len(features) > 0:
                        train_features.append(features)
                        train_labels.append(0)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    continue
            
            # Extract testing features
            test_features = []
            test_labels = []
            
            # Database testing data (correct handwriting)
            for img_path in test_db:
                try:
                    features, breakdown = self.feature_extractor.extract_selected_features(
                        img_path, reliable_end_points, reliable_turning_points
                    )
                    if len(features) > 0:
                        test_features.append(features)
                        test_labels.append(1)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    continue
            
            # Testcase testing data (incorrect handwriting)
            for img_path in test_tc:
                try:
                    features, breakdown = self.feature_extractor.extract_selected_features(
                        img_path, reliable_end_points, reliable_turning_points
                    )
                    if len(features) > 0:
                        test_features.append(features)
                        test_labels.append(0)
                except Exception as e:
                    print(f"Error extracting features from {img_path}: {e}")
                    continue
            
            if len(train_features) == 0 or len(test_features) == 0:
                print(f"Error: No features extracted for digit {digit}")
                print(f"  train_features: {len(train_features)}")
                print(f"  test_features: {len(test_features)}")
                return None
            
            # Convert to numpy arrays and standardize
            train_features = np.array(train_features)
            test_features = np.array(test_features)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)
            
            # Check feature dimension consistency
            if train_features.shape[1] != test_features.shape[1]:
                print(f"Error: Feature dimension mismatch for digit {digit}")
                print(f"  train_features shape: {train_features.shape}")
                print(f"  test_features shape: {test_features.shape}")
                return None
            
            # Standardization
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            # Train SVM
            svm_model = SVC(kernel='linear', random_state=42)
            svm_model.fit(train_features_scaled, train_labels)
            
            # Evaluation
            train_pred = svm_model.predict(train_features_scaled)
            test_pred = svm_model.predict(test_features_scaled)
            
            train_acc = accuracy_score(train_labels, train_pred)
            test_acc = accuracy_score(test_labels, test_pred)
            
            result = {
                'digit': digit,
                'run_id': run_id,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'feature_dim': train_features.shape[1],
                'train_samples': len(train_features),
                'test_samples': len(test_features),
                'reliable_points': len(reliable_end_points) + len(reliable_turning_points) if reliable_end_points else 0
            }
            
            return result
            
        except Exception as e:
            print(f"Unexpected error in run {run_id} for digit {digit}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_multiple_experiments(self, digits=range(1, 10), num_runs=30):
        """Run multiple experiments for all digits"""
        print(f"Starting {num_runs} random sampling experiments for digits {list(digits)}...")
        print("Selected features: Basic + Second Moments + Erosion + Coordinate")
        
        all_results = []
        
        for digit in digits:
            print(f"\nProcessing digit {digit}...")
            digit_results = []
            
            for run_id in range(num_runs):
                print(f"  Run {run_id + 1}/{num_runs}", end=' ')
                
                result = self.train_and_evaluate_single_run(digit, run_id)
                if result:
                    digit_results.append(result)
                    all_results.append(result)
                    print(f"- Test Acc: {result['test_accuracy']:.3f}")
                else:
                    print("- Failed")
            
            if digit_results:
                self.results[digit] = digit_results
                # Print summary for this digit
                test_accs = [r['test_accuracy'] for r in digit_results]
                print(f"  Digit {digit} Summary:")
                print(f"    Successful runs: {len(digit_results)}/{num_runs}")
                print(f"    Test accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
                print(f"    Range: [{np.min(test_accs):.3f}, {np.max(test_accs):.3f}]")
        
        return all_results


def create_statistical_analysis_charts(results_dict):
    """Create statistical analysis charts"""
    
    # Prepare data
    digits = sorted(results_dict.keys())
    stats_data = []
    
    for digit in digits:
        results = results_dict[digit]
        if results:
            test_accs = [r['test_accuracy'] * 100 for r in results]
            train_accs = [r['train_accuracy'] * 100 for r in results]
            
            stats_data.append({
                'digit': digit,
                'test_mean': np.mean(test_accs),
                'test_std': np.std(test_accs),
                'test_min': np.min(test_accs),
                'test_max': np.max(test_accs),
                'train_mean': np.mean(train_accs),
                'train_std': np.std(train_accs),
                'train_min': np.min(train_accs),
                'train_max': np.max(train_accs),
                'test_accs': test_accs,
                'train_accs': train_accs
            })
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Statistical summary table visualization
    ax1 = fig.add_subplot(gs[0, :2])
    
    table_data = []
    for stat in stats_data:
        table_data.append([
            f"Digit {stat['digit']}",
            f"{stat['test_mean']:.1f}%",
            f"{stat['test_std']:.1f}%",
            f"{stat['test_min']:.1f}%",
            f"{stat['test_max']:.1f}%"
        ])
    
    ax1.axis('tight')
    ax1.axis('off')
    table = ax1.table(cellText=table_data,
                     colLabels=['Digit', 'Mean', 'Std Dev', 'Min', 'Max'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax1.set_title('Test Accuracy Statistics Summary (30 Runs Each)', fontsize=14, pad=20)
    
    # 2. Box plot of accuracies
    ax2 = fig.add_subplot(gs[0, 2:])
    
    test_data_for_box = [stats_data[i]['test_accs'] for i in range(len(digits))]
    bp = ax2.boxplot(test_data_for_box, labels=[f'Digit {d}' for d in digits], patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(digits)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy Distribution by Digit')
    ax2.grid(True, alpha=0.3)
    
    # 3. Mean accuracy with error bars
    ax3 = fig.add_subplot(gs[1, :2])
    
    means = [stat['test_mean'] for stat in stats_data]
    stds = [stat['test_std'] for stat in stats_data]
    
    bars = ax3.bar(digits, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=colors, error_kw={'linewidth': 2})
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Mean Test Accuracy with Standard Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. Range visualization (min-max)
    ax4 = fig.add_subplot(gs[1, 2:])
    
    mins = [stat['test_min'] for stat in stats_data]
    maxs = [stat['test_max'] for stat in stats_data]
    ranges = [maxs[i] - mins[i] for i in range(len(digits))]
    
    # Create range plot
    for i, digit in enumerate(digits):
        ax4.plot([digit, digit], [mins[i], maxs[i]], 'o-', linewidth=3, 
                markersize=8, color=colors[i], alpha=0.7)
        ax4.text(digit, maxs[i] + 0.5, f'{ranges[i]:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    ax4.set_xlabel('Digit')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Accuracy Range (Min-Max) by Digit')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(digits)
    
    # 5. Training vs Testing accuracy comparison
    ax5 = fig.add_subplot(gs[2, :2])
    
    train_means = [stat['train_mean'] for stat in stats_data]
    test_means = [stat['test_mean'] for stat in stats_data]
    
    x = np.arange(len(digits))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, train_means, width, label='Training Accuracy', 
                    alpha=0.8, color='skyblue')
    bars2 = ax5.bar(x + width/2, test_means, width, label='Testing Accuracy', 
                    alpha=0.8, color='lightcoral')
    
    ax5.set_xlabel('Digit')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Training vs Testing Accuracy Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'Digit {d}' for d in digits])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Stability analysis (coefficient of variation)
    ax6 = fig.add_subplot(gs[2, 2:])
    
    cv_values = [(stat['test_std'] / stat['test_mean']) * 100 for stat in stats_data]
    
    bars = ax6.bar(digits, cv_values, alpha=0.7, color=colors)
    ax6.set_xlabel('Digit')
    ax6.set_ylabel('Coefficient of Variation (%)')
    ax6.set_title('Model Stability (Lower is More Stable)')
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, cv in zip(bars, cv_values):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Random Sampling SVM Results Analysis (30 Runs per Digit)', 
                 fontsize=16, fontweight='bold')
    plt.savefig('random_sampling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_data

def print_detailed_statistics(stats_data):
    """Print detailed statistics"""
    print("\n" + "="*80)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*80)
    
    print(f"\n{'Digit':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'CV%':<8}")
    print("-" * 64)
    
    overall_means = []
    overall_stds = []
    
    for stat in stats_data:
        mean_acc = stat['test_mean']
        std_acc = stat['test_std']
        min_acc = stat['test_min']
        max_acc = stat['test_max']
        range_acc = max_acc - min_acc
        cv = (std_acc / mean_acc) * 100
        
        overall_means.append(mean_acc)
        overall_stds.append(std_acc)
        
        print(f"{stat['digit']:<8} {mean_acc:<8.2f} {std_acc:<8.2f} {min_acc:<8.2f} "
              f"{max_acc:<8.2f} {range_acc:<8.2f} {cv:<8.2f}")
    
    print("-" * 64)
    overall_mean = np.mean(overall_means)
    overall_std = np.mean(overall_stds)
    overall_min = np.min([stat['test_min'] for stat in stats_data])
    overall_max = np.max([stat['test_max'] for stat in stats_data])
    overall_range = overall_max - overall_min
    
    print(f"{'Overall':<8} {overall_mean:<8.2f} {overall_std:<8.2f} {overall_min:<8.2f} "
          f"{overall_max:<8.2f} {overall_range:<8.2f} {'-':<8}")
    
    print(f"\nFeature Configuration:")
    print(f"• Basic Features: 10 (f1-f5 vertical, f6-f10 horizontal)")
    print(f"• Second Moments: 3 (V_2_0, V_1_1, V_0_2)")
    print(f"• Erosion Features: 3 (e1, e2, e3)")
    print(f"• Coordinate Features: 2K (K = number of reliable points)")
    print(f"• Total per digit: ~16-30 features (depends on reliable points)")


def debug_feature_extraction():
    """Debug function to check feature extraction"""
    print("=== Feature Extraction Debug ===")
    
    # Test with one digit to see feature breakdown
    from week7 import read_standard_files
    standard_files = read_standard_files()
    digit = 1
    
    if digit in standard_files:
        standard_file = standard_files[digit]
        standard_path = f'handwrite/{digit}/database/{standard_file}'
        database_path = f'handwrite/{digit}/database'
        
        if os.path.exists(standard_path) and os.path.exists(database_path):
            # Initialize feature extractor
            feature_extractor = SelectedFeatureExtractor()
            
            # Get database images
            database_images = []
            for filename in sorted(os.listdir(database_path)):
                if filename.endswith('.bmp'):
                    database_images.append(os.path.join(database_path, filename))
            
            print(f"Testing with digit {digit}, {len(database_images)} images")
            
            # Find reliable points
            reliable_end_points, reliable_turning_points = feature_extractor.find_reliable_feature_points(
                standard_path, database_images
            )
            
            print(f"Reliable points found: {len(reliable_end_points)} end + {len(reliable_turning_points)} turning")
            
            # Test feature extraction on first few images
            test_images = database_images[:3]
            for i, img_path in enumerate(test_images):
                print(f"\n--- Image {i+1}: {os.path.basename(img_path)} ---")
                features, breakdown = feature_extractor.extract_selected_features(
                    img_path, reliable_end_points, reliable_turning_points, debug=True
                )
                
                # Print some feature values
                feature_start = 0
                for feature_type, count in breakdown.items():
                    if count > 0:
                        feature_end = feature_start + count
                        print(f"  {feature_type} features ({count}): {features[feature_start:feature_end][:5]}..." if count > 5 else f"  {feature_type} features ({count}): {features[feature_start:feature_end]}")
                        feature_start = feature_end
                
                print(f"  Feature vector sample (first 10): {features[:10]}")
                print(f"  Feature vector sample (last 10): {features[-10:]}")
            
            # Compare with and without week7 features
            print(f"\n=== Comparison: With vs Without Week7 Features ===")
            test_img = test_images[0]
            
            # Without week7 features (set reliable points to None)
            features_without, _ = feature_extractor.extract_selected_features(test_img, None, None)
            
            # With week7 features
            features_with, breakdown_with = feature_extractor.extract_selected_features(test_img, reliable_end_points, reliable_turning_points)
            
            print(f"Without week7 features: {len(features_without)} dimensions")
            print(f"With week7 features: {len(features_with)} dimensions")
            print(f"Difference: {len(features_with) - len(features_without)} additional features")
            print(f"Breakdown with week7: {breakdown_with}")
            
            return True
    
    return False

def main():
    """Main function"""
    print("Random Sampling SVM with Selected Features")
    print("Features: Basic + Second Moments + Erosion + Coordinate")
    print("Sampling: 25 random images per category, 30 runs per digit")
    
    # Initialize feature extractor and SVM trainer
    feature_extractor = SelectedFeatureExtractor()
    svm_trainer = RandomSamplingSVM(feature_extractor)
    
    # Run experiments
    start_time = time.time()
    all_results = svm_trainer.run_multiple_experiments(digits=range(1, 10), num_runs=30)
    end_time = time.time()
    
    # Create visualizations and analysis
    if svm_trainer.results:
        print("\nGenerating statistical analysis charts...")
        stats_data = create_statistical_analysis_charts(svm_trainer.results)
        
        print_detailed_statistics(stats_data)
        
        print(f"\nExperiment completed!")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Total experiments: {len(all_results)}")
        print(f"Generated chart: random_sampling_analysis.png")
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    # Run debug first
    debug_success = debug_feature_extraction()
    if debug_success:
        print("\n" + "="*50)
        print("Debug completed. Starting main experiment...")
        print("="*50)
    else:
        print("Debug failed, but continuing with main experiment...")
    
    main()