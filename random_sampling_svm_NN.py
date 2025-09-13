# -*- coding: utf-8 -*-
# random_sampling_svm_NN.py
# Main functions:
# 1. Use selected feature combination: Basic + Second Moments + Erosion + Coordinate
# 2. Randomly sample 25 images from each category for training/testing
# 3. Compare SVM vs Neural Network performance
# 4. Repeat 30 times and analyze statistical results
# 5. Visualize results with charts

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import random
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
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


class RandomSamplingComparison:
    """Random sampling SVM vs Neural Network trainer and evaluator"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.results_svm = defaultdict(list)
        self.results_nn = defaultdict(list)
        
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
    
    def train_and_evaluate_single_run(self, digit, run_id, model_type='both'):
        """Train and evaluate both SVM and NN for a single run"""
        try:
            # Setup paths
            standard_files = read_standard_files()
            if digit not in standard_files:
                print(f"Error: digit {digit} not in standard_files")
                return None, None
            
            standard_file = standard_files[digit]
            standard_path = f'handwrite/{digit}/database/{standard_file}'
            database_path = f'handwrite/{digit}/database'
            testcase_path = f'handwrite/{digit}/testcase'
            
            # Check paths
            if not all(os.path.exists(path) for path in [standard_path, database_path, testcase_path]):
                return None, None
            
            # Collect images
            database_images = []
            for filename in sorted(os.listdir(database_path)):
                if filename.endswith('.bmp'):
                    database_images.append(os.path.join(database_path, filename))
            
            testcase_images = []
            for filename in sorted(os.listdir(testcase_path)):
                if filename.endswith('.bmp'):
                    testcase_images.append(os.path.join(testcase_path, filename))
            
            min_required = 20
            if len(database_images) < min_required or len(testcase_images) < min_required:
                return None, None
            
            # Random sampling
            try:
                train_db, train_tc, test_db, test_tc = self.random_sample_images(
                    database_images, testcase_images, sample_size=25, random_seed=run_id
                )
            except Exception as e:
                return None, None
            
            # Find reliable feature points
            try:
                reliable_end_points, reliable_turning_points = self.feature_extractor.find_reliable_feature_points(
                    standard_path, database_images
                )
            except Exception as e:
                return None, None
            
            # Extract features
            train_features, train_labels = self._extract_features_batch(
                train_db + train_tc, [1] * len(train_db) + [0] * len(train_tc),
                reliable_end_points, reliable_turning_points
            )
            
            test_features, test_labels = self._extract_features_batch(
                test_db + test_tc, [1] * len(test_db) + [0] * len(test_tc),
                reliable_end_points, reliable_turning_points
            )
            
            if len(train_features) == 0 or len(test_features) == 0:
                return None, None
            
            # Convert to numpy arrays and standardize
            train_features = np.array(train_features)
            test_features = np.array(test_features)
            train_labels = np.array(train_labels)
            test_labels = np.array(test_labels)
            
            # Standardization
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            # Train and evaluate SVM
            result_svm = None
            if model_type in ['svm', 'both']:
                svm_model = SVC(kernel='linear', random_state=42)
                svm_model.fit(train_features_scaled, train_labels)
                
                train_pred_svm = svm_model.predict(train_features_scaled)
                test_pred_svm = svm_model.predict(test_features_scaled)
                
                train_acc_svm = accuracy_score(train_labels, train_pred_svm)
                test_acc_svm = accuracy_score(test_labels, test_pred_svm)
                
                result_svm = {
                    'digit': digit,
                    'run_id': run_id,
                    'model': 'SVM',
                    'train_accuracy': train_acc_svm,
                    'test_accuracy': test_acc_svm,
                    'feature_dim': train_features.shape[1],
                    'train_samples': len(train_features),
                    'test_samples': len(test_features),
                    'reliable_points': len(reliable_end_points) + len(reliable_turning_points)
                }
            
            # Train and evaluate Neural Network
            result_nn = None
            if model_type in ['nn', 'both']:
                nn_model = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10
                )
                nn_model.fit(train_features_scaled, train_labels)
                
                train_pred_nn = nn_model.predict(train_features_scaled)
                test_pred_nn = nn_model.predict(test_features_scaled)
                
                train_acc_nn = accuracy_score(train_labels, train_pred_nn)
                test_acc_nn = accuracy_score(test_labels, test_pred_nn)
                
                result_nn = {
                    'digit': digit,
                    'run_id': run_id,
                    'model': 'NN',
                    'train_accuracy': train_acc_nn,
                    'test_accuracy': test_acc_nn,
                    'feature_dim': train_features.shape[1],
                    'train_samples': len(train_features),
                    'test_samples': len(test_features),
                    'reliable_points': len(reliable_end_points) + len(reliable_turning_points)
                }
            
            return result_svm, result_nn
            
        except Exception as e:
            print(f"Unexpected error in run {run_id} for digit {digit}: {e}")
            return None, None
    
    def _extract_features_batch(self, image_paths, labels, reliable_end_points, reliable_turning_points):
        """Extract features from a batch of images"""
        features = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                feature_vec, _ = self.feature_extractor.extract_selected_features(
                    img_path, reliable_end_points, reliable_turning_points
                )
                if len(feature_vec) > 0:
                    features.append(feature_vec)
                    valid_labels.append(label)
            except Exception as e:
                continue
        
        return features, valid_labels
    
    def run_multiple_experiments(self, digits=range(1, 10), num_runs=30):
        """Run multiple experiments for all digits comparing SVM vs NN"""
        print(f"Starting {num_runs} random sampling experiments for digits {list(digits)}...")
        print("Comparing SVM vs Neural Network")
        print("Features: Basic + Second Moments + Erosion + Coordinate + Week7")
        
        all_results_svm = []
        all_results_nn = []
        
        for digit in digits:
            print(f"\nProcessing digit {digit}...")
            digit_results_svm = []
            digit_results_nn = []
            
            for run_id in range(num_runs):
                print(f"  Run {run_id + 1}/{num_runs}", end=' ')
                
                result_svm, result_nn = self.train_and_evaluate_single_run(digit, run_id)
                
                if result_svm and result_nn:
                    digit_results_svm.append(result_svm)
                    digit_results_nn.append(result_nn)
                    all_results_svm.append(result_svm)
                    all_results_nn.append(result_nn)
                    print(f"- SVM: {result_svm['test_accuracy']:.3f}, NN: {result_nn['test_accuracy']:.3f}")
                else:
                    print("- Failed")
            
            if digit_results_svm and digit_results_nn:
                self.results_svm[digit] = digit_results_svm
                self.results_nn[digit] = digit_results_nn
                
                # Print summary for this digit
                test_accs_svm = [r['test_accuracy'] for r in digit_results_svm]
                test_accs_nn = [r['test_accuracy'] for r in digit_results_nn]
                
                print(f"  Digit {digit} Summary:")
                print(f"    Successful runs: {len(digit_results_svm)}/{num_runs}")
                print(f"    SVM  - Test accuracy: {np.mean(test_accs_svm):.3f} ± {np.std(test_accs_svm):.3f}")
                print(f"    NN   - Test accuracy: {np.mean(test_accs_nn):.3f} ± {np.std(test_accs_nn):.3f}")
                print(f"    Improvement: {np.mean(test_accs_nn) - np.mean(test_accs_svm):+.3f}")
        
        return all_results_svm, all_results_nn

def create_comparison_visualization(results_svm_dict, results_nn_dict):
    """Create SVM vs Neural Network comparison visualization"""
    
    digits = sorted(results_svm_dict.keys())
    
    # Prepare data
    svm_stats = []
    nn_stats = []
    
    for digit in digits:
        svm_results = results_svm_dict[digit]
        nn_results = results_nn_dict[digit]
        
        if svm_results and nn_results:
            svm_test_accs = [r['test_accuracy'] * 100 for r in svm_results]
            nn_test_accs = [r['test_accuracy'] * 100 for r in nn_results]
            
            svm_stats.append({
                'digit': digit,
                'mean': np.mean(svm_test_accs),
                'std': np.std(svm_test_accs),
                'accs': svm_test_accs
            })
            
            nn_stats.append({
                'digit': digit,
                'mean': np.mean(nn_test_accs),
                'std': np.std(nn_test_accs),
                'accs': nn_test_accs
            })
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean accuracy comparison
    ax1 = axes[0, 0]
    x = np.arange(len(digits))
    width = 0.35
    
    svm_means = [s['mean'] for s in svm_stats]
    nn_means = [s['mean'] for s in nn_stats]
    svm_stds = [s['std'] for s in svm_stats]
    nn_stds = [s['std'] for s in nn_stats]
    
    bars1 = ax1.bar(x - width/2, svm_means, width, yerr=svm_stds, 
                    label='SVM', alpha=0.8, color='skyblue', capsize=3)
    bars2 = ax1.bar(x + width/2, nn_means, width, yerr=nn_stds,
                    label='Neural Network', alpha=0.8, color='lightcoral', capsize=3)
    
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('SVM vs Neural Network - Mean Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{d}' for d in digits])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement analysis
    ax2 = axes[0, 1]
    improvements = [nn_means[i] - svm_means[i] for i in range(len(digits))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(digits, improvements, alpha=0.7, color=colors)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Accuracy Improvement (NN - SVM) %')
    ax2.set_title('Neural Network Improvement over SVM')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.1 if imp >= 0 else bar.get_height() - 0.3,
                f'{imp:+.1f}%', ha='center', 
                va='bottom' if imp >= 0 else 'top', fontsize=9)
    
    # 3. Box plot comparison
    ax3 = axes[0, 2]
    svm_data = [s['accs'] for s in svm_stats]
    nn_data = [s['accs'] for s in nn_stats]
    
    bp1 = ax3.boxplot(svm_data, positions=np.arange(len(digits)) - 0.2, 
                      widths=0.3, patch_artist=True, 
                      boxprops=dict(facecolor='skyblue', alpha=0.7))
    bp2 = ax3.boxplot(nn_data, positions=np.arange(len(digits)) + 0.2,
                      widths=0.3, patch_artist=True,
                      boxprops=dict(facecolor='lightcoral', alpha=0.7))
    
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('Accuracy Distribution Comparison')
    ax3.set_xticks(range(len(digits)))
    ax3.set_xticklabels([f'{d}' for d in digits])
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot: SVM vs NN
    ax4 = axes[1, 0]
    ax4.scatter(svm_means, nn_means, c=digits, cmap='tab10', s=100, alpha=0.7)
    
    # Add diagonal line
    min_acc = min(min(svm_means), min(nn_means))
    max_acc = max(max(svm_means), max(nn_means))
    ax4.plot([min_acc, max_acc], [min_acc, max_acc], 'k--', alpha=0.5, label='Equal Performance')
    
    ax4.set_xlabel('SVM Test Accuracy (%)')
    ax4.set_ylabel('Neural Network Test Accuracy (%)')
    ax4.set_title('SVM vs NN Performance Scatter Plot')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add digit labels
    for i, digit in enumerate(digits):
        ax4.annotate(f'{digit}', (svm_means[i], nn_means[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # 5. Statistical summary table
    ax5 = axes[1, 1]
    table_data = []
    for i, digit in enumerate(digits):
        table_data.append([
            f'Digit {digit}',
            f'{svm_means[i]:.1f}%',
            f'{nn_means[i]:.1f}%',
            f'{improvements[i]:+.1f}%'
        ])
    
    # Add overall average
    overall_svm = np.mean(svm_means)
    overall_nn = np.mean(nn_means)
    overall_imp = overall_nn - overall_svm
    table_data.append([
        'Overall',
        f'{overall_svm:.1f}%',
        f'{overall_nn:.1f}%',
        f'{overall_imp:+.1f}%'
    ])
    
    ax5.axis('tight')
    ax5.axis('off')
    table = ax5.table(cellText=table_data,
                     colLabels=['Digit', 'SVM', 'NN', 'Improvement'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax5.set_title('Performance Summary', fontsize=12, pad=20)
    
    # 6. Win/Loss analysis
    ax6 = axes[1, 2]
    wins = sum(1 for imp in improvements if imp > 0)
    losses = sum(1 for imp in improvements if imp < 0)
    ties = sum(1 for imp in improvements if imp == 0)
    
    labels = ['NN Wins', 'SVM Wins', 'Ties']
    sizes = [wins, losses, ties]
    colors = ['lightcoral', 'skyblue', 'lightgray']
    
    ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
    ax6.set_title('Win/Loss Analysis\n(NN vs SVM)')
    
    plt.tight_layout()
    plt.suptitle('Random Sampling: SVM vs Neural Network Comparison (30 Runs per Digit)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('random_sampling_svm_vs_nn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_detailed_comparison_analysis(results_svm_dict, results_nn_dict):
    """Print detailed comparison analysis"""
    print("\n" + "="*80)
    print("DETAILED SVM vs NEURAL NETWORK COMPARISON ANALYSIS")
    print("="*80)
    
    digits = sorted(results_svm_dict.keys())
    
    print(f"\n{'Digit':<6} {'SVM Mean':<10} {'NN Mean':<10} {'Improvement':<12} {'SVM Std':<10} {'NN Std':<10}")
    print("-" * 70)
    
    total_svm = []
    total_nn = []
    improvements = []
    
    for digit in digits:
        svm_results = results_svm_dict[digit]
        nn_results = results_nn_dict[digit]
        
        if svm_results and nn_results:
            svm_accs = [r['test_accuracy'] * 100 for r in svm_results]
            nn_accs = [r['test_accuracy'] * 100 for r in nn_results]
            
            svm_mean = np.mean(svm_accs)
            nn_mean = np.mean(nn_accs)
            svm_std = np.std(svm_accs)
            nn_std = np.std(nn_accs)
            improvement = nn_mean - svm_mean
            
            total_svm.extend(svm_accs)
            total_nn.extend(nn_accs)
            improvements.append(improvement)
            
            print(f"{digit:<6} {svm_mean:<10.2f} {nn_mean:<10.2f} {improvement:<12.2f} "
                  f"{svm_std:<10.2f} {nn_std:<10.2f}")
    
    print("-" * 70)
    overall_svm = np.mean(total_svm)
    overall_nn = np.mean(total_nn)
    overall_improvement = overall_nn - overall_svm
    
    print(f"{'Overall':<6} {overall_svm:<10.2f} {overall_nn:<10.2f} {overall_improvement:<12.2f}")
    
    print(f"\nStatistical Summary:")
    print(f"  Average SVM accuracy: {overall_svm:.2f}%")
    print(f"  Average NN accuracy: {overall_nn:.2f}%")
    print(f"  Average improvement: {overall_improvement:+.2f}%")
    print(f"  Digits where NN > SVM: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    print(f"  Best improvement: {max(improvements):+.2f}% (Digit {digits[improvements.index(max(improvements))]})")
    print(f"  Worst performance: {min(improvements):+.2f}% (Digit {digits[improvements.index(min(improvements))]})")

def main():
    """Main function"""
    print("Random Sampling: SVM vs Neural Network Comparison")
    print("Features: Basic + Second Moments + Erosion + Coordinate + Week7")
    print("Sampling: 25 random images per category, 30 runs per digit")
    
    # Initialize feature extractor and comparison trainer
    feature_extractor = SelectedFeatureExtractor()
    comparison_trainer = RandomSamplingComparison(feature_extractor)
    
    # Run experiments
    start_time = time.time()
    all_results_svm, all_results_nn = comparison_trainer.run_multiple_experiments(
        digits=range(1, 10), num_runs=30
    )
    end_time = time.time()
    
    # Create visualizations and analysis
    if comparison_trainer.results_svm and comparison_trainer.results_nn:
        print("\nGenerating comparison analysis charts...")
        create_comparison_visualization(comparison_trainer.results_svm, comparison_trainer.results_nn)
        
        print_detailed_comparison_analysis(comparison_trainer.results_svm, comparison_trainer.results_nn)
        
        print(f"\nExperiment completed!")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Total SVM experiments: {len(all_results_svm)}")
        print(f"Total NN experiments: {len(all_results_nn)}")
        print(f"Generated chart: random_sampling_svm_vs_nn_comparison.png")
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    main()