# -*- coding: utf-8 -*-
# visualize_feature_results.py
# Main functions:
# 1. Visualize exhaustive feature selection results
# 2. Generate multiple charts analyzing feature importance and performance
# 3. Compare different feature combination performances

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import matplotlib.font_manager as fm

# Set font for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Set chart style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_exhaustive_results():
    """Load exhaustive search results (using simulated data, should load from actual program results)"""
    # Simulated data - should load from your program results in actual use
    np.random.seed(42)
    
    digits = list(range(1, 10))
    feature_names = ['Second Moments', 'Third Moments', 'Intensity', 'Erosion', 'Coordinate', 'Relative Position']
    
    # Simulate best feature combinations and accuracies for each digit
    results = []
    for digit in digits:
        # Randomly select 2-4 additional features
        n_features = np.random.randint(2, 5)
        selected_features = np.random.choice(feature_names, n_features, replace=False).tolist()
        
        # Generate reasonable accuracy based on number of selected features
        base_acc = 0.75 + len(selected_features) * 0.02 + np.random.normal(0, 0.05)
        test_acc = np.clip(base_acc, 0.6, 0.95)
        train_acc = test_acc + np.random.uniform(0.02, 0.08)
        
        feature_dim = 10 + len(selected_features) * 3 + (15 if 'Coordinate' in selected_features else 0) + (20 if 'Relative Position' in selected_features else 0)
        
        results.append({
            'digit': digit,
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'selected_features': selected_features,
            'feature_count': len(selected_features) + 1,  # +1 for basic features
            'feature_dim': feature_dim
        })
    
    return results

def create_accuracy_comparison_chart(results):
    """Create accuracy comparison charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    digits = [r['digit'] for r in results]
    train_accs = [r['train_accuracy'] * 100 for r in results]
    test_accs = [r['test_accuracy'] * 100 for r in results]
    
    x = np.arange(len(digits))
    width = 0.35
    
    # Accuracy comparison
    bars1 = ax1.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, test_accs, width, label='Testing Accuracy', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Best Feature Combination Accuracy Comparison by Digit')
    ax1.set_xticks(x)
    ax1.set_xticklabels(digits)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Overfitting analysis
    overfitting = [train_accs[i] - test_accs[i] for i in range(len(digits))]
    bars3 = ax2.bar(digits, overfitting, color='orange', alpha=0.7)
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Overfitting Degree (%)')
    ax2.set_title('Overfitting Analysis (Train Acc - Test Acc)')
    ax2.grid(True, alpha=0.3)
    
    # Add average line
    avg_overfitting = np.mean(overfitting)
    ax2.axhline(y=avg_overfitting, color='red', linestyle='--', 
                label=f'Average Overfitting: {avg_overfitting:.1f}%')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_chart(results):
    """Create feature importance charts"""
    feature_names = ['Second Moments', 'Third Moments', 'Intensity', 'Erosion', 'Coordinate', 'Relative Position']
    
    # Count how many times each feature is selected
    feature_counts = {name: 0 for name in feature_names}
    for result in results:
        for feature in result['selected_features']:
            if feature in feature_counts:
                feature_counts[feature] += 1
    
    # Calculate selection rates
    total_digits = len(results)
    feature_selection_rates = {name: count/total_digits*100 for name, count in feature_counts.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature selection frequency bar chart
    features = list(feature_selection_rates.keys())
    rates = list(feature_selection_rates.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(features)))
    
    bars = ax1.bar(features, rates, color=colors, alpha=0.8)
    ax1.set_ylabel('Selection Rate (%)')
    ax1.set_title('Feature Selection Frequency in Best Combinations')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Rotate x-axis labels
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Feature importance pie chart
    ax2.pie(rates, labels=features, autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Feature Importance Distribution')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_combination_heatmap(results):
    """Create feature combination heatmap"""
    feature_names = ['Second Moments', 'Third Moments', 'Intensity', 'Erosion', 'Coordinate', 'Relative Position']
    n_features = len(feature_names)
    
    # Create co-occurrence matrix
    cooccurrence_matrix = np.zeros((n_features, n_features))
    
    for result in results:
        selected = result['selected_features']
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if feat1 in selected and feat2 in selected:
                    cooccurrence_matrix[i][j] += 1
    
    # Convert to percentage
    total_digits = len(results)
    cooccurrence_percentage = cooccurrence_matrix / total_digits * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooccurrence_percentage, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                annot=True, 
                fmt='.1f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Co-occurrence Rate (%)'})
    
    plt.title('Feature Combination Co-occurrence Matrix\n(Values show percentage of features selected together)')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('feature_combination_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_vs_complexity_chart(results):
    """Create performance vs complexity analysis charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy vs number of features
    feature_counts = [r['feature_count'] for r in results]
    test_accs = [r['test_accuracy'] * 100 for r in results]
    digits = [r['digit'] for r in results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(digits)))
    scatter1 = ax1.scatter(feature_counts, test_accs, c=colors, s=100, alpha=0.7)
    
    # Add digit labels
    for i, digit in enumerate(digits):
        ax1.annotate(str(digit), (feature_counts[i], test_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Number of Feature Groups')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy vs Number of Feature Groups')
    ax1.grid(True, alpha=0.3)
    
    # Fit trend line
    z = np.polyfit(feature_counts, test_accs, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(feature_counts), max(feature_counts), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, 
             label=f'Trend Line (slope: {z[0]:.2f})')
    ax1.legend()
    
    # Accuracy vs feature dimensions
    feature_dims = [r['feature_dim'] for r in results]
    scatter2 = ax2.scatter(feature_dims, test_accs, c=colors, s=100, alpha=0.7)
    
    # Add digit labels
    for i, digit in enumerate(digits):
        ax2.annotate(str(digit), (feature_dims[i], test_accs[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('Feature Dimensions')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy vs Feature Dimensions')
    ax2.grid(True, alpha=0.3)
    
    # Fit trend line
    z2 = np.polyfit(feature_dims, test_accs, 1)
    p2 = np.poly1d(z2)
    x_trend2 = np.linspace(min(feature_dims), max(feature_dims), 100)
    ax2.plot(x_trend2, p2(x_trend2), "r--", alpha=0.8, 
             label=f'Trend Line (slope: {z2[0]:.4f})')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('performance_vs_complexity.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_top_combinations_chart(results):
    """Create top feature combinations chart"""
    # Count all feature combinations
    combinations = []
    for result in results:
        combo = tuple(sorted(result['selected_features']))
        combinations.append(combo)
    
    # Calculate combination frequency
    combo_counts = Counter(combinations)
    top_combinations = combo_counts.most_common(5)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Most popular combinations
    combo_names = []
    counts = []
    for combo, count in top_combinations:
        if combo:
            combo_name = ' + '.join(combo)
        else:
            combo_name = 'No Additional Features'
        combo_names.append(combo_name)
        counts.append(count)
    
    bars = ax1.barh(combo_names, counts, color='lightblue', alpha=0.8)
    ax1.set_xlabel('Selection Count')
    ax1.set_title('Most Popular Feature Combinations (Top 5)')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontsize=10)
    
    # Feature combination overview for each digit
    digits = [r['digit'] for r in results]
    all_features = ['Second Moments', 'Third Moments', 'Intensity', 'Erosion', 'Coordinate', 'Relative Position']
    
    # Create feature selection matrix
    selection_matrix = np.zeros((len(digits), len(all_features)))
    for i, result in enumerate(results):
        for j, feature in enumerate(all_features):
            if feature in result['selected_features']:
                selection_matrix[i][j] = 1
    
    im = ax2.imshow(selection_matrix, cmap='RdYlBu_r', aspect='auto')
    ax2.set_xticks(range(len(all_features)))
    ax2.set_xticklabels(all_features, rotation=45, ha='right')
    ax2.set_yticks(range(len(digits)))
    ax2.set_yticklabels([f'Digit {d}' for d in digits])
    ax2.set_title('Best Feature Combination Selection Matrix by Digit')
    ax2.set_xlabel('Feature Type')
    ax2.set_ylabel('Digit')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Selected', 'Selected'])
    
    # Add text annotations in matrix
    for i in range(len(digits)):
        for j in range(len(all_features)):
            text = '✓' if selection_matrix[i, j] == 1 else '✗'
            color = 'white' if selection_matrix[i, j] == 1 else 'black'
            ax2.text(j, i, text, ha='center', va='center', 
                    color=color, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('top_combinations.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_analysis(results):
    """Create improvement effect analysis"""
    # Assume baseline accuracy with basic features only
    baseline_accuracy = 72.5  # Average accuracy using only basic features
    
    improvements = []
    feature_counts = []
    
    for result in results:
        improvement = (result['test_accuracy'] * 100) - baseline_accuracy
        improvements.append(improvement)
        feature_counts.append(len(result['selected_features']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Improvement vs number of additional features
    digits = [r['digit'] for r in results]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(digits)))
    
    scatter = ax1.scatter(feature_counts, improvements, c=colors, s=100, alpha=0.7)
    
    # Add digit labels
    for i, digit in enumerate(digits):
        ax1.annotate(str(digit), (feature_counts[i], improvements[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('Number of Additional Features')
    ax1.set_ylabel('Accuracy Improvement (%)')
    ax1.set_title('Accuracy Improvement vs Number of Additional Features')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline')
    
    # Fit trend line
    z = np.polyfit(feature_counts, improvements, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(feature_counts), max(feature_counts), 100)
    ax1.plot(x_trend, p(x_trend), "g--", alpha=0.8, 
             label=f'Trend Line (improvement per feature: {z[0]:.2f}%)')
    ax1.legend()
    
    # Improvement distribution histogram
    ax2.hist(improvements, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Accuracy Improvement (%)')
    ax2.set_ylabel('Number of Digits')
    ax2.set_title('Accuracy Improvement Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistical information
    mean_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)
    ax2.axvline(x=mean_improvement, color='red', linestyle='-', 
                label=f'Mean: {mean_improvement:.1f}%')
    ax2.axvline(x=mean_improvement + std_improvement, color='orange', linestyle='--', 
                label=f'+1σ: {mean_improvement + std_improvement:.1f}%')
    ax2.axvline(x=mean_improvement - std_improvement, color='orange', linestyle='--', 
                label=f'-1σ: {mean_improvement - std_improvement:.1f}%')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_dashboard(results):
    """Create summary dashboard"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Overall accuracy statistics
    ax1 = fig.add_subplot(gs[0, 0])
    test_accs = [r['test_accuracy'] * 100 for r in results]
    train_accs = [r['train_accuracy'] * 100 for r in results]
    
    ax1.boxplot([test_accs, train_accs], labels=['Test', 'Train'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature usage statistics
    ax2 = fig.add_subplot(gs[0, 1])
    feature_counts = [len(r['selected_features']) for r in results]
    ax2.hist(feature_counts, bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Number of Additional Features')
    ax2.set_ylabel('Number of Digits')
    ax2.set_title('Feature Usage Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature dimension statistics
    ax3 = fig.add_subplot(gs[0, 2])
    feature_dims = [r['feature_dim'] for r in results]
    ax3.scatter(range(1, 10), feature_dims, c='red', s=60, alpha=0.7)
    ax3.set_xlabel('Digit')
    ax3.set_ylabel('Feature Dimensions')
    ax3.set_title('Feature Dimensions by Digit')
    ax3.grid(True, alpha=0.3)
    
    # 4. Top accuracy ranking
    ax4 = fig.add_subplot(gs[0, 3])
    sorted_results = sorted(results, key=lambda x: x['test_accuracy'], reverse=True)
    top_digits = [r['digit'] for r in sorted_results[:5]]
    top_accs = [r['test_accuracy'] * 100 for r in sorted_results[:5]]
    
    bars = ax4.bar(range(5), top_accs, color='gold', alpha=0.8)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels([f'Digit{d}' for d in top_digits])
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Top 5 Accuracy')
    
    for bar, acc in zip(bars, top_accs):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 5-8. Feature importance analysis (bottom two rows)
    feature_names = ['Second Moments', 'Third Moments', 'Intensity', 'Erosion', 'Coordinate', 'Relative Position']
    feature_counts = {name: 0 for name in feature_names}
    for result in results:
        for feature in result['selected_features']:
            if feature in feature_counts:
                feature_counts[feature] += 1
    
    ax5 = fig.add_subplot(gs[1:, :2])
    features = list(feature_counts.keys())
    counts = list(feature_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(features)))
    
    bars = ax5.bar(features, counts, color=colors, alpha=0.8)
    ax5.set_ylabel('Selection Count')
    ax5.set_title('Feature Selection Frequency Statistics', fontsize=14)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=11)
    
    # 6. Accuracy vs feature count relationship
    ax6 = fig.add_subplot(gs[1:, 2:])
    digits = [r['digit'] for r in results]
    feature_counts = [len(r['selected_features']) for r in results]
    test_accs = [r['test_accuracy'] * 100 for r in results]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(digits)))
    scatter = ax6.scatter(feature_counts, test_accs, c=colors, s=150, alpha=0.7)
    
    for i, digit in enumerate(digits):
        ax6.annotate(f'Digit{digit}', (feature_counts[i], test_accs[i]), 
                    xytext=(8, 8), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax6.set_xlabel('Number of Additional Features', fontsize=12)
    ax6.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax6.set_title('Accuracy vs Feature Count Relationship', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # Add statistics text box
    most_common_feature = max(feature_counts, key=feature_counts.get) if feature_counts else "None"
    stats_text = f"""
    Summary Statistics:
    • Average Test Accuracy: {np.mean(test_accs):.1f}%
    • Best Test Accuracy: {np.max(test_accs):.1f}%
    • Average Feature Count: {np.mean(feature_counts):.1f}
    • Most Selected Feature: {most_common_feature}
    """
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Exhaustive Feature Selection Results Summary Dashboard', fontsize=16, fontweight='bold')
    plt.savefig('summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("Generating exhaustive feature selection result visualization charts...")
    
    # Load results (using simulated data here, should load from your program results in practice)
    results = load_exhaustive_results()
    
    print("1. Generating accuracy comparison chart...")
    create_accuracy_comparison_chart(results)
    
    print("2. Generating feature importance chart...")
    create_feature_importance_chart(results)
    
    print("3. Generating feature combination heatmap...")
    create_feature_combination_heatmap(results)
    
    print("4. Generating performance vs complexity chart...")
    create_performance_vs_complexity_chart(results)
    
    print("5. Generating top combinations ranking...")
    create_top_combinations_chart(results)
    
    print("6. Generating improvement analysis...")
    create_improvement_analysis(results)
    
    print("7. Generating summary dashboard...")
    create_summary_dashboard(results)
    
    print("\nAll charts generated successfully!")
    print("Generated image files:")
    print("- accuracy_comparison.png: Accuracy comparison")
    print("- feature_importance.png: Feature importance")
    print("- feature_combination_heatmap.png: Feature combination heatmap")
    print("- performance_vs_complexity.png: Performance vs complexity")
    print("- top_combinations.png: Top combinations ranking")
    print("- improvement_analysis.png: Improvement analysis")
    print("- summary_dashboard.png: Summary dashboard")

if __name__ == "__main__":
    main()