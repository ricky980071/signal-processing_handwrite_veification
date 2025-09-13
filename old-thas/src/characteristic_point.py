from edge_indexing import edge_indexing
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def calculate_vector_angles(pixels, d=12):
    """
    計算像素序列中每個點的特徵角度
    
    對於每個像素 P_i，計算向量 P_iP_{i+d} 和 P_iP_{i-d} 之間的夾角
    
    Args:
        pixels: 像素序列 [(y1, x1), (y2, x2), ..., (yn, xn)]
        d: 向量距離參數，默認為12
    
    Returns:
        angles: 角度數組 [theta_1, theta_2, ..., theta_n]（以弧度為單位）
    """
    n = len(pixels)
    if n == 0:
        return []
    
    # 如果像素數量太少，使用較小的d值
    effective_d = min(d, n // 2) if n > 2 else 1
    
    angles = []
    
    for i in range(n):
        # 使用模運算處理邊界情況
        idx_forward = (i + effective_d) % n  # P_{i+d}
        idx_backward = (i - effective_d) % n  # P_{i-d}
        
        # 當前點
        pi = pixels[i]
        # 前向點和後向點
        p_forward = pixels[idx_forward]
        p_backward = pixels[idx_backward]
        
        # 計算兩個向量
        # 向量 P_iP_{i+d}
        vector_forward = (p_forward[0] - pi[0], p_forward[1] - pi[1])
        # 向量 P_iP_{i-d}
        vector_backward = (p_backward[0] - pi[0], p_backward[1] - pi[1])
        
        # 計算向量長度
        len_forward = math.sqrt(vector_forward[0]**2 + vector_forward[1]**2)
        len_backward = math.sqrt(vector_backward[0]**2 + vector_backward[1]**2)
        
        # 避免除零錯誤
        if len_forward == 0 or len_backward == 0:
            angles.append(0.0)
            continue
        
        # 計算點積
        dot_product = vector_forward[0] * vector_backward[0] + vector_forward[1] * vector_backward[1]
        
        # 計算夾角（使用餘弦定理）
        cos_theta = dot_product / (len_forward * len_backward)
        
        # 確保餘弦值在有效範圍內（避免數值誤差）
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        # 計算角度（弧度）
        theta = math.acos(cos_theta)
        angles.append(theta)

        # Debug for the first pixel
        # if i == 0:
        #     print(f"Processing pixel {i}: P_i = {pixels[i]}, P_i+d = {pixels[idx_forward]}, P_i-d = {pixels[idx_backward]}")
        #     print(f"Indices: Forward = {idx_forward}, Backward = {idx_backward}")
        #     print(f"Vector Forward: {vector_forward}, Vector Backward: {vector_backward}")
        #     print(f"Dot Product: {dot_product}, Length Forward: {len_forward}, Length Backward: {len_backward}")
        #     print(f"Cosine Theta: {cos_theta}, Angle (radians): {theta}\n")
    
    return angles

def process_component_angles(component_data, d=12):
    """
    處理單個組件的角度計算
    
    Args:
        component_data: 包含 'traced_pixels' 的組件數據
        d: 向量距離參數
    
    Returns:
        angles: 該組件的角度數組
    """
    traced_pixels = component_data['traced_pixels']
    return calculate_vector_angles(traced_pixels, d)

def process_all_components_angles(indexed_components, d=12):
    """
    處理所有組件的角度計算
    
    Args:
        indexed_components: edge_indexing 函數返回的組件列表
        d: 向量距離參數
    
    Returns:
        results: 每個組件的角度計算結果
    """
    results = []
    
    for i, component_data in enumerate(indexed_components):
        angles = process_component_angles(component_data, d)
        
        result = {
            'component_id': component_data['component_id'],
            'pixel_count': len(component_data['pixels']),
            'traced_pixel_count': len(component_data['traced_pixels']),
            'traced_pixels': component_data['traced_pixels'],
            'angles': angles
        }
        results.append(result)
    
    return results

def classify_angles(angles_degrees):
    """
    將角度分類為三種情況
    
    Args:
        angles_degrees: 角度陣列（以度為單位）
    
    Returns:
        classifications: 分類結果陣列 ['case1', 'case2', 'case3', ...]
    """
    classifications = []
    for angle in angles_degrees:
        if angle < 30:
            classifications.append('case1')
        elif 30 <= angle <= 150:
            classifications.append('case2')
        else:  # angle > 150
            classifications.append('case3')
    return classifications

def find_characteristic_points(angles_degrees):
    """
    找出特徵點：ending points 和 turning points
    
    Args:
        angles_degrees: 角度陣列（以度為單位）
    
    Returns:
        dict: 包含 ending_points 和 turning_points 的字典
    """
    n = len(angles_degrees)
    if n == 0:
        return {'ending_points': [], 'turning_points': []}
    
    # 分類角度
    classifications = classify_angles(angles_degrees)
    
    ending_points = []
    turning_points = []
    
    # 找出連續的相同分類區段
    segments = []
    current_segment = {'type': classifications[0], 'start': 0, 'indices': [0]}
    
    for i in range(1, n):
        if classifications[i] == current_segment['type']:
            current_segment['indices'].append(i)
        else:
            current_segment['end'] = current_segment['indices'][-1]
            segments.append(current_segment)
            current_segment = {'type': classifications[i], 'start': i, 'indices': [i]}
    
    # 添加最後一個區段
    current_segment['end'] = current_segment['indices'][-1]
    segments.append(current_segment)
    
    # 處理頭尾連接的情況（環形結構）
    if len(segments) > 1 and segments[0]['type'] == segments[-1]['type']:
        # 合併第一個和最後一個區段
        merged_indices = segments[-1]['indices'] + segments[0]['indices']
        merged_segment = {
            'type': segments[0]['type'],
            'start': segments[-1]['start'],
            'end': segments[0]['end'],
            'indices': merged_indices
        }
        segments = [merged_segment] + segments[1:-1]
    
    # 找出每個區段的最小角度點
    for segment in segments:
        if len(segment['indices']) > 0:
            # 找出該區段中角度最小的點
            min_angle = float('inf')
            min_index = -1
            
            for idx in segment['indices']:
                if angles_degrees[idx] < min_angle:
                    min_angle = angles_degrees[idx]
                    min_index = idx
            
            # 根據區段類型分類特徵點
            if segment['type'] == 'case1':
                ending_points.append({
                    'index': min_index,
                    'angle': min_angle,
                    'segment_indices': segment['indices']
                })
            elif segment['type'] == 'case2':
                turning_points.append({
                    'index': min_index,
                    'angle': min_angle,
                    'segment_indices': segment['indices']
                })
    
    return {
        'ending_points': ending_points,
        'turning_points': turning_points,
        'classifications': classifications,
        'segments': segments
    }

def analyze_component_characteristic_points(angles_degrees):
    """
    分析組件的特徵點並提供詳細報告
    
    Args:
        angles_degrees: 角度陣列（以度為單位）
    
    Returns:
        dict: 分析結果
    """
    characteristic_points = find_characteristic_points(angles_degrees)
    
    # 統計各類別的數量
    classifications = characteristic_points['classifications']
    case1_count = classifications.count('case1')
    case2_count = classifications.count('case2')
    case3_count = classifications.count('case3')
    
    analysis = {
        'total_points': len(angles_degrees),
        'case1_count': case1_count,
        'case2_count': case2_count,
        'case3_count': case3_count,
        'ending_points': characteristic_points['ending_points'],
        'turning_points': characteristic_points['turning_points'],
        'classifications': classifications,
        'segments': characteristic_points['segments']
    }
    
    return analysis

def visualize_characteristic_points(edge_image, indexed_components, angle_results, save_path=None):
    """
    視覺化特徵點在圖像上的位置
    
    Args:
        edge_image: 邊緣圖像
        indexed_components: 組件數據
        angle_results: 角度分析結果
        save_path: 保存路徑（可選）
    """
    num_subplots = len(indexed_components) + 1
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))
    
    # 確保 axes 總是一個列表
    if num_subplots == 1:
        axes = [axes]
    elif hasattr(axes, '__len__') and not isinstance(axes, list):
        axes = list(axes)
    
    # 顯示原始邊緣圖像
    axes[0].imshow(edge_image, cmap='gray')
    axes[0].set_title('Original Edge Image')
    axes[0].axis('off')
    
    # 為每個組件創建單獨的視覺化（使用原始邊緣圖像作為背景）
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    
    for comp_idx, (component_data, angle_result) in enumerate(zip(indexed_components, angle_results)):
        if comp_idx + 1 < len(axes):
            ax = axes[comp_idx + 1]
        else:
            break
            
        # 使用原始邊緣圖像作為背景
        ax.imshow(edge_image, cmap='gray')
        ax.set_title(f'Component {component_data["component_id"]} Characteristic Points')
        ax.axis('off')
        
        # 獲取追蹤像素
        traced_pixels = component_data['traced_pixels']
        
        # 分析特徵點
        angles_degrees = [math.degrees(angle) for angle in angle_result['angles']]
        analysis = analyze_component_characteristic_points(angles_degrees)
        
        # 標示 Ending Points (藍色實心圓點)
        for i, ep in enumerate(analysis['ending_points']):
            pixel_idx = ep['index']
            if pixel_idx < len(traced_pixels):
                y, x = traced_pixels[pixel_idx]
                circle = Circle((x, y), radius=2, color='blue', fill=True, linewidth=0)
                ax.add_patch(circle)
        
        # 標示 Turning Points (紅色實心圓點)
        for i, tp in enumerate(analysis['turning_points']):
            pixel_idx = tp['index']
            if pixel_idx < len(traced_pixels):
                y, x = traced_pixels[pixel_idx]
                circle = Circle((x, y), radius=2, color='red', fill=True, linewidth=0)
                ax.add_patch(circle)
    
    # 創建完整圖像視覺化
    if len(axes) > len(indexed_components) + 1:
        # 如果有額外的軸，創建一個完整的視覺化
        ax_full = axes[-1]
        
        # 創建彩色顯示所有組件
        height, width = edge_image.shape
        full_display = np.zeros((height, width, 3))
        
        for comp_idx, (component_data, angle_result) in enumerate(zip(indexed_components, angle_results)):
            traced_pixels = component_data['traced_pixels']
            component_color = colors[comp_idx % len(colors)]
            
            for y, x in traced_pixels:
                if component_color == 'red':
                    full_display[y, x] = [1, 0, 0]
                elif component_color == 'green':
                    full_display[y, x] = [0, 1, 0]
                elif component_color == 'blue':
                    full_display[y, x] = [0, 0, 1]
                elif component_color == 'yellow':
                    full_display[y, x] = [1, 1, 0]
                elif component_color == 'magenta':
                    full_display[y, x] = [1, 0, 1]
                elif component_color == 'cyan':
                    full_display[y, x] = [0, 1, 1]
            
            # 標示特徵點
            angles_degrees = [math.degrees(angle) for angle in angle_result['angles']]
            analysis = analyze_component_characteristic_points(angles_degrees)
            
            # Ending Points (藍色實心圓點)
            for i, ep in enumerate(analysis['ending_points']):
                pixel_idx = ep['index']
                if pixel_idx < len(traced_pixels):
                    y, x = traced_pixels[pixel_idx]
                    circle = Circle((x, y), radius=2, color='blue', fill=True, linewidth=0)
                    ax_full.add_patch(circle)
            
            # Turning Points (紅色實心圓點)
            for i, tp in enumerate(analysis['turning_points']):
                pixel_idx = tp['index']
                if pixel_idx < len(traced_pixels):
                    y, x = traced_pixels[pixel_idx]
                    circle = Circle((x, y), radius=2, color='red', fill=True, linewidth=0)
                    ax_full.add_patch(circle)
        
        ax_full.imshow(full_display)
        ax_full.set_title('All Components with Characteristic Points')
        ax_full.axis('off')
        
        # 添加圖例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Ending Points'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Turning Points')
        ]
        ax_full.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"圖像已保存至: {save_path}")
    
    plt.show()
    return fig

if __name__ == "__main__":
    # 測試用例
    # test_image_path = '../data/1/database/base_1_1_1.bmp'
    # test_image_path = '../data/2/database/base_1_1_2.bmp'
    # test_image_path = '../data/3/database/base_1_1_3.bmp'
    # test_image_path = '../data/4/database/base_1_1_4.bmp'
    test_image_path = '../data/5/database/base_1_1_5.bmp'
    # test_image_path = '../data/6/database/base_1_1_6.bmp'
    # test_image_path = '../data/7/database/base_1_1_7.bmp'
    # test_image_path = '../data/8/database/base_1_1_8.bmp'
    # test_image_path = '../data/9/database/base_1_1_9.bmp'
    
    print("正在處理圖像...")
    print(f"圖像路徑: {test_image_path}")
    
    # 獲取邊緣索引結果
    edge_result, indexed_components = edge_indexing(test_image_path)
    
    print(f"\n找到 {len(indexed_components)} 個組件")
    
    # 計算所有組件的角度
    d = 12  # 向量距離參數
    print(f"使用向量距離參數 d = {d}")
    
    angle_results = process_all_components_angles(indexed_components, d)
    
    # 顯示結果
    for result in angle_results:
        comp_id = result['component_id']
        pixel_count = result['pixel_count']
        traced_count = result['traced_pixel_count']
        angles = result['angles']
        
        print(f"\n組件 {comp_id}:")
        print(f"  總像素數: {pixel_count}")
        print(f"  追蹤像素數: {traced_count}")
        print(f"  角度數組長度: {len(angles)}")
        
        if len(angles) > 0:
            angles_degrees = [math.degrees(angle) for angle in angles]
            
            # 顯示所有角度
            # print(f"  所有角度 (度): {', '.join(f'{angle:.2f}' for angle in angles_degrees)}")
            
            # 分析特徵點
            analysis = analyze_component_characteristic_points(angles_degrees)
            
            print(f"  角度分類統計:")
            print(f"    Case1 (<30°): {analysis['case1_count']} 個點")
            print(f"    Case2 (30°-150°): {analysis['case2_count']} 個點")
            print(f"    Case3 (>150°): {analysis['case3_count']} 個點")
            
            print(f"  特徵點分析:")
            print(f"    Ending Points (Case1區段最小值): {len(analysis['ending_points'])} 個")
            for i, ep in enumerate(analysis['ending_points']):
                print(f"      EP{i+1}: 索引{ep['index']}, 角度{ep['angle']:.2f}°, 區段{ep['segment_indices']}")
            
            print(f"    Turning Points (Case2區段最小值): {len(analysis['turning_points'])} 個")
            for i, tp in enumerate(analysis['turning_points']):
                print(f"      TP{i+1}: 索引{tp['index']}, 角度{tp['angle']:.2f}°, 區段{tp['segment_indices']}")
            
            # 角度統計信息
            print(f"  角度統計:")
            print(f"    最小角度: {min(angles_degrees):.2f}°")
            print(f"    最大角度: {max(angles_degrees):.2f}°")
            print(f"    平均角度: {np.mean(angles_degrees):.2f}°")
            print(f"    標準差: {np.std(angles_degrees):.2f}°")
    
    print(f"\n角度計算完成！")
    print(f"每個組件都產生了一個長度等於其追蹤像素數的角度數組。")
    
    # 視覺化特徵點
    print(f"\n正在生成特徵點視覺化圖像...")
    fig = visualize_characteristic_points(edge_result, indexed_components, angle_results)
    
    print("視覺化完成！")
    print("圖例說明：")
    print("- 藍色圓點：Ending Points (Case1區段最小值)")
    print("- 紅色圓點：Turning Points (Case2區段最小值)")
