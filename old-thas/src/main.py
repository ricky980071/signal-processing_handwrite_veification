import os
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import glob
import csv
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

Y_STROKE_MAX = 220

def stroke_extraction(image):
    def is_stroke(pixel):
        return pixel < Y_STROKE_MAX
    if image.mode != 'YCbCr':
        image = image.convert('YCbCr')  # Ensure the image is in 'YCbCr' mode
    y_channel, _, _ = image.split()  # Extract the Y channel
    y_array = np.array(y_channel)
    stroke_array = np.vectorize(is_stroke)(y_array)
    stroke_image = Image.fromarray(stroke_array.astype(np.uint8) * 255, mode='L')  # Convert back to image
    return stroke_image, y_array

def binary_array_extraction(image):
    stroke_image, _ = stroke_extraction(image)
    stroke_array = np.array(stroke_image)
    return (stroke_array > 0).astype(np.float32)

# B_0 is the original binary array (stroke pixels)
def compute_B_k_list(binary_array):
    # print(binary_array)
    height, width = binary_array.shape
    B_0 = binary_array.copy()
    B_k_list = [B_0]
    for k in range(1, 4):
        B_prev = B_k_list[k-1]
        B_k = np.zeros_like(B_prev)
        for m in range(width):
            for n in range(height):
                if B_prev[n, m] == 0:
                    continue
                self_ok = (B_prev[n, m] > 0)
                top_ok = (n > 0 and B_prev[n-1, m] > 0)
                bottom_ok = (n < height-1 and B_prev[n+1, m] > 0)
                left_ok = (m > 0 and B_prev[n, m-1] > 0)
                right_ok = (m < width-1 and B_prev[n, m+1] > 0)
                if self_ok and top_ok and bottom_ok and left_ok and right_ok:
                    B_k[n, m] = 1
        B_k_list.append(B_k)
    return B_k_list

def feature_extraction(image):
    stroke_image, y_array = stroke_extraction(image)
    stroke_array = np.array(stroke_image)
    height, width = stroke_array.shape

    # Divide the image into 5 rows and 5 columns
    row_height = height // 5 + 1
    col_width = width // 5 + 1

    # Initialize features
    # f1 to f5: # of stroke pixels for columns
    # f6 to f10: # of stroke pixels for rows
    # f11: min intensity of stroke pixels
    # f12: standard deviation of the intensity of stroke pixels
    # f13: m_0 (weighted average of horizontal position of stroke pixels)
    # f14: n_0 (weighted average of vertical position of stroke pixels)
    # f15: v_{2,0} (second order moment)
    # f16: v_{1,1} (second order moment)
    # f17: v_{0,2} (second order moment)
    # f18: v_{3,0} (third order moment)
    # f19: v_{2,1} (third order moment)
    # f20: v_{1,2} (third order moment)
    # f21: v_{0,3} (third order moment)
    # f22: r_1 (ratio of B_1 to B_0)
    # f23: r_2 (ratio of B_2 to B_0)
    # f24: r_3 (ratio of B_3 to B_0)
    # f25: w (width/height ratio of stroke region)
    features = {}

    # Calculate stroke pixels for each column (f1 to f5)
    for col in range(5):
        col_start = col * col_width
        col_end = (col + 1) * col_width if col < 4 else width
        features[f"f{col + 1}"] = np.sum(stroke_array[:, col_start:col_end] > 0)

    # Calculate stroke pixels for each row (f6 to f10)
    for row in range(5):
        row_start = row * row_height
        row_end = (row + 1) * row_height if row < 4 else height
        features[f"f{row + 6}"] = np.sum(stroke_array[row_start:row_end, :] > 0)

    # Calculate min intensity of stroke pixels (f11)
    y_stroke_pixels = y_array[(stroke_array > 0) & (y_array < Y_STROKE_MAX)]
    features["f11"] = np.min(y_stroke_pixels) if y_stroke_pixels.size > 0 else 0

    # Calculate standard deviation of the intensity of stroke pixels (f12)
    features["f12"] = np.std(y_stroke_pixels) if y_stroke_pixels.size > 0 else 0
    
    # Calculate f13 (m_0) and f14 (n_0): weighted average of positions of stroke pixels
    # B(m,n) is 1 for stroke pixels and 0 for background
    binary_array = binary_array_extraction(stroke_image)

    # Create coordinate arrays
    m_coords, n_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    # Calculate sum(B(m,n))
    sum_B = np.sum(binary_array)
    
    # Skip calculations if there are no stroke pixels
    if sum_B == 0:
        for i in range(13, 26):  # f13 to f25
            features[f"f{i}"] = 0
    else:
        # Calculate sum(m * B(m,n)) and sum(n * B(m,n))
        sum_m_times_B = np.sum(m_coords * binary_array)
        sum_n_times_B = np.sum(n_coords * binary_array)
        
        # m_0 = sum(m * B(m,n)) / sum(B(m,n))
        m0 = sum_m_times_B / sum_B
        features["f13"] = m0
        
        # n_0 = sum(n * B(m,n)) / sum(B(m,n))
        n0 = sum_n_times_B / sum_B
        features["f14"] = n0
        
        # Calculate moment-based features (f15 to f21)
        # Create arrays for powers of (m-m0) and (n-n0)
        m_minus_m0 = m_coords - m0
        n_minus_n0 = n_coords - n0
        
        # Second order moments
        # v_{2,0} = sum((m-m0)^2 * B(m,n)) / sum(B(m,n))
        features["f15"] = np.sum(np.power(m_minus_m0, 2) * binary_array) / sum_B
        
        # v_{1,1} = sum((m-m0) * (n-n0) * B(m,n)) / sum(B(m,n))
        features["f16"] = np.sum(m_minus_m0 * n_minus_n0 * binary_array) / sum_B
        
        # v_{0,2} = sum((n-n0)^2 * B(m,n)) / sum(B(m,n))
        features["f17"] = np.sum(np.power(n_minus_n0, 2) * binary_array) / sum_B
        
        # Third order moments
        # v_{3,0} = sum((m-m0)^3 * B(m,n)) / sum(B(m,n))
        features["f18"] = np.sum(np.power(m_minus_m0, 3) * binary_array) / sum_B
        
        # v_{2,1} = sum((m-m0)^2 * (n-n0) * B(m,n)) / sum(B(m,n))
        features["f19"] = np.sum(np.power(m_minus_m0, 2) * n_minus_n0 * binary_array) / sum_B
        
        # v_{1,2} = sum((m-m0) * (n-n0)^2 * B(m,n)) / sum(B(m,n))
        features["f20"] = np.sum(m_minus_m0 * np.power(n_minus_n0, 2) * binary_array) / sum_B
        
        # v_{0,3} = sum((n-n0)^3 * B(m,n)) / sum(B(m,n))
        features["f21"] = np.sum(np.power(n_minus_n0, 3) * binary_array) / sum_B
        
        # Calculate B_k features (f22 to f24)
        B_k_list = compute_B_k_list(binary_array)
        
        # Calculate ratio features r_k = sum(B_k) / sum(B_0)
        sum_B0 = np.sum(B_k_list[0])
        for k in range(1, 4):
            sum_Bk = np.sum(B_k_list[k])
            features[f"f{21+k}"] = sum_Bk / sum_B0 if sum_B0 > 0 else 0
        
        # Calculate width/height ratio feature (f25)
        # Find stroke pixel region boundaries
        stroke_indices = np.where(binary_array > 0)
        if len(stroke_indices[0]) > 0:
            n_min = np.min(stroke_indices[0])  # Vertical (row) min
            n_max = np.max(stroke_indices[0])  # Vertical (row) max
            m_min = np.min(stroke_indices[1])  # Horizontal (col) min
            m_max = np.max(stroke_indices[1])  # Horizontal (col) max
            
            # Calculate w = (m_max - m_min) / (n_max - n_min)
            height_diff = n_max - n_min
            width_diff = m_max - m_min
            features["f25"] = width_diff / height_diff if height_diff > 0 else 0
        else:
            features["f25"] = 0

    # Convert dictionary to feature vector
    feature_vector = [features[f"f{i}"] for i in range(1, 26)]  # f1 to f25

    edge_image = np.logical_xor(B_k_list[0], B_k_list[1])  # XOR of B_0 and B_1

    # return feature_vector
    return feature_vector, edge_image

def export_features_to_csv(data_dir, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        header = ['filename'] + [f"f{i}" for i in range(1, 26)]  # f1 to f25
        writer.writerow(header)

        # Process all images in data_dir and all of its subdirectories
        for subdir, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.bmp'):
                    image_path = os.path.join(subdir, file)
                    image = Image.open(image_path)
                    features, _ = feature_extraction(image)
                    writer.writerow([file] + features)

    print(f"特徵已匯出至 {output_csv}")

def load_and_prepare_data(data_dir):
    """
    載入資料並準備訓練和測試資料集
    - 從database目錄中取前25張作為真實訓練集，後25張作為真實測試集
    - 從testcase目錄中取前25張作為偽造訓練集，後25張作為偽造測試集
    """
    # 初始化資料列表
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    
    # 處理真實樣本 (database)
    database_dir = os.path.join(data_dir, 'database')
    database_files = sorted([f for f in os.listdir(database_dir) if f.endswith('.bmp')])
    
    # 前25張作為訓練資料
    for file in database_files[:25]:
        image_path = os.path.join(database_dir, file)
        image = Image.open(image_path)
        features, _ = feature_extraction(image)
        train_features.append(features)
        train_labels.append(1)  # 1表示真實樣本
    
    # 後25張作為測試資料
    for file in database_files[25:50]:
        image_path = os.path.join(database_dir, file)
        image = Image.open(image_path)
        features, _ = feature_extraction(image)
        test_features.append(features)
        test_labels.append(1)  # 1表示真實樣本
    
    # 處理偽造樣本 (testcase)
    testcase_dir = os.path.join(data_dir, 'testcase')
    testcase_files = sorted([f for f in os.listdir(testcase_dir) if f.endswith('.bmp')])
    
    # 前25張作為訓練資料
    for file in testcase_files[:25]:
        image_path = os.path.join(testcase_dir, file)
        image = Image.open(image_path)
        features, _ = feature_extraction(image)
        train_features.append(features)
        train_labels.append(0)  # 0表示偽造樣本
    
    # 後25張作為測試資料
    for file in testcase_files[25:50]:
        image_path = os.path.join(testcase_dir, file)
        image = Image.open(image_path)
        features, _ = feature_extraction(image)
        test_features.append(features)
        test_labels.append(0)  # 0表示偽造樣本
    
    return np.array(train_features), np.array(train_labels), np.array(test_features), np.array(test_labels)

def normalize_features(train_features, test_features):
    """
    對特徵進行標準化處理：(x - mean) / std
    使用訓練數據的均值和標準差來標準化測試數據
    """
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_features)
    test_normalized = scaler.transform(test_features)
    
    return train_normalized, test_normalized

def train_svm_classifier(train_features, train_labels):
    """
    訓練SVM分類器
    """
    svm = SVC(kernel='rbf', C=10, gamma='scale')
    svm.fit(train_features, train_labels)
    return svm

def evaluate_classifier(model, test_features, test_labels):
    """
    評估分類器的表現
    """
    predictions = model.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=['Fake', 'Real'])
    
    return accuracy, report

def export_normalized_features(train_features_norm, test_features_norm, train_labels, test_labels, output_csv):
    """
    將標準化後的特徵匯出到CSV檔案
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 寫入標頭
        header = ['label'] + [f"f{i}" for i in range(1, 26)]
        writer.writerow(header)
        
        # 寫入訓練資料
        for i in range(len(train_features_norm)):
            writer.writerow([train_labels[i]] + list(train_features_norm[i]))
        
        # 寫入測試資料
        for i in range(len(test_features_norm)):
            writer.writerow([test_labels[i]] + list(test_features_norm[i]))
    
    print(f"標準化特徵已匯出至 {output_csv}")

def generate_report(model, train_features, train_labels, test_features, test_labels, results_dir):
    """
    產生詳細的分析報告並儲存到指定目錄
    """
    # 建立報告目錄
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # 準備報告檔案
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f"classification_report_{timestamp}.txt")
    
    # 取得預測結果
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)
    
    # 計算準確率
    train_accuracy = accuracy_score(train_labels, train_predictions)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    
    # 產生混淆矩陣
    train_cm = confusion_matrix(train_labels, train_predictions)
    test_cm = confusion_matrix(test_labels, test_predictions)
    
    # 產生分類報告
    train_report = classification_report(train_labels, train_predictions, target_names=['Fake', 'Real'])
    test_report = classification_report(test_labels, test_predictions, target_names=['Fake', 'Real'])
    
    # 寫入報告檔案
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== Chinese Character Verification System Analysis Report ===\n\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== Training Data Performance ==\n")
        f.write(f"Training data size: {len(train_labels)}\n")
        f.write(f"Training accuracy: {train_accuracy:.4f}\n\n")
        f.write("Classification report for training data:\n")
        f.write(train_report)
        f.write("\nConfusion matrix for training data:\n")
        f.write(f"[[{train_cm[0][0]}, {train_cm[0][1]}],\n [{train_cm[1][0]}, {train_cm[1][1]}]]\n\n")
        
        f.write("== Testing Data Performance ==\n")
        f.write(f"Testing data size: {len(test_labels)}\n")
        f.write(f"Testing accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification report for testing data:\n")
        f.write(test_report)
        f.write("\nConfusion matrix for testing data:\n")
        f.write(f"[[{test_cm[0][0]}, {test_cm[0][1]}],\n [{test_cm[1][0]}, {test_cm[1][1]}]]\n\n")
        
        f.write("== Model Parameters ==\n")
        f.write(f"Model type: SVM\n")
        f.write(f"Kernel: {model.kernel}\n")
        f.write(f"C value: {model.C}\n")
        f.write(f"Gamma value: {model.gamma}\n")
        
    print(f"Analysis report saved to: {report_file}")
    
    # 產生混淆矩陣視覺化 (使用英文標籤)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Training Data Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('Testing Data Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    cm_image_path = os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_image_path)
    print(f"Confusion matrix visualization saved to: {cm_image_path}")
    
    return report_file, cm_image_path

def export_vector_features_to_csv(data_dir, output_csv):
    """
    匯出向量形式的特徵 f26-f29 到CSV檔案
    每張圖片的向量特徵會以多行方式記錄
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        header = ['filename', 'feature_type', 'index', 'value']
        writer.writerow(header)

        # Process all images in data_dir and all of its subdirectories
        for subdir, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.bmp'):
                    image_path = os.path.join(subdir, file)
                    image = Image.open(image_path)
                    features, _ = feature_extraction(image)
                    
                    # Extract vectors if they exist
                    if 'f26_vector' in features:
                        # f26 vector
                        for i, value in enumerate(features['f26_vector']):
                            writer.writerow([file, 'f26_m_hat', i, value])
                        
                        # f27 vector
                        for i, value in enumerate(features['f27_vector']):
                            writer.writerow([file, 'f27_n_hat', i, value])
                        
                        # f28 vector
                        for i, value in enumerate(features['f28_vector']):
                            writer.writerow([file, 'f28_m_ratio', i, value])
                        
                        # f29 vector
                        for i, value in enumerate(features['f29_vector']):
                            writer.writerow([file, 'f29_n_ratio', i, value])

    print(f"向量特徵已匯出至 {output_csv}")

def export_vector_features_wide_format(data_dir, output_csv):
    """
    匯出向量形式的特徵到CSV (寬格式)
    每張圖片一行，向量元素作為多個欄位
    """
    all_data = []
    
    # First pass: collect all data to determine max vector lengths
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.bmp'):
                image_path = os.path.join(subdir, file)
                image = Image.open(image_path)
                features, _ = feature_extraction(image)
                
                if 'f26_vector' in features:
                    all_data.append({
                        'filename': file,
                        'f26_vector': features['f26_vector'],
                        'f27_vector': features['f27_vector'],
                        'f28_vector': features['f28_vector'],
                        'f29_vector': features['f29_vector']
                    })
    
    if not all_data:
        print("沒有找到向量特徵資料")
        return
    
    # Find maximum vector lengths
    max_f26_len = max(len(data['f26_vector']) for data in all_data)
    max_f27_len = max(len(data['f27_vector']) for data in all_data)
    max_f28_len = max(len(data['f28_vector']) for data in all_data)
    max_f29_len = max(len(data['f29_vector']) for data in all_data)
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Create header
        header = ['filename']
        
        # Add f26 columns
        for i in range(max_f26_len):
            header.append(f'f26_m_hat_{i}')
        
        # Add f27 columns
        for i in range(max_f27_len):
            header.append(f'f27_n_hat_{i}')
        
        # Add f28 columns
        for i in range(max_f28_len):
            header.append(f'f28_m_ratio_{i}')
        
        # Add f29 columns
        for i in range(max_f29_len):
            header.append(f'f29_n_ratio_{i}')
        
        writer.writerow(header)
        
        # Write data
        for data in all_data:
            row = [data['filename']]
            
            # Add f26 values (pad with zeros if needed)
            f26_padded = data['f26_vector'] + [0] * (max_f26_len - len(data['f26_vector']))
            row.extend(f26_padded)
            
            # Add f27 values (pad with zeros if needed)
            f27_padded = data['f27_vector'] + [0] * (max_f27_len - len(data['f27_vector']))
            row.extend(f27_padded)
            
            # Add f28 values (pad with zeros if needed)
            f28_padded = data['f28_vector'] + [0] * (max_f28_len - len(data['f28_vector']))
            row.extend(f28_padded)
            
            # Add f29 values (pad with zeros if needed)
            f29_padded = data['f29_vector'] + [0] * (max_f29_len - len(data['f29_vector']))
            row.extend(f29_padded)
            
            writer.writerow(row)

    print(f"向量特徵 (寬格式) 已匯出至 {output_csv}")
    print(f"向量長度: f26={max_f26_len}, f27={max_f27_len}, f28={max_f28_len}, f29={max_f29_len}")

if __name__ == "__main__":
    data_dir = '../data/1'
    
    # 建立結果目錄
    results_dir = '../results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 載入並準備資料
    print("載入並準備資料...")
    train_features, train_labels, test_features, test_labels = load_and_prepare_data(data_dir)
    
    # 標準化特徵
    print("標準化特徵...")
    train_features_norm, test_features_norm = normalize_features(train_features, test_features)
    
    # 匯出標準化特徵到CSV
    output_csv = '../results/normalized_features.csv'
    export_normalized_features(train_features_norm, test_features_norm, train_labels, test_labels, output_csv)
    
    # 訓練SVM分類器
    print("訓練SVM分類器...")
    svm_model = train_svm_classifier(train_features_norm, train_labels)
    
    # 評估分類器
    print("評估分類器...")
    accuracy, report = evaluate_classifier(svm_model, test_features_norm, test_labels)
    
    print(f"準確率: {accuracy:.4f}")
    print("詳細評估報告:")
    print(report)
    
    # 產生詳細報告
    print("產生詳細分析報告...")
    report_file, cm_image_path = generate_report(
        svm_model, train_features_norm, train_labels, 
        test_features_norm, test_labels, results_dir
    )
    
    # 匯出向量特徵到CSV
    output_vector_csv = '../results/vector_features.csv'
    export_vector_features_to_csv(data_dir, output_vector_csv)
    
    # 匯出向量特徵 (寬格式) 到CSV
    output_vector_wide_csv = '../results/vector_features_wide.csv'
    export_vector_features_wide_format(data_dir, output_vector_wide_csv)