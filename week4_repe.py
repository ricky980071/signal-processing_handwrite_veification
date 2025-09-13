from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.widgets import Slider

def extract_strokes(image_path, verbose=False):
    """從圖片中提取筆劃"""
    # 開啟圖片並轉灰階
    img = Image.open(image_path).convert('RGB')
    np_img = np.array(img)

    # RGB -> Grayscale
    Y = 0.299 * np_img[:, :, 0] + 0.587 * np_img[:, :, 1] + 0.114 * np_img[:, :, 2]
    binary_img = (Y > 220).astype(np.uint8) * 255  # 字跡 = 黑 (0), 背景 = 白 (255)

    # 把黑色 (0) 當作筆跡
    stroke_mask = (binary_img == 0).astype(np.uint8)
    
    if verbose:
        plt.figure(figsize=(8, 8))
        plt.imshow(stroke_mask, cmap='gray')
        plt.title('提取的筆劃')
        plt.show()
        
    return stroke_mask

def detect_edges(stroke_mask):
    """偵測筆劃的邊緣，使用直接的輪廓檢測方法"""
    # 確保輸入是 uint8 格式
    stroke_mask_uint8 = (stroke_mask * 255).astype(np.uint8)
    
    # 使用 findContours 直接檢測輪廓
    contours, hierarchy = cv2.findContours(stroke_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # 創建一個空白圖像來畫輪廓
    edges = np.zeros_like(stroke_mask, dtype=np.uint8)
    
    # 畫出所有輪廓的邊緣
    for i, contour in enumerate(contours):
        # 檢查輪廓面積，過濾太小的輪廓
        area = cv2.contourArea(contour)
        if area > 10:  # 只保留面積大於10的輪廓
            # 只畫輪廓邊緣，不填充
            cv2.drawContours(edges, [contour], -1, 1, thickness=1)
    
    return edges

def clock_wise_edge_numbering(edges):
    h, w = edges.shape
    numbered_edges = np.zeros_like(edges, dtype=np.int32)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    edge_points = np.where(edges > 0)
    edge_coords = list(zip(edge_points[0], edge_points[1]))
    total_edge_points = len(edge_coords)
    processed = set()
    counter = 1
    while len(processed) < total_edge_points:
        # 找到最靠近左上角且有鄰居的未處理點作為起點
        start_point = None
        min_distance = float('inf')
        for y, x in edge_coords:
            if (y, x) not in processed:
                neighbor_count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < h and 0 <= nx < w and edges[ny, nx] > 0 and (ny, nx) not in processed):
                            neighbor_count += 1
                if neighbor_count >= 2:
                    distance = y * y + x * x
                    if distance < min_distance:
                        min_distance = distance
                        start_point = (y, x)
        if not start_point:
            for y, x in edge_coords:
                if (y, x) not in processed:
                    distance = y * y + x * x
                    if distance < min_distance:
                        min_distance = distance
                        start_point = (y, x)
        if not start_point:
            break
        current_stroke_points = []
        current = start_point
        numbered_edges[current[0], current[1]] = counter
        processed.add(current)
        current_stroke_points.append(current)
        counter += 1
        # 找第二個點
        second_point = None
        for i in range(8):
            dy, dx = directions[i]
            ny, nx = current[0] + dy, current[1] + dx
            if (0 <= ny < h and 0 <= nx < w and edges[ny, nx] > 0 and (ny, nx) not in processed):
                second_point = (ny, nx)
                prev_idx = i
                break
        if second_point is None:
            continue
        current = second_point
        numbered_edges[current[0], current[1]] = counter
        processed.add(current)
        current_stroke_points.append(current)
        counter += 1
        prev = current_stroke_points[-2]
        start = start_point
        max_iterations = total_edge_points * 3
        iteration_count = 0
        while iteration_count < max_iterations:
            iteration_count += 1
            cy, cx = current
            py, px = prev
            rel = (py - cy, px - cx)
            prev_idx = None
            for i, (dy, dx) in enumerate(directions):
                if (dy, dx) == rel:
                    prev_idx = i
                    break
            found = False
            # 順時鐘掃描，從(prev_idx+1)開始，依序掃8個方向
            for offset in range(1, 9):
                idx = (prev_idx + offset) % 8
                dy, dx = directions[idx]
                ny, nx = cy + dy, cx + dx
                next_point = (ny, nx)
                if 0 <= ny < h and 0 <= nx < w and edges[ny, nx] > 0:
                    if next_point == prev:
                        # 筆劃結束，但允許重複編號，繼續順時鐘掃描
                        numbered_edges[next_point[0], next_point[1]] = counter
                        counter += 1
                        prev, current = current, next_point
                        found = True
                        break
                    if next_point == start and len(current_stroke_points) > 2:
                        # 回到起始點，這一段輪廓結束
                        numbered_edges[next_point[0], next_point[1]] = counter
                        counter += 1
                        found = True
                        break
                    # 正常走到下一個點
                    prev, current = current, next_point
                    numbered_edges[next_point[0], next_point[1]] = counter
                    processed.add(next_point)
                    current_stroke_points.append(next_point)
                    counter += 1
                    found = True
                    break
            if found and current == start and len(current_stroke_points) > 2:
                break
            if not found:
                break
        # 如果這段輪廓長度太短，視為雜訊，全部標記為已處理但不給編號
        if len(current_stroke_points) <= 5:
            for pt in current_stroke_points:
                numbered_edges[pt[0], pt[1]] = 0
            counter -= len(current_stroke_points)
    return numbered_edges

def visualize_numbered_edges(original_img, numbered_edges, image_name=""):
    """將編號後的邊緣像素可視化"""
    plt.figure(figsize=(12, 10))
    
    # 顯示原始圖像
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f'原始圖像: {image_name}')
    
    # 顯示編號後的邊緣
    plt.subplot(2, 2, 2)
    plt.imshow(numbered_edges, cmap='viridis')
    plt.title('編號後的邊緣')
    
    # 顯示部分編號標記
    plt.subplot(2, 2, 3)
    display_img = original_img.copy()
    max_num = np.max(numbered_edges)
    
    # 只標記部分點，以10為間隔，方便觀察
    for i in range(1, max_num + 1, 10):
        points = np.where(numbered_edges == i)
        if len(points[0]) > 0:
            y, x = points[0][0], points[1][0]
            plt.text(x, y, str(i), color='red', fontsize=8)
    
    plt.imshow(display_img, cmap='gray')
    plt.title('編號標記 (每 10 點標記一次)')
    
    # 創建一個只顯示編號的圖
    plt.subplot(2, 2, 4)
    blank = np.zeros_like(original_img)
    for i in range(1, max_num + 1, 10):
        points = np.where(numbered_edges == i)
        if len(points[0]) > 0:
            y, x = points[0][0], points[1][0]
            plt.text(x, y, str(i), color='red', fontsize=8)
    
    plt.imshow(blank, cmap='gray')
    plt.title('只顯示編號')
    
    plt.tight_layout()
    plt.show()

def visualize_with_slider(original_img, numbered_edges, image_name=""):
    """使用滑桿來顯示編號順序"""
    max_num = np.max(numbered_edges)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # 初始顯示全部編號
    display_img = original_img.copy()
    
    # 創建彩色疊加圖像
    overlay = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    
    # 初始顯示編號1的點
    initial_num = 1
    
    # 更新函數
    def update(val):
        num = int(slider.val)
        
        # 清空疊加圖像
        overlay.fill(0)
        
        # 顯示到當前編號的點
        for i in range(1, num + 1):
            points = np.where(numbered_edges == i)
            if len(points[0]) > 0:
                for y, x in zip(points[0], points[1]):
                    # 使用彩色標記，讓編號順序更加明顯
                    # 使用不同顏色來表示不同的編號範圍
                    if i % 30 == 0:  # 每30個點使用紅色標記
                        overlay[y, x] = [255, 0, 0]  # 紅色
                    elif i % 20 == 0:  # 每20個點使用綠色標記
                        overlay[y, x] = [0, 255, 0]  # 綠色
                    elif i % 10 == 0:  # 每10個點使用藍色標記
                        overlay[y, x] = [0, 0, 255]  # 藍色
                    else:
                        overlay[y, x] = [0, 255, 255]  # 青色
        
        # 創建背景圖像（灰度轉為RGB）
        bg_image = np.stack([display_img] * 3, axis=-1)
        
        # 將覆蓋層疊加到背景上
        mask = (overlay > 0).any(axis=2)
        bg_image[mask] = overlay[mask]
        
        # 更新圖像
        img_plot.set_data(bg_image)
        ax.set_title(f'編號順序滑桿 ({image_name}) - 顯示到編號: {num}/{max_num}')
        
        fig.canvas.draw_idle()
    
    # 創建初始圖像
    overlay_init = np.zeros((original_img.shape[0], original_img.shape[1], 3), dtype=np.uint8)
    points = np.where(numbered_edges == initial_num)
    if len(points[0]) > 0:
        for y, x in zip(points[0], points[1]):
            overlay_init[y, x] = [0, 255, 255]  # 青色
    
    bg_image_init = np.stack([display_img] * 3, axis=-1)
    mask_init = (overlay_init > 0).any(axis=2)
    bg_image_init[mask_init] = overlay_init[mask_init]
    
    # 設置初始圖像
    img_plot = ax.imshow(bg_image_init)
    ax.set_title(f'編號順序滑桿 ({image_name}) - 顯示到編號: {initial_num}/{max_num}')
    
    # 添加滑桿
    axnum = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(axnum, '編號', 1, max_num, valinit=initial_num, valstep=1)
    
    slider.on_changed(update)
    plt.show()

def process_images():
    """處理資料夾中的圖像"""
    for digit in range(1, 10):
        # 處理每個數字資料夾的第一張圖片
        database_path = f'handwrite/{digit}/database'
        
        if not os.path.exists(database_path):
            print(f"警告：資料夾 {database_path} 不存在")
            continue
        
        # 獲取資料夾中的第一個圖像
        image_files = sorted(os.listdir(database_path))
        if not image_files:
            print(f"警告：資料夾 {database_path} 中沒有圖像")
            continue
        
        image_path = os.path.join(database_path, image_files[0])
        print(f"處理圖像: {image_path}")
        
        # 提取筆劃
        stroke_mask = extract_strokes(image_path)
        
        # 偵測邊緣
        edges = detect_edges(stroke_mask)
        
        # 按順時針方式給邊緣像素編號
        numbered_edges = clock_wise_edge_numbering(edges)
        
        # 可視化結果
        visualize_numbered_edges(stroke_mask, numbered_edges, f"數字 {digit}")
        
        # 為每個數字的第一張圖像添加滑桿視覺化
        visualize_with_slider(stroke_mask, numbered_edges, f"數字 {digit}")

if __name__ == "__main__":
    process_images()