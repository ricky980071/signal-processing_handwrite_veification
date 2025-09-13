# -*- coding: utf-8 -*-
# week5_noinside.py
# 這個檔案主要功能：
# 1. 讀取手寫字圖像，提取外輪廓，順時針編號每個部件像素
# 2. 依照順時針編號分群，計算每個部件的角度特徵
# 3. 可視化順時針編號、特徵點、角度滑桿等

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.widgets import Slider

# 讀取圖像並轉成二值筆劃遮罩
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
    """
    偵測筆劃的邊緣，只保留最外層輪廓
    輸入: stroke_mask (二值遮罩)
    輸出: edges (二值邊緣圖)
    """
    # 確保輸入是 uint8 格式
    stroke_mask_uint8 = (stroke_mask * 255).astype(np.uint8)
    
    # 使用 findContours 只偵測最外層輪廓
    contours, hierarchy = cv2.findContours(stroke_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
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

# 順時鐘編號與分群
def clock_wise_edge_numbering(edges):
    """
    對邊緣圖像進行順時鐘編號，並分群每個部件
    回傳:
    - numbered_edges: 每個像素的順時鐘編號（全域唯一遞增）
    - all_stroke_coords: list of list，每個部件的像素座標依照順時鐘編號順序
    """
    h, w = edges.shape  # 取得圖像高度與寬度
    numbered_edges = np.zeros_like(edges, dtype=np.int32)  # 建立同樣大小的編號陣列
    edge_points = np.where(edges > 0)  # 找出所有邊緣點的座標
    edge_coords = list(zip(edge_points[0], edge_points[1]))  # 轉成(y, x)座標list
    total_edge_points = len(edge_coords)  # 邊緣點總數
    counter = 1  # 全域順時鐘編號計數器，從1開始
    # 定義8個方向（順時針）：右、右下、下、左下、左、左上、上、右上
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    processed = set()  # 已處理過的點
    all_stroke_coords = []  # 儲存每個部件的順時鐘像素list
    while len(processed) < total_edge_points:  # 還有未處理的點就繼續
        start_point = None
        # 從左上到右下找一個未處理的邊緣點作為新部件的起點
        for y in range(h):
            for x in range(w):
                if edges[y, x] > 0 and (y, x) not in processed:
                    start_point = (y, x)
                    break
            if start_point:
                break
        if not start_point:
            break  # 沒有新起點就結束
        current = start_point  # 設定目前點為起點
        current_stroke_points = []  # 新部件的像素list
        start_num = counter  # 記錄這個部件的起始順時鐘編號
        numbered_edges[current[0], current[1]] = counter  # 給起點編號
        current_stroke_points.append(current)  # 加入部件list
        counter += 1
        processed.add(current)  # 標記已處理
        current_direction = 0  # 初始方向設為右
        max_iterations = total_edge_points * 4  # 防止無窮迴圈
        iteration_count = 0
        num_to_coord = {}  # 新增：順時鐘編號對應座標dict
        while iteration_count < max_iterations:
            iteration_count += 1
            found_next = False
            # 依序檢查8個方向，找未處理的鄰居
            for i in range(8):
                search_direction = (current_direction + i) % 8
                dy, dx = directions[search_direction]
                next_y, next_x = current[0] + dy, current[1] + dx
                next_point = (next_y, next_x)
                # 如果是合法的未處理邊緣點
                if (0 <= next_y < h and 0 <= next_x < w and edges[next_y, next_x] > 0 and next_point not in processed):
                    current = next_point  # 移動到新點
                    numbered_edges[current[0], current[1]] = counter  # 給新點編號
                    current_stroke_points.append(current)  # 加入部件list
                    counter += 1
                    processed.add(current)  # 標記已處理
                    num_to_coord[counter-1] = current  # 新增：記錄編號對應座標
                    # 更新方向（右手貼牆法）
                    current_direction = (search_direction + 4 + 2) % 8
                    found_next = True
                    # 如果回到起點且已繞過2個點以上，結束這個部件
                    if current == start_point and len(current_stroke_points) > 2:
                        break
                    else:
                        break
            if found_next:
                continue  # 找到新點就繼續
            # 如果沒找到未處理點，允許走已處理但不是起點的點（防止斷裂）
            for i in range(8):
                search_direction = (current_direction + i) % 8
                dy, dx = directions[search_direction]
                next_y, next_x = current[0] + dy, current[1] + dx
                next_point = (next_y, next_x)
                if (0 <= next_y < h and 0 <= next_x < w and edges[next_y, next_x] > 0 and next_point != start_point):
                    current = next_point
                    current_stroke_points.append(current)
                    current_direction = (search_direction + 4 + 2) % 8
                    found_next = True
                    break
            # 如果還是沒找到，或回到起點且已繞過2個點以上，結束這個部件
            if not found_next or (current == start_point and len(current_stroke_points) > 2):
                break
        end_num = counter - 1  # 記錄這個部件的結束順時鐘編號
        if len(current_stroke_points) > 2:
            # 用 dict 快速查找，不用 np.where
            block_coords = [num_to_coord[num] for num in range(start_num, end_num + 1) if num in num_to_coord]
            all_stroke_coords.append(block_coords)  # 儲存這個部件
    return numbered_edges, all_stroke_coords  # 回傳順時鐘編號陣列與所有部件的像素list

# 計算每個部件的角度特徵
def compute_edge_angles(all_stroke_coords, d=12):
    """
    直接用 all_stroke_coords（每個區塊的像素座標list）計算每個點的夾角 theta_i
    回傳: List of theta_array, 每個區塊一個 array
    """
    theta_arrays = []
    for block_idx, block_coords in enumerate(all_stroke_coords):
        n = len(block_coords)
        if n < 2 * d + 1:
            theta_arrays.append(np.zeros(n))
            continue
        theta_arr = []
        for i in range(n):
            y0, x0 = block_coords[i]
            y1, x1 = block_coords[(i + d) % n]
            y2, x2 = block_coords[(i - d) % n]
            v1 = np.array([y1 - y0, x1 - x0])
            v2 = np.array([y2 - y0, x2 - x0])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                print(f"[異常] block {block_idx+1} idx={i}, norm1={norm1}, norm2={norm2}, v1={v1}, v2={v2}, coord={block_coords[i]}")
                theta = np.nan  # 用 nan 標記異常
            else:
                cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
                theta = np.arccos(cos_theta)
            theta_arr.append(theta)
        theta_arrays.append(np.array(theta_arr))
    return theta_arrays

# 可視化順時鐘編號、特徵點、滑桿等
def visualize_numbered_edges(original_img, numbered_edges, image_name="", start_points=None, angle_points=None):
    """將編號後的邊緣像素可視化，並標記每個部件的起點與角度特徵點"""
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
    
    # 標記部件起點
    if start_points is not None:
        for idx, (y, x) in enumerate(start_points):
            plt.plot(x, y, marker='o', color='lime', markersize=10, markeredgewidth=2, fillstyle='none')
            plt.text(x, y, f'S{idx+1}', color='lime', fontsize=12, fontweight='bold')
    
    # 標記角度特徵點
    if angle_points is not None:
        for pt in angle_points:
            y, x, typ = pt
            if typ == 'ending':
                plt.plot(x, y, marker='*', color='blue', markersize=14, markeredgewidth=2, fillstyle='none', label='ending')
            elif typ == 'turning':
                plt.plot(x, y, marker='^', color='orange', markersize=12, markeredgewidth=2, fillstyle='none', label='turning')
    
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
    """使用滑桿來顯示編號順序，顏色漸變且每次只顯示最新筆畫顏色，重複走到的點會被新顏色蓋過"""
    import matplotlib.cm as cm
    max_num = np.max(numbered_edges)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    display_img = original_img.copy()
    cmap = cm.get_cmap('jet', max_num)
    def update(val):
        num = int(slider.val)
        bg_image = np.stack([display_img] * 3, axis=-1)
        # 只顯示最新的顏色（當前步驟的點）
        for i in range(1, num + 1):
            points = np.where(numbered_edges == i)
            if len(points[0]) > 0:
                color = (np.array(cmap(i / max_num)[:3]) * 255).astype(np.uint8)
                for y, x in zip(points[0], points[1]):
                    bg_image[y, x] = color
        img_plot.set_data(bg_image)
        ax.set_title(f'編號順序滑桿 ({image_name}) - 顯示到編號: {num}/{max_num}')
        fig.canvas.draw_idle()
    # 初始圖像
    bg_image_init = np.stack([display_img] * 3, axis=-1)
    i = 1
    points = np.where(numbered_edges == i)
    if len(points[0]) > 0:
        color = (np.array(cmap(i / max_num)[:3]) * 255).astype(np.uint8)
        for y, x in zip(points[0], points[1]):
            bg_image_init[y, x] = color
    img_plot = ax.imshow(bg_image_init)
    ax.set_title(f'編號順序滑桿 ({image_name}) - 顯示到編號: 1/{max_num}')
    # 添加滑桿
    axnum = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(axnum, '編號', 1, max_num, valinit=1, valstep=1)
    slider.on_changed(update)
    plt.show()

# 角度特徵點偵測
def find_angle_points(theta_arr_deg, case=1, window=12):
    """
    找出角度 array 中屬於 case1/case2 的 ending/turning point。
    case=1: <35 度，case=2: 35~150 度
    window: 前後各 window 個點（共 2*window+1 個，環狀）
    回傳: [(index, angle)]
    - case1 與 case2 判斷完全獨立，各自有自己的 used 標記
    """
    n = len(theta_arr_deg)
    points = []  # 儲存特徵點 (index, angle)
    if case == 1:
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            if used[i]:
                continue
            angle = theta_arr_deg[i]
            if angle >= 35:
                continue
            idxs = [(i + j) % n for j in range(-window, window+1)]
            if not all(angle <= theta_arr_deg[j] for j in idxs):
                continue
            for j in idxs:
                used[j] = True
            points.append((i, angle))
    elif case == 2:
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            if used[i]:
                continue
            angle = theta_arr_deg[i]
            if angle < 35 or angle > 150:
                continue
            idxs = [(i + j) % n for j in range(-window, window+1)]
            if not all(angle <= theta_arr_deg[j] for j in idxs):
                continue
            for j in idxs:
                used[j] = True
            points.append((i, angle))
    return points

def visualize_all_blocks_with_points(original_img, numbered_edges, all_stroke_coords, theta_arrays, image_name="", debug=True): 
    """將所有區塊的起點、ending/turning point 同時標記在一张圖上，並對ending point畫出與前後第d個點的連線。debug模式下會print座標與距離。"""
    d = 12  # 與角度判斷一致
    plt.figure(figsize=(7, 7))
    plt.imshow(original_img, cmap='gray')
    plt.title(f'所有區塊特徵點: {image_name}')
    colors = ['lime', 'cyan', 'magenta', 'yellow', 'red', 'blue', 'orange', 'purple', 'brown']
    for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
        color = colors[block_idx % len(colors)]
        # 在每個部件的第一個點標上部件編號
        if len(coords) > 0:
            y, x = coords[0]
            plt.plot(x, y, marker='o', color=color, markersize=6, markeredgewidth=1, fillstyle='none')
            plt.text(x, y, f'B{block_idx+1}', color='black', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor=color))
        # 起點
        if len(coords) > 0:
            y, x = coords[0]
            plt.plot(x, y, marker='o', color=color, markersize=6, markeredgewidth=1, fillstyle='none')
            plt.text(x, y, f'S{block_idx+1}', color=color, fontsize=8, fontweight='bold')
        # ending/turning point
        theta_deg = np.degrees(theta_arr)
        ending_pts = find_angle_points(theta_deg, case=1, window=11)
        turning_pts = find_angle_points(theta_deg, case=2, window=11)
        n = len(coords)
        # 標記異常點
        for i, theta in enumerate(theta_arr):
            if np.isnan(theta):
                y0, x0 = coords[i]
                plt.plot(x0, y0, marker='x', color='red', markersize=10, markeredgewidth=2, label='異常')
        for idx, _ in ending_pts:
            y0, x0 = coords[idx]
            plt.plot(x0, y0, marker='*', color='blue', markersize=8, markeredgewidth=1, fillstyle='none')
            for offset in [-d, d]:
                j = (idx + offset) % n
                y1, x1 = coords[j]
                plt.plot([x0, x1], [y0, y1], color='deepskyblue', linewidth=1.5, alpha=0.8)
        for idx, _ in turning_pts:
            y, x = coords[idx]
            plt.plot(x, y, marker='^', color='orange', markersize=7, markeredgewidth=1, fillstyle='none')
    plt.tight_layout()
    plt.show()

def visualize_angle_slider(original_img, all_stroke_coords, theta_arrays, image_name="", d=12):
    """滑桿視窗，逐點顯示角度判斷依據與特徵點，並顯示左右夾角與點編號"""
    import matplotlib.cm as cm
    for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
        n = len(coords)
        if n == 0:
            continue
        theta_deg = np.degrees(theta_arr)
        ending_pts = find_angle_points(theta_deg, case=1, window=11)
        turning_pts = find_angle_points(theta_deg, case=2, window=11)
        ending_idx = set(idx for idx, _ in ending_pts)
        turning_idx = set(idx for idx, _ in turning_pts)
        abnormal_idx = set(i for i, t in enumerate(theta_arr) if np.isnan(t))
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(original_img, cmap='gray')
        ax.set_title(f'{image_name} 區塊{block_idx+1} 角度滑桿')
        # 畫所有 ending/turning/異常點
        for i, (y, x) in enumerate(coords):
            if i in ending_idx:
                ax.plot(x, y, marker='*', color='blue', markersize=10)
            if i in turning_idx:
                ax.plot(x, y, marker='^', color='orange', markersize=8)
            if i in abnormal_idx:
                ax.plot(x, y, marker='x', color='red', markersize=10, markeredgewidth=2)
        # 畫起點
        y0, x0 = coords[0]
        ax.plot(x0, y0, marker='o', color='lime', markersize=8, markeredgewidth=2, fillstyle='none')
        # 滑桿
        axnum = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(axnum, '點 index', 0, n-1, valinit=0, valstep=1)
        highlight, = ax.plot([], [], marker='o', color='magenta', markersize=12, markeredgewidth=2, fillstyle='none')
        line1, = ax.plot([], [], color='deepskyblue', linewidth=2)
        line2, = ax.plot([], [], color='deepskyblue', linewidth=2)
        neighbor1, = ax.plot([], [], marker='s', color='deepskyblue', markersize=8)
        neighbor2, = ax.plot([], [], marker='s', color='deepskyblue', markersize=8)
        text_theta = ax.text(0, 0, '', color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        text_idx = ax.text(0, 0, '', color='purple', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        def update(val):
            idx = int(slider.val)
            y, x = coords[idx]
            highlight.set_data([x], [y])
            j1 = (idx - d) % n
            j2 = (idx + d) % n
            y1, x1 = coords[j1]
            y2, x2 = coords[j2]
            line1.set_data([x, x1], [y, y1])
            line2.set_data([x, x2], [y, y2])
            neighbor1.set_data([x1], [y1])
            neighbor2.set_data([x2], [y2])
            theta_val = theta_deg[idx] if not np.isnan(theta_deg[idx]) else '異常'
            theta_left = theta_deg[j1] if not np.isnan(theta_deg[j1]) else '異常'
            theta_right = theta_deg[j2] if not np.isnan(theta_deg[j2]) else '異常'
            text_theta.set_position((x+5, y+5))
            text_theta.set_text(f'θ={theta_val:.1f}°\n左θ={theta_left:.1f}° 右θ={theta_right:.1f}°' if all(isinstance(v, float) for v in [theta_val, theta_left, theta_right]) else f'θ={theta_val if isinstance(theta_val, float) else "異常"}\n左θ={theta_left if isinstance(theta_left, float) else "異常"} 右θ={theta_right if isinstance(theta_right, float) else "異常"}')
            text_idx.set_position((x+5, y-15))
            text_idx.set_text(f'index={idx}\n左={j1} 右={j2}')
            fig.canvas.draw_idle()
        update(0)
        slider.on_changed(update)
        plt.show()

# # 依照順時針編號斷裂分群
# def group_by_numbered_edges(numbered_edges):
#     """
#     根據順時鐘編號直接分群，確保每個部件的像素list是依照順時鐘編號順序，
#     只要遇到順時鐘編號斷裂（不是連號），就視為新部件開始。
#     """
#     points = np.where(numbered_edges > 0)
#     coords = list(zip(points[0], points[1]))
#     values = [numbered_edges[y, x] for y, x in coords]
#     sorted_points = sorted(zip(values, coords))
#     all_stroke_coords = []
#     current_block = []
#     prev_val = None
#     for val, coord in sorted_points:
#         if prev_val is not None and val != prev_val + 1:
#             if len(current_block) > 2:
#                 all_stroke_coords.append(current_block)
#             current_block = []
#         current_block.append(coord)
#         prev_val = val
#     if len(current_block) > 2:
#         all_stroke_coords.append(current_block)
#     return all_stroke_coords

# 主流程：批次處理圖像、計算特徵、可視化
def process_images():
    """處理資料夾中的圖像"""
    for digit in range(3,4):
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
        numbered_edges, all_stroke_coords = clock_wise_edge_numbering(edges)
        
        # 計算每個區塊的theta array
        theta_arrays = compute_edge_angles(all_stroke_coords, d=12)
        for idx, theta_arr in enumerate(theta_arrays):
            print(f"數字 {digit} 區塊 {idx+1} theta array (degree):")
            print(np.degrees(theta_arr))
        
        # 可視化結果
        start_points = [coords[0] for coords in all_stroke_coords if len(coords) > 0]
        visualize_numbered_edges(stroke_mask, numbered_edges, f"數字 {digit}", start_points=start_points)
        
        # 為每個數字的第一張圖像添加滑桿視覺化
        visualize_with_slider(stroke_mask, numbered_edges, f"數字 {digit}")

        # 標記角度特徵點
        for block_idx, (coords, theta_arr) in enumerate(zip(all_stroke_coords, theta_arrays)):
            theta_deg = np.degrees(theta_arr)
            ending_pts = find_angle_points(theta_deg, case=1, window=11)
            turning_pts = find_angle_points(theta_deg, case=2, window=11)
            angle_points = []
            for idx, _ in ending_pts:
                y, x = coords[idx]
                angle_points.append((y, x, 'ending'))
            for idx, _ in turning_pts:
                y, x = coords[idx]
                angle_points.append((y, x, 'turning'))
            visualize_numbered_edges(stroke_mask, numbered_edges, f"數字 {digit} 區塊{block_idx+1}", start_points=[coords[0]], angle_points=angle_points)
        
        # 可視化所有區塊的特徵點與連線（d=12）
        visualize_all_blocks_with_points(stroke_mask, numbered_edges, all_stroke_coords, theta_arrays, image_name=f"數字 {digit}")
        # 新增：角度滑桿視窗
        visualize_angle_slider(stroke_mask, all_stroke_coords, theta_arrays, image_name=f"數字 {digit}", d=12)

if __name__ == "__main__":
    process_images()