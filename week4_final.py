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
    """按順時針方式給邊緣像素編號（可跨已處理點，直到無路可走才結束）"""
    h, w = edges.shape
    numbered_edges = np.zeros_like(edges, dtype=np.int32)
    edge_points = np.where(edges > 0)
    edge_coords = list(zip(edge_points[0], edge_points[1]))
    total_edge_points = len(edge_coords)
    counter = 1
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    # print(f"找到 {total_edge_points} 個邊緣點需要編號")
    processed = set()
    while len(processed) < total_edge_points:
        # 找到最左上角還沒走過的邊緣點作為起點
        start_point = None
        for y in range(h):
            for x in range(w):
                if edges[y, x] > 0 and (y, x) not in processed:
                    start_point = (y, x)
                    break
            if start_point:
                break
        if not start_point:
            # print("所有點已處理完畢")
            break
        # print(f"開始處理新的筆劃，起點：{start_point}，當前已處理 {len(processed)}/{total_edge_points} 個點")
        current = start_point
        current_stroke_points = [current]
        numbered_edges[current[0], current[1]] = counter
        counter += 1
        processed.add(current)
        current_direction = 0  # 右
        max_iterations = total_edge_points * 4
        iteration_count = 0
        while iteration_count < max_iterations:
            iteration_count += 1
            found_next = False
            # 先找未處理過的點
            for i in range(8):
                search_direction = (current_direction + i) % 8
                dy, dx = directions[search_direction]
                next_y, next_x = current[0] + dy, current[1] + dx
                next_point = (next_y, next_x)
                if (0 <= next_y < h and 0 <= next_x < w and edges[next_y, next_x] > 0 and next_point not in processed):
                    current = next_point
                    numbered_edges[current[0], current[1]] = counter
                    counter += 1
                    current_stroke_points.append(current)
                    processed.add(current)
                    current_direction = (search_direction + 4 + 2) % 8
                    found_next = True
                    # 如果回到起點且已經繞過2個點以上，結束
                    if current == start_point and len(current_stroke_points) > 2:
                        # print(f"  回到起點，結束當前筆劃繞行")
                        break
                    else:
                        break
            if found_next:
                continue
            # 如果沒找到未處理點，允許走已處理過但不是起點的點
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
            # 如果還是沒找到，或回到起點且已經繞過2個點以上，才結束
            if not found_next or (current == start_point and len(current_stroke_points) > 2):
                break
        # print(f"完成一個筆劃的處理，共標記了 {len(current_stroke_points)} 個點，剩餘 {total_edge_points - len(processed)} 個點未處理")
    # print(f"編號完成，總共標記了 {counter-1} 個點")
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