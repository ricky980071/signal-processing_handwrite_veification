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
    """偵測筆劃的邊緣"""
    # 使用 OpenCV 的 findContours 函數找到輪廓
    contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 創建一個空白圖像來畫輪廓
    edges = np.zeros_like(stroke_mask)
    
    # 在空白圖像上畫出所有輪廓
    cv2.drawContours(edges, contours, -1, 1, 1)
    
    # 另外檢測內部輪廓
    contours_inner, _ = cv2.findContours(stroke_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours_inner, -1, 1, 1)
    
    return edges

def clock_wise_edge_numbering(edges):
    """按順時針方式給邊緣像素編號"""
    h, w = edges.shape
    numbered_edges = np.zeros_like(edges, dtype=np.int32)
    
    # 尋找所有邊緣像素的座標
    edge_points = np.where(edges > 0)
    edge_coords = list(zip(edge_points[0], edge_points[1]))
    total_edge_points = len(edge_coords)
    
    # 記錄已處理的邊緣像素
    processed = set()
    
    # 編號計數器
    counter = 1
    
    # 定義8個相鄰方向 (順時針順序): 右、右下、下、左下、左、左上、上、右上
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    
    print(f"找到 {total_edge_points} 個邊緣點需要編號")
    
    # 防止無窮迴圈的最大迭代次數
    max_total_iterations = total_edge_points * 10  # 設置一個合理但寬裕的上限
    total_iterations = 0
    
    # 處理所有可能的筆劃，直到所有邊緣點都被編號或達到最大迭代次數
    while len(processed) < total_edge_points and total_iterations < max_total_iterations:
        # 找到最左上角還未處理的邊緣點作為起點
        start_point = None
        for y in range(h):
            for x in range(w):
                if edges[y, x] > 0 and (y, x) not in processed:
                    start_point = (y, x)
                    break
            if start_point:
                break
        
        if not start_point:
            print("所有點已處理完畢")
            break
        
        print(f"開始處理新的筆劃，起點：{start_point}，當前已處理 {len(processed)}/{total_edge_points} 個點")
        
        # 當前筆劃中已處理的點
        current_stroke_points = set()
        
        # 設置起點的編號
        current = start_point
        numbered_edges[current[0], current[1]] = counter
        processed.add(current)
        current_stroke_points.add(current)
        counter += 1
        
        # 防止單個筆劃無窮迴圈的計數器
        stroke_iterations = 0
        max_stroke_iterations = min(total_edge_points * 2, 1000) # 限制單個筆劃的最大迭代次數
        
        # 使用右手貼牆法順時針繞行
        while stroke_iterations < max_stroke_iterations:
            stroke_iterations += 1
            total_iterations += 1
            
            if stroke_iterations % 100 == 0:
                print(f"  當前筆劃已迭代 {stroke_iterations} 次，總迭代 {total_iterations} 次")
            
            # 用來找下一個點的標記
            found_next = False
            
            # 嘗試找一個未處理的相鄰點
            for dir_idx in range(8):
                dy, dx = directions[dir_idx]
                next_y, next_x = current[0] + dy, current[1] + dx
                next_point = (next_y, next_x)
                
                # 檢查是否在範圍內且是邊緣點
                if (0 <= next_y < h and 0 <= next_x < w and 
                    edges[next_y, next_x] > 0):
                    
                    # 如果點未處理過，則進行編號
                    if next_point not in processed:
                        current = next_point
                        numbered_edges[current[0], current[1]] = counter
                        processed.add(current)
                        current_stroke_points.add(current)
                        counter += 1
                        found_next = True
                        break
                    # 如果是起點且處理了其他點，則繞行完成
                    elif next_point == start_point and len(current_stroke_points) > 1:
                        found_next = False  # 設置為False以結束當前筆劃
                        print(f"  回到起點，結束當前筆劃繞行")
                        break
            
            # 如果無法找到未處理的點，則嘗試找一個已處理的點繼續走
            if not found_next:
                # 檢查是否可以走回起點
                for dir_idx in range(8):
                    dy, dx = directions[dir_idx]
                    next_y, next_x = current[0] + dy, current[1] + dx
                    
                    if (0 <= next_y < h and 0 <= next_x < w and 
                        (next_y, next_x) == start_point and 
                        len(current_stroke_points) > 1):
                        print(f"  找到路徑回到起點，結束當前筆劃")
                        found_next = False  # 結束當前筆劃
                        break
                
                # 如果找不到路徑回到起點，則嘗試任何已處理的點
                if not found_next:
                    for dir_idx in range(8):
                        dy, dx = directions[dir_idx]
                        next_y, next_x = current[0] + dy, current[1] + dx
                        next_point = (next_y, next_x)
                        
                        if (0 <= next_y < h and 0 <= next_x < w and 
                            edges[next_y, next_x] > 0 and 
                            next_point in current_stroke_points and 
                            next_point != current):
                            
                            # 移動到已處理的點，但不重新編號
                            current = next_point
                            found_next = True
                            break
            
            # 如果無法找到任何可以走的點，則結束當前筆劃的繞行
            if not found_next:
                print(f"  無法找到下一個點，結束當前筆劃繞行")
                break
        
        # 檢查是否達到單個筆劃的最大迭代次數
        if stroke_iterations >= max_stroke_iterations:
            print(f"  警告：達到最大迭代次數 {max_stroke_iterations}，強制結束當前筆劃")
        
        print(f"完成一個筆劃的處理，共標記了 {len(current_stroke_points)} 個點，剩餘 {total_edge_points - len(processed)} 個點未處理")
    
    # 檢查是否達到總迭代次數上限
    if total_iterations >= max_total_iterations:
        print(f"警告：達到總最大迭代次數 {max_total_iterations}，提前結束處理")
        print(f"總共處理了 {len(processed)}/{total_edge_points} 個點")
    
    print(f"編號完成，總共標記了 {counter-1} 個點")
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