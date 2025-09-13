import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from week4_final import extract_strokes, detect_edges, clock_wise_edge_numbering

def save_color_path_and_edge(stroke_mask, numbered_edges, edges, save_dir, image_name):
    """
    儲存彩色繞行圖與邊緣圖
    """
    # 彩色繞行圖
    max_num = np.max(numbered_edges)
    cmap = cm.get_cmap('jet', max_num if max_num > 0 else 1)
    color_img = np.stack([stroke_mask * 255] * 3, axis=-1).astype(np.uint8)
    for i in range(1, max_num + 1):
        points = np.where(numbered_edges == i)
        if len(points[0]) > 0:
            color = (np.array(cmap(i / max_num)[:3]) * 255).astype(np.uint8)
            for y, x in zip(points[0], points[1]):
                color_img[y, x] = color
    color_img_pil = Image.fromarray(color_img)
    color_img_pil.save(os.path.join(save_dir, f'{image_name}_colorpath.png'))
    # 邊緣圖
    edge_img = (edges * 255).astype(np.uint8)
    edge_img_pil = Image.fromarray(edge_img)
    edge_img_pil.save(os.path.join(save_dir, f'{image_name}_edge.png'))

def process_and_save_all():
    base_dir = 'handwrite'
    verify_dir = os.path.join(base_dir, 'handwrite_verification')
    os.makedirs(verify_dir, exist_ok=True)
    for digit in range(1, 10):
        for subfolder in ['database', 'testcase']:
            src_dir = os.path.join(base_dir, str(digit), subfolder)
            if not os.path.exists(src_dir):
                continue
            save_digit_dir = os.path.join(verify_dir, str(digit), subfolder)
            os.makedirs(save_digit_dir, exist_ok=True)
            bmp_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.bmp')])[:5]
            for fname in bmp_files:
                img_path = os.path.join(src_dir, fname)
                # 1. 提取筆劃
                stroke_mask = extract_strokes(img_path)
                # 2. 邊緣
                edges = detect_edges(stroke_mask)
                # 3. 繞行編號
                numbered_edges = clock_wise_edge_numbering(edges)
                # 4. 儲存彩色繞行圖與邊緣圖
                image_name = os.path.splitext(fname)[0]
                save_color_path_and_edge(stroke_mask, numbered_edges, edges, save_digit_dir, image_name)
        print(f"handwrite/{digit} 資料夾處理完成")

def process_and_save_all_numbered_edges():
    all_numbered_edges = {}
    base_dir = 'handwrite'
    for digit in range(1, 10):
        all_numbered_edges[str(digit)] = {}
        for subfolder in ['database', 'testcase']:
            src_dir = os.path.join(base_dir, str(digit), subfolder)
            if not os.path.exists(src_dir):
                continue
            all_numbered_edges[str(digit)][subfolder] = {}
            bmp_files = sorted([f for f in os.listdir(src_dir) if f.endswith('.bmp')])
            for fname in bmp_files:
                img_path = os.path.join(src_dir, fname)
                stroke_mask = extract_strokes(img_path)
                edges = detect_edges(stroke_mask)
                numbered_edges = clock_wise_edge_numbering(edges)
                key = os.path.splitext(fname)[0]
                all_numbered_edges[str(digit)][subfolder][key] = numbered_edges
        print(f"handwrite/{digit} 資料夾繞行編號處理完成")
    np.savez('all_numbered_edges.npz', **all_numbered_edges)
    print("所有字的繞行編號已儲存到 all_numbered_edges.npz")

if __name__ == "__main__":
    process_and_save_all()
    process_and_save_all_numbered_edges()
