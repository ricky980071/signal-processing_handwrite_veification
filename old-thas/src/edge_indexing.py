from main import stroke_extraction, compute_B_k_list
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from collections import deque

def edge_extraction(image_path):
    image = Image.open(image_path)
    stroke_image, _ = stroke_extraction(image)
    stroke_array = np.array(stroke_image)
    binary_array = (stroke_array > 0).astype(np.float32)
    B_k_list = compute_B_k_list(binary_array)
    B_xor = np.logical_xor(B_k_list[0], B_k_list[1])
    return B_xor

def find_connected_components(edge_image):
    """
    找到筆畫邊緣圖中的連通區塊
    如果兩個pixel的x座標差和y座標差都在1以內，則屬於同一個區塊
    """
    height, width = edge_image.shape
    visited = np.zeros((height, width), dtype=bool)
    components = []
    
    # 8-connected directions
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    for i in range(height):
        for j in range(width):
            if edge_image[i, j] and not visited[i, j]:
                # BFS to find connected component
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    y, x = queue.popleft()
                    component.append((y, x))
                    
                    # Check 8 neighbors
                    for dy, dx in directions:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            edge_image[ny, nx] and not visited[ny, nx]):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                
                if component:
                    components.append(component)
    
    return components

def sort_components_by_leftmost(components):
    """
    將區塊從最左開始編號
    比較誰的最左邊筆畫邊緣pixel比較左邊
    """
    def get_leftmost_x(component):
        return min(x for y, x in component)
    
    # Sort components by leftmost x coordinate
    sorted_components = sorted(components, key=get_leftmost_x)
    return sorted_components

def trace_pixels_in_component(component, edge_image):
    """
    對區塊內的筆畫pixel進行編號
    使用深度優先搜索但是改進處理未訪問像素的邏輯
    """
    if not component:
        return []
    
    # Find leftmost pixel (if multiple, choose the topmost one)
    leftmost_pixels = [pixel for pixel in component if pixel[1] == min(x for y, x in component)]
    start_pixel = min(leftmost_pixels)  # Choose topmost among leftmost
    
    # 8-connected directions (starting from top, going clockwise)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    
    component_set = set(component)
    visited = set()
    traced_pixels = []
    
    def dfs(pixel):
        if pixel in visited or pixel not in component_set:
            return
        
        visited.add(pixel)
        traced_pixels.append(pixel)
        
        # Debug print for specific pixels around the problematic area
        # if len(traced_pixels) in [340, 341, 342, 343, 344, 345]:
        #     print(f"  Pixel #{len(traced_pixels)}: {pixel}")
        
        y, x = pixel
        # Sort neighbors to ensure consistent ordering
        neighbors = []
        for dy, dx in directions:
            next_pixel = (y + dy, x + dx)
            if next_pixel in component_set and next_pixel not in visited:
                neighbors.append(next_pixel)
        
        # Sort by distance, then by position for consistency
        neighbors.sort(key=lambda p: (abs(p[0]-y) + abs(p[1]-x), p[0], p[1]))
        
        # if len(traced_pixels) in [341, 342]:  # Debug neighbors for problematic pixels
        #     print(f"    From {pixel}, neighbors: {neighbors}")
        
        for next_pixel in neighbors:
            if next_pixel not in visited:
                dfs(next_pixel)
    
    print(f"Starting DFS from leftmost pixel: {start_pixel}")
    dfs(start_pixel)
    
    # If there are still unvisited pixels, find the closest one to the last traced pixel
    while len(visited) < len(component):
        unvisited_pixels = [p for p in component if p not in visited]
        if not unvisited_pixels:
            break
            
        # Find the closest unvisited pixel to the last traced pixel
        last_pixel = traced_pixels[-1] if traced_pixels else start_pixel
        closest_pixel = min(unvisited_pixels, 
                          key=lambda p: abs(p[0] - last_pixel[0]) + abs(p[1] - last_pixel[1]))
        
        print(f"Found {len(unvisited_pixels)} unvisited pixels, continuing from closest: {closest_pixel}")
        print(f"  Distance from last pixel {last_pixel}: {abs(closest_pixel[0] - last_pixel[0]) + abs(closest_pixel[1] - last_pixel[1])}")
        
        dfs(closest_pixel)
    
    return traced_pixels

def edge_indexing(image_path):
    """
    完整的邊緣索引流程：
    1. 提取邊緣
    2. 找到連通區塊
    3. 對區塊排序
    4. 對每個區塊內的像素進行追蹤編號
    """
    # Step 1: Extract edges
    edge_image = edge_extraction(image_path)
    
    # Step 2: Find connected components
    components = find_connected_components(edge_image)
    
    # Step 3: Sort components by leftmost pixel
    sorted_components = sort_components_by_leftmost(components)
    
    # Step 4: Trace pixels in each component
    indexed_components = []
    for i, component in enumerate(sorted_components):
        traced_pixels = trace_pixels_in_component(component, edge_image)
        indexed_components.append({
            'component_id': i + 1,
            'pixels': component,
            'traced_pixels': traced_pixels
        })
    
    return edge_image, indexed_components
def visualize_tracing_with_slider(edge_image, indexed_components):
    """
    使用進度條視覺化像素追蹤過程，包含自動播放功能
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create main subplot for the image
    ax_main = plt.subplot2grid((5, 3), (0, 0), colspan=2, rowspan=3)
    
    # Create subplot for component selection
    ax_comp = plt.subplot2grid((5, 3), (0, 2))
    
    # Create subplot for slider
    ax_slider = plt.subplot2grid((5, 3), (3, 0), colspan=3)
    
    # Create subplot for control buttons
    ax_play = plt.subplot2grid((5, 3), (4, 0))
    ax_speed = plt.subplot2grid((5, 3), (4, 1))
    ax_reset = plt.subplot2grid((5, 3), (4, 2))
    
    height, width = edge_image.shape
    
    # Initialize display image
    display_image = np.zeros((height, width, 3))
    
    # Different colors for different components
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    
    # Show all components in light gray initially
    for i, comp_data in enumerate(indexed_components):
        for y, x in comp_data['pixels']:
            display_image[y, x] = [0.3, 0.3, 0.3]  # Light gray
    
    im = ax_main.imshow(display_image)
    ax_main.set_title('Pixel Tracing Visualization')
    ax_main.axis('off')
    
    # Component info display
    ax_comp.axis('off')
    comp_text = ax_comp.text(0, 0.5, 'Select component\nwith slider', 
                            transform=ax_comp.transAxes, fontsize=12, 
                            verticalalignment='center')
    
    # Find max traced pixels for slider range
    total_pixels = sum(len(comp['traced_pixels']) for comp in indexed_components)
    
    # Create slider
    slider = Slider(ax_slider, 'Progress', 0, total_pixels, 
                    valinit=0, valstep=1, valfmt='%d')
    
    # Create control buttons
    play_button = Button(ax_play, 'Play')
    speed_button = Button(ax_speed, 'Speed: 1x')
    reset_button = Button(ax_reset, 'Reset')
    
    # Animation control variables
    is_playing = False
    play_speed = 1  # pixels per update
    speed_levels = [1, 2, 5, 10, 20]
    current_speed_idx = 0
    timer = None
    
    def update_display(val):
        progress = int(slider.val)
        
        # Reset display image
        display_image[:] = 0
        
        # Show all components in light gray
        for i, comp_data in enumerate(indexed_components):
            for y, x in comp_data['pixels']:
                display_image[y, x] = [0.3, 0.3, 0.3]
        
        # Determine which component and pixel index
        current_comp_idx = 0
        remaining_progress = progress
        current_pixel_idx = 0
        
        # Find current component and pixel
        for i, comp_data in enumerate(indexed_components):
            traced_count = len(comp_data['traced_pixels'])
            if remaining_progress <= traced_count:
                current_comp_idx = i
                current_pixel_idx = remaining_progress
                break
            else:
                remaining_progress -= traced_count
        
        if current_comp_idx < len(indexed_components):
            comp_data = indexed_components[current_comp_idx]
            traced_pixels = comp_data['traced_pixels']
            color = colors[current_comp_idx % len(colors)]
            
            # Show all previous components completely
            for i in range(current_comp_idx):
                prev_comp = indexed_components[i]
                prev_color = colors[i % len(colors)]
                for y, x in prev_comp['traced_pixels']:
                    display_image[y, x] = prev_color
            
            # Show traced pixels up to current progress in current component
            for i in range(current_pixel_idx):
                y, x = traced_pixels[i]
                # Gradually increase brightness for newer pixels
                brightness = 0.6 + 0.4 * (i + 1) / len(traced_pixels)
                display_image[y, x] = [c * brightness for c in color]
            
            # Highlight current pixel with white border
            if current_pixel_idx > 0 and current_pixel_idx <= len(traced_pixels):
                current_y, current_x = traced_pixels[current_pixel_idx - 1]
                # Add white border around current pixel
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = current_y + dy, current_x + dx
                        if (0 <= ny < height and 0 <= nx < width and 
                            (dy == 0 or dx == 0)):  # Cross pattern
                            if display_image[ny, nx].sum() < 1.5:  # Only if it's not too bright
                                display_image[ny, nx] = [1, 1, 1]  # White
            
            # Update component info
            info_text = f"Component {comp_data['component_id']}\n"
            info_text += f"Total pixels: {len(comp_data['pixels'])}\n"
            info_text += f"Traced pixels: {len(traced_pixels)}\n"
            if current_pixel_idx > 0:
                info_text += f"Current pixel: #{current_pixel_idx}\n"
                current_pixel = traced_pixels[current_pixel_idx - 1]
                info_text += f"Position: {current_pixel}\n"
            
            # Show next few pixels preview
            if current_pixel_idx < len(traced_pixels):
                next_pixels = traced_pixels[current_pixel_idx:current_pixel_idx+3]
                info_text += f"Next: {next_pixels}"
            
            comp_text.set_text(info_text)
        
        im.set_array(display_image)
        ax_main.set_title(f'Component {current_comp_idx + 1}/{len(indexed_components)} - Pixel #{current_pixel_idx}')
        fig.canvas.draw()
    
    def animate():
        """Animation function for auto-play"""
        nonlocal timer
        if is_playing and slider.val < total_pixels:
            new_val = min(slider.val + play_speed, total_pixels)
            slider.set_val(new_val)
            if new_val < total_pixels:
                # Schedule next update
                timer = fig.canvas.new_timer(interval=50)  # 50ms = 20 FPS
                timer.single_shot = True
                timer.add_callback(animate)
                timer.start()
            else:
                # Animation finished
                toggle_play(None)
    
    def toggle_play(event):
        """Toggle play/pause"""
        nonlocal is_playing, timer
        is_playing = not is_playing
        
        if is_playing:
            play_button.label.set_text('Pause')
            if slider.val >= total_pixels:
                slider.set_val(0)  # Reset if at end
            animate()
        else:
            play_button.label.set_text('Play')
            if timer:
                timer.stop()
    
    def change_speed(event):
        """Change playback speed"""
        nonlocal current_speed_idx, play_speed
        current_speed_idx = (current_speed_idx + 1) % len(speed_levels)
        play_speed = speed_levels[current_speed_idx]
        speed_button.label.set_text(f'Speed: {play_speed}x')
        fig.canvas.draw()
    
    def reset_animation(event):
        """Reset animation to beginning"""
        nonlocal is_playing, timer
        if timer:
            timer.stop()
        is_playing = False
        play_button.label.set_text('Play')
        slider.set_val(0)
    
    # Connect events
    slider.on_changed(update_display)
    play_button.on_clicked(toggle_play)
    speed_button.on_clicked(change_speed)
    reset_button.on_clicked(reset_animation)
    
    # Initial update
    update_display(0)
    
    plt.tight_layout()
    return fig, slider

if __name__ == "__main__":
    test_image_path = '../data/1/database/base_1_1_1.bmp'
    # test_image_path = '../data/2/database/base_1_1_2.bmp'
    # test_image_path = '../data/3/database/base_1_1_3.bmp'
    # test_image_path = '../data/4/database/base_1_1_4.bmp'
    # test_image_path = '../data/5/database/base_1_1_5.bmp'
    # test_image_path = '../data/6/database/base_1_1_6.bmp'
    # test_image_path = '../data/7/database/base_1_1_7.bmp'
    # test_image_path = '../data/8/database/base_1_1_8.bmp'
    # test_image_path = '../data/9/database/base_1_1_9.bmp'
    
    # Test edge extraction
    edge_image = edge_extraction(test_image_path)
    
    # Test full edge indexing
    edge_result, indexed_components = edge_indexing(test_image_path)
    
    # Print component information
    print(f"Found {len(indexed_components)} components:")
    for comp_data in indexed_components:
        comp_id = comp_data['component_id']
        pixel_count = len(comp_data['pixels'])
        traced_count = len(comp_data['traced_pixels'])
        leftmost_x = min(x for y, x in comp_data['pixels'])
        print(f"Component {comp_id}: {pixel_count} pixels, traced {traced_count} pixels, leftmost position: x={leftmost_x}")
        
        # Print first few traced pixels
        if traced_count > 0:
            print(f"  Tracing order first 5 pixels: {comp_data['traced_pixels'][:5]}")
        
        # Special debug for component 3
        if comp_id == 3:
            print(f"  Component 3 detailed analysis:")
            traced_pixels = comp_data['traced_pixels']
            for i in range(340, min(346, len(traced_pixels))):
                if i < len(traced_pixels):
                    curr_pixel = traced_pixels[i]
                    print(f"    Pixel #{i+1}: {curr_pixel}")
                    # Check if next pixel is adjacent
                    if i+1 < len(traced_pixels):
                        next_pixel = traced_pixels[i+1]
                        y1, x1 = curr_pixel
                        y2, x2 = next_pixel
                        distance = max(abs(y2-y1), abs(x2-x1))
                        print(f"    Distance to next pixel #{i+2} {next_pixel}: {distance}")
                        if distance > 1:
                            print(f"    WARNING: Distance > 1! Not adjacent!")
        print()
    
    # Print detailed tracing for first component to debug
    if indexed_components:
        first_comp = indexed_components[0]
        print(f"Detailed tracing for Component 1:")
        print(f"All component pixels: {sorted(first_comp['pixels'])}")
        print(f"Traced pixels sequence (first 20): {first_comp['traced_pixels'][:20]}")
        print(f"Total traced: {len(first_comp['traced_pixels'])}")
        print()
    
    # Show interactive visualization with slider
    print("Showing interactive visualization interface...")
    print("Controls:")
    print("- Drag slider to manually view pixel numbering sequence")
    print("- Click 'Play' to start/pause auto-play animation")
    print("- Click 'Speed' to change playback speed (1x, 2x, 5x, 10x, 20x)")
    print("- Click 'Reset' to return to the beginning")
    print("Watch pixels gradually appear to form the stroke outline!")
    
    fig, slider = visualize_tracing_with_slider(edge_result, indexed_components)
    plt.show()