import os
import cv2
import json
import numpy as np
from PIL import Image
import time
import cProfile
import pstats
import io

# Global variables
emoji_palette = {}
emoji_cache = {}
color_to_emoji_cache = {}

def load_emoji_palette():
    global emoji_palette
    with open('emoji_palette.json', 'r') as f:
        emoji_palette = json.load(f)
    emoji_palette = {tuple(map(int, k[1:-1].split(','))): v for k, v in emoji_palette.items()}
    print(f"Loaded emoji palette with {len(emoji_palette)} colors.")

def load_emoji_images():
    global emoji_cache
    cell_size = 8
    for emoji_name in set(emoji_palette.values()):
        emoji_path = os.path.join('svg_downloded/png_resized', emoji_name)
        if os.path.exists(emoji_path):
            try:
                emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                emoji_img = cv2.resize(emoji_img, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
                emoji_cache[emoji_name] = emoji_img
            except Exception as e:
                print(f"Error processing {emoji_name}: {str(e)}")
    print(f"Loaded {len(emoji_cache)} emoji images into cache.")

def color_difference(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

def get_emoji_for_color(r, g, b):
    rgb = (r, g, b)
    if rgb in color_to_emoji_cache:
        return color_to_emoji_cache[rgb]
    
    # Adjust color matching to prefer warmer colors
    adjusted_rgb = (r * 1.0, g * 1.0, b * 0.9)
    
    closest_color = min(emoji_palette.keys(), key=lambda c: color_difference(adjusted_rgb, c))
    emoji = emoji_palette[closest_color]
    color_to_emoji_cache[rgb] = emoji
    return emoji

def create_emoji_grid(frame):
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    resized = cv2.resize(cropped, (360, 360), interpolation=cv2.INTER_AREA)
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    emoji_grid = []
    color_grid = np.zeros((180, 180, 3), dtype=np.uint8)
    
    for y in range(0, 360, 2):
        row = []
        for x in range(0, 360, 2):
            area = rgb_frame[y:y+2, x:x+2]
            r, g, b = np.mean(area, axis=(0, 1)).astype(int)
            emoji = get_emoji_for_color(r, g, b)
            row.append(emoji)
            color_grid[y//2, x//2] = [r, g, b]
        emoji_grid.append(row)
    
    return cropped, resized, emoji_grid, color_grid

def draw_emoji_grid(emoji_grid):
    # Get the size of the emoji grid (should be 180x180)
    grid_size = len(emoji_grid)
    # Set the size of each emoji in pixels
    cell_size = 8
    # Calculate the size of the final image (180 * 8 = 1440 pixels)
    image_size = grid_size * cell_size
    
    # Create a blank image with an alpha channel (for transparency)
    image = np.zeros((image_size, image_size, 4), dtype=np.uint8)

    # Iterate over the emoji grid and place each emoji in the image
    for y, row in enumerate(emoji_grid):
        for x, emoji_name in enumerate(row):
            if emoji_name in emoji_cache:
                # Get the pre-loaded emoji image from the cache
                emoji_img = emoji_cache[emoji_name]
                # Calculate the position to place the emoji
                pos_x = x * cell_size
                pos_y = y * cell_size
                # Place the emoji in the image
                image[pos_y:pos_y+cell_size, pos_x:pos_x+cell_size] = emoji_img

    # Convert the image from RGBA to BGR (OpenCV default color space)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

def check_emoji_colors():
    for emoji_name in list(emoji_cache.keys())[:5]:  # Check first 5 emojis
        emoji_img = emoji_cache[emoji_name]
        unique_colors = np.unique(emoji_img.reshape(-1, emoji_img.shape[2]), axis=0)
        print(f"Emoji {emoji_name} unique colors:")
        print(unique_colors)

def adjust_color_balance(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def main():
    load_emoji_palette()
    load_emoji_images()
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_times = []
    start_time = time.time()
    frame_count = 0

    check_emoji_colors()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = adjust_color_balance(frame)
        cropped, resized, emoji_grid, color_grid = create_emoji_grid(frame)
        emoji_frame = draw_emoji_grid(emoji_grid)

        # Display color grid for debugging
        cv2.imshow('Emoji Grid', cv2.cvtColor(emoji_frame, cv2.COLOR_RGB2BGR))

        current_time = time.time()
        frame_times.append(current_time)
        frame_times = [t for t in frame_times if t > current_time - 1]  # Keep only the last second
        fps = len(frame_times)
        
        cv2.putText(emoji_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Total frames processed: {frame_count}")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.2f}")

if __name__ == '__main__':
    main()
