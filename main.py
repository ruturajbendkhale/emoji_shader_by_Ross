import os
import cv2
import json
import numpy as np
from PIL import Image
import time
import cProfile
import pstats
import io
from scipy.spatial import cKDTree  # Using cKDTree for sqeuclidean metric

# Global variables
emoji_palette = {}
emoji_cache = {}
color_to_emoji_cache = {}
palette_kdtree = None
palette_colors = None
palette_emojis = None

def load_emoji_palette():
    global emoji_palette, palette_kdtree, palette_colors, palette_emojis
    with open('emoji_palette.json', 'r') as f:
        emoji_palette = json.load(f)
    emoji_palette = {tuple(map(int, k[1:-1].split(','))): v for k, v in emoji_palette.items()}
    
    # Create optimized data structures for fast color lookup
    palette_colors = np.array(list(emoji_palette.keys()), dtype=np.float32)  # Use float32 for precision
    palette_emojis = np.array(list(emoji_palette.values()))
    palette_kdtree = cKDTree(palette_colors, leafsize=10)  # Use cKDTree with default euclidean first
    
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
    
    # Adjust color matching to prefer warmer colors (prevents bluish tint)
    adjusted_rgb = np.array([r * 1.0, g * 1.0, b * 0.9])
    
    # Use KDTree for fast nearest neighbor search
    _, closest_idx = palette_kdtree.query(adjusted_rgb)
    emoji = palette_emojis[closest_idx]
    color_to_emoji_cache[rgb] = emoji
    return emoji

def get_emojis_for_colors_vectorized(rgb_array):
    """Highly optimized vectorized version with efficient caching"""
    # Get array dimensions
    height, width = rgb_array.shape[:2]
    total_pixels = height * width
    
    # Flatten the array for processing
    flattened_colors = rgb_array.reshape(-1, 3)
    
    # Pre-allocate result array
    emoji_names = np.empty(total_pixels, dtype=object)
    
    # Convert colors to tuples for cache lookup (vectorized)
    color_tuples = [tuple(map(int, color)) for color in flattened_colors]
    
    # Separate cached and uncached colors efficiently
    uncached_indices = []
    uncached_colors_list = []
    
    for i, color_tuple in enumerate(color_tuples):
        if color_tuple in color_to_emoji_cache:
            emoji_names[i] = color_to_emoji_cache[color_tuple]
        else:
            uncached_indices.append(i)
            uncached_colors_list.append(flattened_colors[i])
    
    # Process uncached colors in batch if any exist
    if uncached_indices:
        uncached_colors = np.array(uncached_colors_list)
        
        # Apply color adjustment vectorized: (r * 1.0, g * 1.0, b * 0.9)
        adjusted_colors = uncached_colors.astype(np.float32)
        adjusted_colors[:, 2] *= 0.9  # Reduce blue component
        
        # Vectorized KDTree query for all uncached colors
        _, closest_indices = palette_kdtree.query(adjusted_colors)
        
        # Get corresponding emojis and update cache
        for i, (idx, closest_idx, original_color) in enumerate(zip(uncached_indices, closest_indices, uncached_colors)):
            emoji = palette_emojis[closest_idx]
            color_tuple = tuple(map(int, original_color))
            color_to_emoji_cache[color_tuple] = emoji
            emoji_names[idx] = emoji
    
    # Reshape back to original grid shape
    emoji_grid = emoji_names.reshape(height, width)
    return emoji_grid

def create_emoji_grid(frame):
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to 180x180 for processing
    resized = cv2.resize(cropped, (180, 180), interpolation=cv2.INTER_AREA)
    # Convert BGR to RGB once
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Vectorized downsampling using array slicing - much faster!
    # Take every 2nd pixel in both dimensions to get 90x90 grid
    downsampled = rgb_frame[::2, ::2]  # This gives us 90x90 directly
    
    # Further optimization: average 2x2 blocks for better color representation
    # Reshape to process 2x2 blocks efficiently
    reshaped = rgb_frame.reshape(90, 2, 90, 2, 3)
    downsampled = reshaped.mean(axis=(1, 3)).astype(np.uint8)
    
    # Use optimized vectorized emoji matching
    emoji_grid = get_emojis_for_colors_vectorized(downsampled)
    
    return cropped, resized, emoji_grid, downsampled

def draw_emoji_grid(emoji_grid):
    # Get the size of the emoji grid (should be 90x90)
    grid_height, grid_width = emoji_grid.shape
    # Set the size of each emoji in pixels
    cell_size = 8
    # Calculate the size of the final image
    image_height = grid_height * cell_size
    image_width = grid_width * cell_size
    
    # Create a blank image with 4 channels (RGBA)
    image = np.zeros((image_height, image_width, 4), dtype=np.uint8)

    # Optimized emoji placement using vectorized operations
    for y in range(grid_height):
        for x in range(grid_width):
            emoji_name = emoji_grid[y, x]
            if emoji_name in emoji_cache:
                # Get the pre-loaded emoji image from the cache
                emoji_img = emoji_cache[emoji_name]
                # Calculate the position to place the emoji
                pos_y = y * cell_size
                pos_x = x * cell_size
                # Place the emoji in the image using array slicing
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

def debug_color_matching_detailed(test_color):
    """Debug function to compare original algorithm vs KDTree results"""
    r, g, b = test_color
    print(f"\n=== DEBUGGING COLOR: ({r}, {g}, {b}) ===")
    
    # Original method calculation
    adjusted_rgb_original = (r * 1.0, g * 1.0, b * 0.9)
    print(f"Original adjusted color: {adjusted_rgb_original}")
    
    # Find closest using original method (brute force)
    min_diff = float('inf')
    best_color_original = None
    best_emoji_original = None
    
    for palette_color, emoji in emoji_palette.items():
        diff = color_difference(adjusted_rgb_original, palette_color)
        if diff < min_diff:
            min_diff = diff
            best_color_original = palette_color
            best_emoji_original = emoji
    
    print(f"Original method: closest color {best_color_original} -> {best_emoji_original}")
    print(f"Original method distance: {min_diff}")
    
    # KDTree method calculation
    adjusted_rgb_kdtree = np.array([r * 1.0, g * 1.0, b * 0.9])
    distance, closest_idx = palette_kdtree.query(adjusted_rgb_kdtree)
    best_color_kdtree = tuple(palette_colors[closest_idx])
    best_emoji_kdtree = palette_emojis[closest_idx]
    
    print(f"KDTree method: closest color {best_color_kdtree} -> {best_emoji_kdtree}")
    print(f"KDTree method distance: {distance}")
    
    # Check if they match
    if best_emoji_original == best_emoji_kdtree:
        print("✅ MATCH: Both methods return same emoji")
    else:
        print("❌ MISMATCH: Different emojis selected!")
        print("This explains the color difference!")
    
    print("=== END DEBUG ===\n")

def main():
    load_emoji_palette()
    load_emoji_images()
    
    # DEBUG: Test color matching for representative colors
    print("=== TESTING COLOR MATCHING ACCURACY ===")
    test_colors = [
        [255, 180, 42],   # Orange/yellow - should not be blue
        [200, 100, 50],   # Brown/orange - should not be blue  
        [150, 150, 150],  # Gray - should not be blue
        [100, 200, 80],   # Green - should not be blue
        [50, 100, 200],   # Blue - this should be blue
    ]
    
    for test_color in test_colors:
        debug_color_matching_detailed(test_color)
    
    print("=== STARTING CAMERA ===")
    
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

        # Display emoji grid
        cv2.imshow('Emoji Grid', emoji_frame)

        current_time = time.time()
        frame_times.append(current_time)
        frame_times = [t for t in frame_times if t > current_time - 1]  # Keep only the last second
        fps = len(frame_times)
        
        # Add FPS counter to the emoji frame
        cv2.putText(emoji_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emoji Grid', emoji_frame)

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
