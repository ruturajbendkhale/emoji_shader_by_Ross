import os
import cv2
import json
import numpy as np
from PIL import Image
import time

# Load the emoji palette
with open('emoji_palette.json', 'r') as f:
    emoji_palette = json.load(f)

# Convert keys back to tuples
emoji_palette = {tuple(map(int, k[1:-1].split(','))): v for k, v in emoji_palette.items()}

print(f"Loaded emoji palette with {len(emoji_palette)} colors.")

# Function to convert RGB to Lab color space
def rgb_to_lab(rgb):
    r, g, b = rgb
    r, g, b = r/255.0, g/255.0, b/255.0
    
    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92
    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92
    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    x = x ** (1/3) if x > 0.008856 else (7.787 * x) + 16/116
    y = y ** (1/3) if y > 0.008856 else (7.787 * y) + 16/116
    z = z ** (1/3) if z > 0.008856 else (7.787 * z) + 16/116

    L = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    return L, a, b

# Function to calculate color difference
def color_difference(rgb1, rgb2):
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return sum((c1 - c2) ** 2 for c1, c2 in zip(lab1, lab2)) ** 0.5

# Function to find the closest color in our palette
def get_closest_color(rgb):
    return min(emoji_palette.keys(), key=lambda c: color_difference(rgb, c))

# Function to get the emoji for a given color
def get_emoji_for_color(rgb):
    closest_color = get_closest_color(rgb)
    return emoji_palette[closest_color]

# Create emoji grid from webcam input
def create_emoji_grid(frame):
    # Crop to 720x720
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to 90x90
    resized = cv2.resize(cropped, (90, 90), interpolation=cv2.INTER_AREA)
    
    # Enhance contrast and color
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Increase saturation
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)  # Increase saturation by 30
    hsv_enhanced = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    emoji_grid = []
    for y in range(90):
        row = []
        for x in range(90):
            pixel = enhanced[y, x]
            emoji = get_emoji_for_color((int(pixel[2]), int(pixel[1]), int(pixel[0])))  # BGR to RGB
            row.append(emoji)
        emoji_grid.append(row)
    
    return cropped, resized, emoji_grid

# Function to draw emoji grid
def draw_emoji_grid(emoji_grid):
    grid_size = len(emoji_grid)
    cell_size = 8  # Size of each emoji
    image_size = grid_size * cell_size
    image = Image.new('RGB', (image_size, image_size), (0, 0, 0))  # Black background

    for y, row in enumerate(emoji_grid):
        for x, emoji_name in enumerate(row):
            emoji_path = os.path.join('svg_downloded/png_resized', emoji_name)
            if os.path.exists(emoji_path):
                try:
                    emoji_img = Image.open(emoji_path).convert('RGBA')
                    emoji_img = emoji_img.resize((cell_size, cell_size), Image.LANCZOS)
                    pos_x = x * cell_size
                    pos_y = y * cell_size
                    image.paste(emoji_img, (pos_x, pos_y), emoji_img)
                except Exception as e:
                    print(f"Error processing {emoji_name}: {str(e)}")

    return np.array(image)

# Main webcam loop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_times = []
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cropped, resized, emoji_grid = create_emoji_grid(frame)
    emoji_frame = draw_emoji_grid(emoji_grid)

    # Calculate and display FPS
    current_time = time.time()
    frame_times.append(current_time)
    frame_times = [t for t in frame_times if t > current_time - 1]  # Keep only the last second
    fps = len(frame_times)
    
    cv2.putText(cropped, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(emoji_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the three windows
    #cv2.imshow('Original (Cropped)', cropped)
    #cv2.imshow('Compressed (90x90)', cv2.resize(resized, (720, 720), interpolation=cv2.INTER_NEAREST))
    cv2.imshow('Emoji Grid', cv2.cvtColor(emoji_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Average FPS: {len(frame_times) / (time.time() - start_time):.2f}")