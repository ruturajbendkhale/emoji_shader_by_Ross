import os
import cv2
import json
import numpy as np
from PIL import Image

# Load the emoji palette
with open('emoji_palette.json', 'r') as f:
    emoji_palette = json.load(f)

# Convert keys back to tuples
emoji_palette = {tuple(map(int, k[1:-1].split(','))): v for k, v in emoji_palette.items()}

print(f"Loaded emoji palette with {len(emoji_palette)} colors.")

# Function to find the closest color in our palette
def get_closest_color(rgb):
    return min(emoji_palette.keys(), key=lambda c: sum((a - b) ** 2 for a, b in zip(c, rgb)))

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
    
    emoji_grid = []
    for y in range(90):
        row = []
        for x in range(90):
            pixel = resized[y, x]
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cropped, resized, emoji_grid = create_emoji_grid(frame)
    emoji_frame = draw_emoji_grid(emoji_grid)

    # Display the three windows
    #cv2.imshow('Original (Cropped)', cropped)
    #cv2.imshow('Compressed (90x90)', cv2.resize(resized, (720, 720), interpolation=cv2.INTER_NEAREST))
    cv2.imshow('Emoji Grid', cv2.cvtColor(emoji_frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()