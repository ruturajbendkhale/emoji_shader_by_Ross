import os
import json
import numpy as np
from PIL import Image

# Define base colors (RGB)
base_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 255, 255),# White
    (128, 128, 128),# Gray
    (0, 0, 0)       # Black
]

# Generate shades
def generate_shades(color, num_shades=8):
    shades = []
    for i in range(num_shades):
        shade = tuple(int(c * (i + 1) / (num_shades + 1)) for c in color)
        shades.append(shade)
    return shades

# Function to calculate color difference
def color_difference(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

# Function to get average color of an image
def get_average_color(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGBA')
        img_array = np.array(img)
        rgb_array = img_array[:,:,:3]
        return tuple(np.mean(rgb_array, axis=(0,1)).astype(int))

# Generate color palette
color_palette = []
for base_color in base_colors:
    color_palette.extend(generate_shades(base_color))

# Directory containing emoji PNGs
emoji_dir = 'svg_downloded/png_resized'

# Find best matching emoji for each color in the palette
emoji_palette = {}
for color in color_palette:
    best_match = None
    min_difference = float('inf')
    for filename in os.listdir(emoji_dir):
        if filename.endswith('.png'):
            emoji_path = os.path.join(emoji_dir, filename)
            emoji_color = get_average_color(emoji_path)
            difference = color_difference(color, emoji_color)
            if difference < min_difference:
                min_difference = difference
                best_match = filename
    if best_match:
        emoji_palette[str(color)] = best_match

# Save the emoji palette to a JSON file
with open('emoji_palette.json', 'w') as f:
    json.dump(emoji_palette, f, indent=2)

print(f"Emoji palette saved with {len(emoji_palette)} colors.")