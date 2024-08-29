import os
import json
import numpy as np
from PIL import Image

# Define primary colors (RGB)
primary_colors = [
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

# Function to generate secondary colors
def generate_secondary_colors(color1, color2, num_colors=3):
    colors = []
    for i in range(1, num_colors + 1):
        r = int(color1[0] * (num_colors + 1 - i) / (num_colors + 1) + color2[0] * i / (num_colors + 1))
        g = int(color1[1] * (num_colors + 1 - i) / (num_colors + 1) + color2[1] * i / (num_colors + 1))
        b = int(color1[2] * (num_colors + 1 - i) / (num_colors + 1) + color2[2] * i / (num_colors + 1))
        colors.append((r, g, b))
    return colors

# Function to generate shades
def generate_shades(color, num_shades=2):
    shades = []
    for i in range(1, num_shades + 1):
        shade = tuple(int(c * (num_shades + 1 - i) / (num_shades + 1)) for c in color)
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
color_palette = primary_colors.copy()

# Add secondary colors
for i in range(len(primary_colors)):
    for j in range(i+1, len(primary_colors)):
        color_palette.extend(generate_secondary_colors(primary_colors[i], primary_colors[j]))

# Add shades to reach 60 colors
while len(color_palette) < 60:
    for color in primary_colors:
        if len(color_palette) >= 60:
            break
        color_palette.extend(generate_shades(color))

# Ensure we have exactly 60 colors
color_palette = color_palette[:60]

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