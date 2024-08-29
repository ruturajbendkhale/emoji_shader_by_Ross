import os
import json
import numpy as np
from PIL import Image

# Define a more natural color palette
natural_colors = [
    (255, 255, 255),  # White
    (0, 0, 0),        # Black
    (128, 128, 128),  # Gray
    (255, 0, 0),      # Red
    (0, 128, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (165, 42, 42),    # Brown
    (210, 180, 140),  # Tan
    (244, 164, 96),   # Sandy brown
    (218, 165, 32),   # Goldenrod
    (0, 128, 128),    # Teal
    (0, 255, 255),    # Cyan
    (255, 192, 203),  # Pink
    (255, 127, 80),   # Coral
    (50, 205, 50),    # Lime green
]

# Fitzpatrick skin tone colors
skin_tones = [
    (255, 224, 196),  # Type I
    (241, 194, 125),  # Type II
    (224, 172, 105),  # Type III
    (198, 134, 66),   # Type IV
    (141, 85, 36),    # Type V
    (70, 39, 23)      # Type VI
]

# Function to generate shades and tints
def generate_variations(color, num_variations=2):
    variations = []
    for i in range(1, num_variations + 1):
        # Shade (darker)
        shade = tuple(int(c * (num_variations + 1 - i) / (num_variations + 1)) for c in color)
        variations.append(shade)
        # Tint (lighter)
        tint = tuple(int(c + (255 - c) * i / (num_variations + 1)) for c in color)
        variations.append(tint)
    return variations

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
color_palette = natural_colors + skin_tones

# Add variations to reach 60 colors
while len(color_palette) < 60:
    for color in natural_colors + skin_tones:
        if len(color_palette) >= 60:
            break
        color_palette.extend(generate_variations(color))

# Ensure we have exactly 60 colors
color_palette = list(set(color_palette))[:60]  # Remove duplicates and limit to 60

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