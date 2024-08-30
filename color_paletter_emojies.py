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

def generate_variations(color, num_variations=2):
    variations = []
    for i in range(1, num_variations + 1):
        shade = tuple(int(c * (num_variations + 1 - i) / (num_variations + 1)) for c in color)
        tint = tuple(int(c + (255 - c) * i / (num_variations + 1)) for c in color)
        variations.extend([shade, tint])
    return variations

def color_difference(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))

def get_average_color(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGBA')
        img_array = np.array(img)
        rgb_array = img_array[:,:,:3]
        return tuple(np.mean(rgb_array, axis=(0,1)).astype(int))

# Generate color palette
color_palette = list(set(natural_colors + skin_tones))

# Add variations to reach 60 unique colors
while len(color_palette) < 60:
    new_variations = []
    for color in natural_colors + skin_tones:
        variations = generate_variations(color)
        new_variations.extend(variations)
    
    color_palette.extend(new_variations)
    color_palette = list(set(color_palette))  # Remove duplicates
    color_palette = color_palette[:60]  # Limit to 60 colors

# Directory containing emoji PNGs
emoji_dir = 'svg_downloded/png_resized'

# Find best matching emoji for each color in the palette
emoji_palette = {}
used_emojis = set()

for color in color_palette:
    best_match = None
    min_difference = float('inf')
    for filename in os.listdir(emoji_dir):
        if filename.endswith('.png') and filename not in used_emojis:
            emoji_path = os.path.join(emoji_dir, filename)
            emoji_color = get_average_color(emoji_path)
            difference = color_difference(color, emoji_color)
            if difference < min_difference:
                min_difference = difference
                best_match = filename
    if best_match:
        emoji_palette[str(color)] = best_match
        used_emojis.add(best_match)

# Save the emoji palette to a JSON file
with open('emoji_palette.json', 'w') as f:
    json.dump(emoji_palette, f, indent=2)

print(f"Emoji palette saved with {len(emoji_palette)} unique colors and emojis.")