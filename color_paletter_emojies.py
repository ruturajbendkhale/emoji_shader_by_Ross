import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import colorsys

color_palette_size = 90  # Defines the number of unique emojies in the palette

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
color_palette = natural_colors + skin_tones

# Add variations to reach 120 unique colors
while len(color_palette) < color_palette_size:
    new_variations = []
    for color in natural_colors + skin_tones:
        variations = generate_variations(color, num_variations=5)  # Increased from 3 to 5
        new_variations.extend(variations)
    
    color_palette.extend(new_variations)
    color_palette = list(set(color_palette))  # Remove duplicates
    color_palette = color_palette[:color_palette_size]  # Limit to 120 colors

# Ensure we have exactly 120 colors
if len(color_palette) < color_palette_size:
    # If we're short, add some random colors
    while len(color_palette) < color_palette_size:
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if new_color not in color_palette:
            color_palette.append(new_color)

# Check color wheel coverage
def rgb_to_hsv(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    s = 0 if mx == 0 else (diff / mx) * 100
    v = mx * 100
    return h, s, v

def check_color_wheel_coverage(colors):
    hue_bins = [0] * 12
    for color in colors:
        h, _, _ = rgb_to_hsv(color)
        bin_index = int(h / 30)
        hue_bins[bin_index] += 1
    
    print("Color wheel coverage:")
    for i, count in enumerate(hue_bins):
        print(f"Hue {i*30}-{(i+1)*30}: {count} colors")

def plot_color_wheel(colors):
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for color in colors:
        h, s, v = rgb_to_hsv(color)
        # Normalize the color values to 0-1 range
        normalized_color = tuple(c / 255 for c in color)
        ax.scatter(np.radians(h), s, c=[normalized_color], s=100, alpha=0.7)
    
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
    ax.set_title("Color Wheel Distribution")
    
    plt.savefig('color_wheel_distribution.png')
    plt.close()

# Check color wheel coverage
check_color_wheel_coverage(color_palette)

# Plot color wheel
plot_color_wheel(color_palette)

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
print("Color wheel distribution graph saved as 'color_wheel_distribution.png'.")