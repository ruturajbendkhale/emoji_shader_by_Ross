import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import colorsys

color_palette_size = 90  # Defines the number of unique emojies in the palette

# Define a more natural color palette with emphasis on warm tones to avoid blue bias
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
    # Additional warm tones
    (255, 200, 100),  # Light orange
    (200, 150, 100),  # Light brown
    (150, 100, 50),   # Dark orange
    (100, 50, 20),    # Deep brown
]

# Fitzpatrick skin tone colors (emphasizing warm tones)
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
    # Use weighted Euclidean distance with bias against blue (reduce blue weight)
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return (r1 - r2)**2 + (g1 - g2)**2 + 0.8 * (b1 - b2)**2  # Downweight blue difference

def get_average_color(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGBA')
        img_array = np.array(img)
        mask = img_array[:, :, 3] > 0  # Ignore transparent pixels
        if not np.any(mask):
            return (0, 0, 0)  # Fallback for fully transparent
        rgb_array = img_array[mask, :3]
        # Weighted average: give more weight to warmer pixels (higher red/orange)
        weights = rgb_array[:, 0] + 0.5 * rgb_array[:, 1]  # Bias towards red/green (warm)
        weighted_rgb = np.average(rgb_array, axis=0, weights=weights)
        return tuple(weighted_rgb.astype(int))

# Generate color palette with more warm variations
color_palette = natural_colors + skin_tones

# Add more variations, especially for warm colors
while len(color_palette) < color_palette_size:
    new_variations = []
    for color in natural_colors + skin_tones:
        variations = generate_variations(color, num_variations=5)
        new_variations.extend(variations)
    
    color_palette.extend(new_variations)
    color_palette = list(set(color_palette))  # Remove duplicates
    color_palette = color_palette[:color_palette_size]  # Limit to 90 colors

# If short, add random warm-biased colors
if len(color_palette) < color_palette_size:
    while len(color_palette) < color_palette_size:
        new_color = (random.randint(100, 255), random.randint(50, 200), random.randint(0, 100))  # Bias to warm (high red, low blue)
        if new_color not in color_palette:
            color_palette.append(new_color)

# ... (rest of the original code remains the same - check coverage, plot, match emojis, save JSON)