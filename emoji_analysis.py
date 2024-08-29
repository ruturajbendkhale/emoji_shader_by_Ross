import os
import numpy as np
from PIL import Image

def get_color_profile(image_path, num_colors=3):
    with Image.open(image_path) as img:
        img = img.convert('RGBA')
        pixels = np.array(img)
        pixels = pixels[pixels[:,:,3] > 0]  # Remove transparent pixels
        if len(pixels) == 0:
            return None  # Completely transparent image
        
        pixels = pixels[:,:3]  # Remove alpha channel
        pixels = pixels.reshape(-1, 3)
        
        # Use k-means clustering to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        
        # Get the colors and their proportions
        colors = kmeans.cluster_centers_.astype(int)
        proportions = np.bincount(kmeans.labels_) / len(pixels)
        
        # Sort colors by proportion
        color_props = sorted(zip(colors, proportions), key=lambda x: x[1], reverse=True)
        
        return color_props

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def rename_emoji_files(folder_path, num_colors=3):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            color_profile = get_color_profile(file_path, num_colors)
            
            if color_profile is None:
                print(f"Skipping {filename}: Completely transparent")
                continue
            
            new_name_parts = []
            for color, prop in color_profile:
                hex_color = rgb_to_hex(color)
                percentage = int(prop * 100)
                new_name_parts.append(f"{hex_color}_{percentage}")
            
            new_name = '_'.join(new_name_parts) + '.png'
            new_path = os.path.join(folder_path, new_name)
            
            # Handle naming conflicts
            counter = 1
            while os.path.exists(new_path):
                new_name = '_'.join(new_name_parts) + f'_{counter}.png'
                new_path = os.path.join(folder_path, new_name)
                counter += 1
            
            os.rename(file_path, new_path)
            print(f"Renamed {filename} to {new_name}")

# Usage
emoji_folder = 'svg_downloded/png'
rename_emoji_files(emoji_folder, num_colors=3)