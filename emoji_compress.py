import os
from PIL import Image

def resize_emoji(input_path, output_path, size=(8, 8)):
    try:
        with Image.open(input_path) as img:
            img = img.convert('RGBA')
            img.thumbnail(size, Image.LANCZOS)
            new_img = Image.new('RGBA', size, (0, 0, 0, 0))
            new_img.paste(img, ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2), img)
            new_img.save(output_path, 'PNG')
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def resize_all_emojis(input_folder, output_folder, size=(8, 8)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_emoji(input_path, output_path, size)
            print(f"Resized {filename}")

# Usage
input_folder = 'svg_downloded/png'
output_folder = 'svg_downloded/png_resized'
resize_all_emojis(input_folder, output_folder, size=(8, 8))