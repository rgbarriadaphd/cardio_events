"""
# Author: ruben 
# Date: 21/9/22
# Project: CardioEvents
# File: filter_images_by_size.py

Description: Filter useless images
"""
import os
import shutil

from PIL import Image

BASE_FOLDER = '/home/ruben/Documentos/Doctorado/datasets/CardiovascularEvents'
ORIG_FOLDER = os.path.join(BASE_FOLDER, 'original_retina_images')
TARGET_SIZE_FOLDER = os.path.join(BASE_FOLDER, 'filtered_by_size')

def main():

    sizes = []
    for image_name in os.listdir(ORIG_FOLDER):
        image = Image.open(os.path.join(ORIG_FOLDER, image_name))
        width, height = image.size
        sizes.append((width, height))
        size_name = f'{width}x{height}'

        size_folder = os.path.join(TARGET_SIZE_FOLDER, size_name)
        if not os.path.exists(size_folder):
            os.makedirs(size_folder)

        shutil.copy(os.path.join(ORIG_FOLDER, image_name), os.path.join(size_folder, image_name))




if __name__ == '__main__':
    main()
