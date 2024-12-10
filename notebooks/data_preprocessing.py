import os
import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_data(raw_dir, processed_dir):
    """
    Preprocesa imágenes del directorio crudo y las guarda en processed_dir.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for category in os.listdir(raw_dir):
        category_path = os.path.join(raw_dir, category)
        output_path = os.path.join(processed_dir, category)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(output_path, img_name), img)
    print("Preprocesamiento completado.")