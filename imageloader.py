import cv2
import os
import numpy as np


class ImageLoader:
    @staticmethod
    def load_pokemon_dataset(pokemon_images_location, load_alpha, scale_to_128):
        dataset = []
        for filename in os.listdir(pokemon_images_location):
            file_path = os.path.join(pokemon_images_location, filename)
            img = ImageLoader.load_rgba(file_path) if load_alpha else ImageLoader.load_rgb(file_path)
            if scale_to_128:
                img = cv2.resize(img, (128, 128), cv2.INTER_CUBIC)  # Scale up to 128x128
            dataset.append(img)
        dataset_array = np.asarray(dataset)
        dataset_array.astype('float32')
        dataset_array = (dataset_array - 127.5) / 127.5
        return list(dataset_array)

    @staticmethod
    def load_rgb(location):
        img = cv2.imread(location)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def load_rgba(location):
        img = cv2.imread(location, cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
