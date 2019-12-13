import cv2
import os
import numpy as np


class ImageLoader:
    @staticmethod
    def load_pokemon_dataset(pokemon_images_location):
        dataset = []
        for filename in os.listdir(pokemon_images_location):
            img = cv2.imread(os.path.join(pokemon_images_location, filename))
            if img is not None:
                dataset.append(img)
        dataset_array = np.asarray(dataset)
        dataset_array.astype('float32')
        dataset_array = (dataset_array - 127.5) / 127.5
        return list(dataset_array)
