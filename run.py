import tensorflow as tf

from progan import GAN
from imageloader import ImageLoader
from matplotlib import pyplot as plt

USE_ALPHA = False
SCALE_TO_128 = False

channels = 4 if USE_ALPHA else 3
start_dimension = 4 if SCALE_TO_128 else 15
n_blocks = 6 if SCALE_TO_128 else 4  # [4, 8, 16, 32 ,64 ,128] or [15, 30, 60, 120]
n_batch = [64, 64, 32, 32, 16, 16] if SCALE_TO_128 else [64, 64, 32, 16]
n_epochs = [250, 250, 500, 500, 1000, 1000] if SCALE_TO_128 else [250, 250, 500, 1000]

start_shape = (start_dimension, start_dimension, channels)

latent_dim = 100
images_location = '/home/dustin/Documents/Projects/Python/PokeGan/images'
dataset = ImageLoader.load_pokemon_dataset(images_location, USE_ALPHA, SCALE_TO_128)

print(f'Loaded dataset with shape {len(dataset)}x{dataset[0].shape}')

plt.imshow(dataset[600])
plt.show()

# Prevent out of memory issue
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

gan = GAN(dataset, n_blocks, start_shape, latent_dim)
gan.train(n_epochs, n_epochs, n_batch)
