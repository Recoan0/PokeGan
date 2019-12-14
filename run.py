from progan import GAN
from imageloader import ImageLoader
from matplotlib import pyplot as plt

n_blocks = 4  # Sizes = [15, 30, 60, 120]
start_shape = (15, 15, 4)  # Start shape of 15x15 in rgb
latent_dim = 100
images_location = '/home/dustin/Documents/Projects/Python/PokeGan/images'
dataset = ImageLoader.load_pokemon_dataset(images_location)

print(f'Loaded dataset with shape {len(dataset)}x{dataset[0].shape}')

plt.imshow(dataset[600])
plt.show()

n_batch = [64, 64, 128, 256]
n_epochs = [250, 250, 500, 1000]
gan = GAN(dataset, n_blocks, start_shape, latent_dim)
gan.train(n_epochs, n_epochs, n_batch)
