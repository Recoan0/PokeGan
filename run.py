from progan import GAN
from imageloader import ImageLoader

n_blocks = 4  # Sizes = [15, 30, 60, 120]
start_shape = (15, 15, 3)  # Start shape of 15x15 in rgb
latent_dim = 100
images_location = '/home/dustin/Documents/Projects/Python/PokeGan/images'
dataset = ImageLoader.load_pokemon_dataset(images_location)

print(f'Loaded dataset with shape {len(dataset)}x{dataset[0].shape}')

n_batch = [64, 64, 32, 16]
n_epochs = [100, 100, 300, 500]
gan = GAN(dataset, n_blocks, start_shape, latent_dim)
gan.train(n_epochs, n_epochs, n_batch)
