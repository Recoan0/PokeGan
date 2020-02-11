import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K


class GAN:
    def __init__(self, dataset, latent_dim):
        self.dataset = dataset
        self.latent_dim = latent_dim

    def _summarize_performance(self, status, g_model, n_samples=25):
        gen_shape = g_model.output_shape
        name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)

        X, _ = self._generate_fake_samples(g_model, self.latent_dim, n_samples)
        X = (X - X.min()) / (X.max() - X.min())

        square = int(np.sqrt(n_samples))
        for i in range(n_samples):
            plt.subplot(square, square, 1 + i)
            plt.axis('off')
            plt.imshow(X[i])

        plot_filename = f'plot_{name}.png'
        plt.savefig(plot_filename)
        plt.close()

        model_filename = f'model_{name}.h5'
        g_model.save(model_filename)

        print(f'>Saved: {plot_filename} and {model_filename}')

    @staticmethod
    def _generate_real_samples(dataset, n_samples):
        indexes = np.random.randint(0, dataset.shape[0], n_samples)
        X = dataset[indexes]
        y = np.ones((n_samples, 1))
        return X, y

    @staticmethod
    def _generate_fake_samples(generator, latent_dim, n_samples):
        x_input = GAN._generate_latent_points(latent_dim, n_samples)
        X = generator.predict(x_input)
        y = -np.ones((n_samples, 1))
        return X, y

    @staticmethod
    def _generate_latent_points(latent_dim, n_samples):
        return np.random.randn(n_samples, latent_dim)

    @staticmethod
    def _downscale_data(data_list, new_shape):
        return np.asarray(list(map(lambda data: cv2.resize(data, new_shape[0:2], cv2.INTER_AREA), data_list)))

    @staticmethod
    def _plot_generated(images, n_images):
        # plot images
        square = int(np.sqrt(n_images))
        # normalize pixel values to the range [0,1]
        images = (images - images.min()) / (images.max() - images.min())
        for i in range(n_images):
            # define subplot
            plt.subplot(square, square, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(images[i])
        plt.show()

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)
