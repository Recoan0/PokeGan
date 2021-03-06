import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from gan import GAN
from imageloader import ImageLoader
from layers import WeightedSum, MinibatchStdDev, PixelNormalisation

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, AveragePooling2D, Flatten, Dense, UpSampling2D, Reshape
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K


class ProGAN(GAN):
    def __init__(self, dataset, n_blocks, start_shape, latent_dim):
        GAN.__init__(self, dataset, latent_dim)

        self.start_shape = start_shape
        self.discriminator_models = Discriminator.define_discriminator(n_blocks, start_shape)
        self.generator_models = Generator.define_generator(latent_dim, n_blocks, start_shape)

        self.gan_models = ProGAN._define_composite(self.discriminator_models, self.generator_models)

        self.constant_latent_vector = ProGAN._generate_latent_points(self.latent_dim, 1)

    def train(self, e_norm, e_fadein, n_batch):
        # Fit baseline model
        g_normal, d_normal, gan_normal = self.generator_models[0][0], self.discriminator_models[0][0], self.gan_models[0][0]
        gen_shape = g_normal.output_shape
        scaled_data = self._downscale_data(self.dataset, gen_shape[1:])
        print('Scaled Data shape:', scaled_data.shape)
        plt.title(f'Scaled Data {scaled_data.shape}')
        plt.imshow(scaled_data[600])
        plt.show()

        self._train_epochs(g_normal, d_normal, gan_normal, scaled_data, self.latent_dim, e_norm[0], n_batch[0])
        self._summarize_performance('tuned', g_normal, self.latent_dim)

        # Process each level of growth
        for i in range(1, len(self.generator_models)):
            g_normal, g_fadein = self.generator_models[i]
            d_normal, d_fadein = self.discriminator_models[i]
            gan_normal, gan_fadein = self.gan_models[i]

            # Scale dataset
            gen_shape = g_normal.output_shape
            scaled_data = self._downscale_data(self.dataset, gen_shape[1:])
            print('Scaled Data shape:', scaled_data.shape)
            plt.title(f'Scaled Data {scaled_data.shape}')
            plt.imshow(scaled_data[600])
            plt.show()

            # Train fade-in models
            self._train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, self.latent_dim, e_fadein[i], n_batch[i], True)
            self._summarize_performance('faded', g_fadein, self.latent_dim)

            # Train straight through models
            self._train_epochs(g_normal, d_normal, gan_normal, scaled_data, self.latent_dim, e_norm[i], n_batch[i])
            self._summarize_performance('tuned', g_normal, self.latent_dim)

            # After every level of growth, show a batch of images to see progress
            picture_amount = 16
            generated_pictures = g_normal.predict(self._generate_latent_points(self.latent_dim, picture_amount))
            self._plot_generated(generated_pictures, picture_amount)

    def _train_epochs(self, g_model, d_model, gan_model, dataset, latent_dim, n_epochs, n_batch, fadein=False):
        batches_per_epoch = int(dataset.shape[0] / n_batch)
        n_steps = batches_per_epoch * n_epochs
        half_batch = int(n_batch / 2)

        for i in range(n_steps):
            if fadein:
                ProGAN._update_fadein([g_model, d_model, gan_model], i, n_steps)

            X_real, y_real = ProGAN._generate_real_samples(dataset, half_batch)
            X_fake, y_fake = ProGAN._generate_fake_samples(g_model, latent_dim, half_batch)

            # Update discriminator
            d_loss_real = d_model.train_on_batch(X_real, y_real)
            d_loss_fake = d_model.train_on_batch(X_fake, y_fake)

            # Update generator
            z_input = ProGAN._generate_latent_points(latent_dim, n_batch)
            y_update_to_real = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(z_input, y_update_to_real)

            # Print loss this batch
            if not i % 10:
                print('>%d, d_real=%.3f, d_fake=%.3f g=%.3f' % (i + 1, d_loss_real, d_loss_fake, g_loss))

            # Show image currently generated for constant latent vector by the generator
            if not i % batches_per_epoch:
                print(f'Epoch {i // batches_per_epoch} out of {n_epochs}')
                if not i % (batches_per_epoch * 20):
                    print('Image generated!')
                    image = g_model.predict(self.constant_latent_vector)
                    self._plot_generated(image, 1)

    @staticmethod
    def _define_composite(discriminators, generators):
        model_list = []

        for i in range(len(discriminators)):
            gen_models, disc_models = generators[i], discriminators[i]

            disc_models[0].trainable = False
            straight_through_model = Sequential()
            straight_through_model.add(gen_models[0])
            straight_through_model.add(disc_models[0])
            straight_through_model.compile(loss=ProGAN.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            disc_models[1].trainable = False
            fade_in_model = Sequential()
            fade_in_model.add(gen_models[1])
            fade_in_model.add(disc_models[1])
            fade_in_model.compile(loss=ProGAN.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

            model_list.append([straight_through_model, fade_in_model])

        return model_list

    @staticmethod
    def _update_fadein(models, step, n_steps):
        alpha = step / float(n_steps - 1)
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    K.set_value(layer.alpha, alpha)


class Discriminator:
    @staticmethod
    def define_discriminator(n_blocks, input_shape=(4, 4, 3)):
        weight_init = RandomNormal(stddev=0.02)
        weight_constr = max_norm(1.0)
        model_list = []

        img_input = Input(shape=input_shape)
        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(img_input)
        d = LeakyReLU(alpha=0.2)(d)

        d = MinibatchStdDev()(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (4, 4), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Flatten()(d)
        out_class = Dense(1)(d)

        model = Model(img_input, out_class)
        model.compile(loss=ProGAN.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model, model])

        # Create sub_models
        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            new_models = Discriminator._add_discriminator_block(old_model)
            model_list.append(new_models)
        return model_list

    @staticmethod
    def _add_discriminator_block(old_model, n_input_layers=3):
        weight_init = RandomNormal(stddev=0.02)
        weight_constr = max_norm(1.0)

        old_input_shape = list(old_model.input.shape)
        # New Input shape is double size of the old input shape
        new_input_shape = (old_input_shape[-2] * 2, old_input_shape[-2] * 2, old_input_shape[-1])
        img_input = Input(shape=new_input_shape)

        # New input processing layer
        d = Conv2D(128, (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(img_input)
        d = LeakyReLU(alpha=0.2)(d)

        # New block
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D()(d)
        block_new = d

        # Add old layers of the model, skipping the input layers
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        # Model without fading, new block fully active
        straight_through_model = Model(img_input, d)
        straight_through_model.compile(loss=ProGAN.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        downsample = AveragePooling2D()(img_input)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        d = WeightedSum()([block_old, block_new])  # fade in the new block

        # Add old layers of the model, skipping the input layers
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        # Model with fading in, new block is slowly getting more influence over output
        fade_in_model = Model(img_input, d)
        fade_in_model.compile(loss=ProGAN.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        return straight_through_model, fade_in_model


class Generator:
    @staticmethod
    def define_generator(latent_dim, n_blocks, output_shape=(4, 4, 3)):
        weight_init = RandomNormal(stddev=0.02)
        weight_constr = max_norm(1.0)
        model_list = []

        in_latent = Input(shape=(latent_dim,))

        g = Dense(128 * output_shape[0] * output_shape[1], kernel_initializer=weight_init, kernel_constraint=weight_constr)(in_latent)
        g = Reshape((output_shape[0], output_shape[1], 128))(g)

        g = Conv2D(128, (4, 4), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)
        g = PixelNormalisation()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)
        g = PixelNormalisation()(g)
        g = LeakyReLU(alpha=0.2)(g)

        img_output = Conv2D(output_shape[-1], (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)

        model = Model(in_latent, img_output)

        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            new_models = Generator._add_generator_block(old_model, output_shape)
            model_list.append(new_models)

        return model_list

    @staticmethod
    def _add_generator_block(old_model, output_shape):
        weight_init = RandomNormal(stddev=0.02)
        weight_constr = max_norm(1.0)

        block_end = old_model.layers[-2].output
        upsampling = UpSampling2D()(block_end)

        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(upsampling)
        g = PixelNormalisation()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (3, 3), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)
        g = PixelNormalisation()(g)
        g = LeakyReLU(alpha=0.2)(g)

        img_output_new = Conv2D(output_shape[-1], (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)

        straight_through_model = Model(old_model.input, img_output_new)

        old_output = old_model.layers[-1]
        img_output_old = old_output(upsampling)

        merged = WeightedSum()([img_output_old, img_output_new])

        fade_in_model = Model(old_model.input, merged)

        return straight_through_model, fade_in_model


class ProGANRunner:
    USE_ALPHA = False
    SCALE_TO_128 = False

    def run(self):
        channels = 4 if self.USE_ALPHA else 3
        start_dimension = 4 if self.SCALE_TO_128 else 15
        n_blocks = 6 if self.SCALE_TO_128 else 4  # [4, 8, 16, 32 ,64 ,128] or [15, 30, 60, 120]
        n_batch = [64, 64, 32, 32, 16, 16] if self.SCALE_TO_128 else [64, 64, 32, 8]
        n_epochs = [250, 250, 500, 500, 1000, 1000] if self.SCALE_TO_128 else [250, 250, 500, 1000]

        start_shape = (start_dimension, start_dimension, channels)

        latent_dim = 100
        images_location = '/home/dustin/Documents/Projects/Python/PokeGan/images'
        dataset = ImageLoader.load_pokemon_dataset(images_location, self.USE_ALPHA, self.SCALE_TO_128)

        print(f'Loaded dataset with shape {len(dataset)}x{dataset[0].shape}')

        plt.imshow(dataset[600])
        plt.show()

        # Prevent out of memory issue
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(gpus)
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        # Select eGPU if present
        gpu_count = len(gpus)
        device = '/GPU:1' if gpu_count > 1 else '/GPU:0'

        with tf.device(device):
            gan = ProGAN(dataset, n_blocks, start_shape, latent_dim)
            gan.train(n_epochs, n_epochs, n_batch)


ProGANRunner().run()
