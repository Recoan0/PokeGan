import tensorflow as tf
import numpy as np

from layers import WeightedSum, MinibatchStdDev, PixelNormalisation
from keras.initializers import RandomNormal
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, LeakyReLU, AveragePooling2D, Flatten, Dense, UpSampling2D, Reshape
from keras import Model
from keras import backend as K


class GAN:
    def __init__(self, n_blocks, start_shape, latent_dim):
        self.discriminators = Discriminator.define_discriminator(n_blocks, start_shape)
        self.generators = Generator.define_generator(latent_dim, n_blocks, start_shape)


class Discriminator:
    @staticmethod
    def define_discriminator(n_blocks, input_shape=(4, 4, 1)):
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
        model.compile(loss=Discriminator.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=1e-8))
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
        new_input_shape = (old_input_shape[-2].value * 2, old_input_shape[-2].value * 2, old_input_shape[-1].value)
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
        model1 = Model(img_input, d)
        model1.compile(loss=Discriminator.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=1e-8))

        downsample = AveragePooling2D()(img_input)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        d = WeightedSum()([block_old, block_new])  # fade in the new block

        # Add old layers of the model, skipping the input layers
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        # Model with fading in, new block is slowly getting more influence over output
        model2 = Model(img_input, d)
        model2.compile(loss=Discriminator.wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=1e-8))

        return model1, model2

    @staticmethod
    def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)


class Generator:
    @staticmethod
    def define_generator(latent_dim, n_blocks, output_shape=(4, 4)):
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

        img_output = Conv2D(3, (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)

        model = Model(in_latent, img_output)

        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            new_models = Generator._add_generator_block(old_model)
            model_list.append(new_models)

        return model_list

    @staticmethod
    def _add_generator_block(old_model):
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

        img_output_new = Conv2D(3, (1, 1), padding='same', kernel_initializer=weight_init, kernel_constraint=weight_constr)(g)

        model1 = Model(old_model.input, img_output_new)

        old_output = old_model.layers[-1]
        img_output_old = old_output(upsampling)

        merged = WeightedSum()([img_output_old, img_output_new])

        model2 = Model(old_model.input, merged)

        return model1, model2
