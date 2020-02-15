import numpy as np
from random import random
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, LeakyReLU, UpSampling2D, Lambda, Add, Reshape
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras import Model

from gan import GAN
from layers import Conv2DMod


class StyleGan2(GAN):
    LATENT_DIM = 512
    CHA = 24

    def __init__(self, dataset, n_blocks, final_shape):
        GAN.__init__(self, dataset, self.LATENT_DIM)

        self.n_blocks = n_blocks
        self.final_shape = final_shape
        # Works only for 3 dimensional images (HxWxC)
        self.image_size = final_shape[0:2]  # Allows for rectangular images
        self.channels = final_shape[2]

    def _noise(self, n):
        return np.random.normal(0.0, 1.0, size=[n, self.LATENT_DIM]).astype('float32')

    def _noise_list(self, n):
        return [self._noise(n)] * self.n_blocks

    def _mixed_list(self, n):
        tt = int(random() * self.n_blocks)
        p1 = [self._noise(n)] * tt
        p2 = [self._noise(n)] * (self.n_blocks - tt)

        return p1 + [] + p2

    def _noise_images(self, n):
        return np.random.uniform(0.0, 1.0, size=[n, self.image_size[0], self.image_size[1], 1]).astype('float32')

    # Losses
    @staticmethod
    def gradient_penalty(samples, output, weight):
        gradients = K.gradients(output, samples)[0]
        gradients_sqr = K.square(gradients)
        gradient_penalty = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))

        return K.mean(gradient_penalty) * weight

    @staticmethod
    def hinge_d(y_true, y_pred):
        return K.mean(K.relu(1.0 + (y_true * y_pred)))


class Generator:
    @staticmethod
    def define_generator(latent_dim, n_blocks, cha, final_shape=(120, 120, 3)):
        generator_processing, style_network = Generator._define_generator_parts(latent_dim, n_blocks, cha, final_shape=(120, 120, 3))

        inp_style, style = [], []

        for i in range(n_blocks):
            inp_style.append(Input(shape=(latent_dim,)))
            style.append(style_network(inp_style[-1]))

        inp_noise = Input([final_shape[0], final_shape[1], 1])

        generator_output = generator_processing(style + [inp_noise])

        return Model(inputs=inp_style + [inp_noise], outputs=generator_output)

    @staticmethod
    def _define_generator_parts(latent_dim, n_blocks, cha, final_shape=(120, 120, 3)):
        # Style Mapping
        in_latent = Input(shape=(latent_dim,))
        w = Dense(latent_dim)(in_latent)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)
        w = Dense(latent_dim)(w)
        w = LeakyReLU(alpha=0.2)(w)

        style_network = Model(inputs=in_latent, outputs=w)

        # Generator

        # Input Styles
        inp_styles = []

        for i in range(n_blocks):
            inp_styles.append(Input(shape=(latent_dim,)))

        inp_noise = Input([final_shape[0], final_shape[1], 1])

        # Latent vector
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_styles[0])

        outs = []

        # Actual Model
        x = Dense(4*4*4*cha, activation='relu', kernel_initializer='random_normal')(x)
        x = Reshape([4, 4, 4*cha])(x)

        for i in range(n_blocks):
            x, r = Generator._generator_block(x, inp_styles[i], inp_noise, pow(2, n_blocks - i) * cha,
                                             final_shape, upsample=i > 0)
            outs.append(r)

        x = Add()(outs)
        x = Lambda(lambda y: y/2 + 0.5)(x)

        generator_processing = Model(inputs=inp_styles + [inp_noise], outputs=x)

        return generator_processing, style_network


    @staticmethod
    def _generator_block(inp, input_style, input_noise, filters, final_image_size, upsample=True, channels=3):
        if upsample:
            out = UpSampling2D(interpolation='bilinear')(inp)
        else:
            out = inp

        image_style = Dense(filters, kernel_initializer=VarianceScaling(200/out.shape[2]))(input_style)

        style = Dense(inp.shape[-1], kernel_initializer='he_uniform')(input_style)
        cropped_noise = Lambda(Generator.crop_noise_to_size)([input_noise, out])
        noise = Dense(filters, kernel_initializer='zeros')(cropped_noise)

        out = Conv2DMod(filters=filters, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
        out = Add()([out, noise])
        out = LeakyReLU(0.2)(out)

        style = Dense(filters, kernel_initializer='he_uniform')(input_style)
        noise = Dense(filters, kernel_initializer='zeros')(cropped_noise)

        out = Conv2DMod(filters=filters, kernel_size=3, padding='same', kernel_initializer='he_uniform')([out, style])
        out = Add()([out, noise])
        out = LeakyReLU(0.2)(out)

        return out, Generator.to_image(out, image_style, channels, final_image_size)

    @staticmethod
    def crop_noise_to_size(x):
        # x[0] is noise, x[1] is the size required
        height = x[1].shape[1]
        width = x[1].shape[2]
        return x[0][:, :height, :width, :]

    @staticmethod
    def to_image(inp, style, channels, image_size):
        # Not sure if this works for rectangular images
        x = Conv2DMod(channels, 1, kernel_initializer=VarianceScaling(200 / inp.shape[2]), demod=False)([inp, style])
        return Lambda(Generator.upsample_to_size, output_shape=[None, image_size[0], image_size[1], None])([x, image_size])

    @staticmethod
    def upsample_to_size(x):
        # x[0] is the image, x[1] is the size in HxW
        y = x[1][0] / x.shape[2]
        return K.resize_images(x, y, y, "channels_last", interpolation='bilinear')  # Scales same in both directions
