import numpy as np
from random import random

from gan import GAN

class StyleGan2(GAN):
    LATENT_DIM = 512

    def __init__(self, dataset, image_size, n_blocks):
        GAN.__init__(self, dataset, self.LATENT_DIM)

        self.n_blocks = n_blocks

    def _noise(self, n):
        return np.random.normal(0.0, 1.0, size=[n, self.LATENT_DIM]).astype('float32')

    def _noise_list(self, n):
        return [self._noise(n)] * self.n_blocks

    def _mixed_list(self, n):
        tt = int(random() * self.n_blocks)
        p1 = [self._noise(n)] * tt
        p2 = [self._noise(n)] * (self.n_blocks - tt)


