from keras.layers import Add, Layer
from keras import backend as K


class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    # Merges the 2 layers, input 0 is the regular up/downscaling layer,
    # input 1 is the new convolutional layer
    def _merge_function(self, inputs):
        assert(len(inputs) == 2)
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]


class MinibatchStdDev(Layer):
    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=0, keepdims=True)
        square_diffs = K.square(inputs - mean)
        sq_diffs_mean = K.mean(square_diffs, axis=0, keepdims=True) + 1e-8
        stdev = K.sqrt(sq_diffs_mean)
        mean_pix = K.mean(stdev, keepdims=True)
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        return K.concatenate([inputs, output], axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class PixelNormalisation(Layer):
    def call(self, inputs, **kwargs):
        values = inputs ** 2
        mean_values = K.mean(values, axis=-1, keepdims=True) + 1e-8
        l2 = K.sqrt(mean_values)
        return inputs / l2

    def compute_output_shape(self, input_shape):
        return input_shape
