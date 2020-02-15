import tensorflow as tf
from tensorflow.keras.layers import Add, Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import backend as K


class WeightedSum(Add):
    def __init__(self, alpha=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    # Merges the 2 layers, input 0 is the regular up/downscaling layer,
    # input 1 is the new convolutional layer
    def _merge_function(self, inputs):
        assert(len(inputs) == 2)
        return (1 - self.alpha) * inputs[0] + self.alpha * inputs[1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': K.get_value(self.alpha)
        })
        return config


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


class Conv2DMod(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 demod=True,
                 **kwargs):
        super(Conv2DMod, self).__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = [InputSpec(ndim=4, axes={channel_axis: input_dim}), InputSpec(ndim=2)]
        self.built = True

    def call(self, inputs):

        # To channels last
        x = tf.transpose(inputs[0], [0, 3, 1, 2])

        # Get weight and bias modulations
        # Make sure w's shape is compatible with self.kernel
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1], axis=1), axis=1), axis=-1)

        # Add minibatch layer to weights
        wo = K.expand_dims(self.kernel, axis=0)

        # Modulate
        weights = wo * (w+1)

        # Demodulate
        if self.demod:
            d = K.sqrt(K.sum(K.square(weights), axis=[1, 2, 3], keepdims=True) + 1e-8)
            weights = weights / d

        # Reshape/scale input
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])  # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        x = tf.nn.conv2d(x, w, strides=self.strides, padding="SAME", data_format="NCHW")

        # Reshape/scale output.
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]])  # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])

        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }
        base_config = super(Conv2DMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))