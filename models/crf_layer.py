from tensorflow.python.framework import ops
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

custom_module = tf.load_op_library(os.path.join(
    os.path.dirname(__file__), 'cpp', 'high_dim_filter.so'))


@ops.RegisterGradient('HighDimFilter')
def _high_dim_filter_grad(op, grad):
    rgb = op.inputs[1]
    grad_vals = custom_module.high_dim_filter(grad, rgb,
                                              bilateral=op.get_attr(
                                                  'bilateral'),
                                              theta_alpha=op.get_attr(
                                                  'theta_alpha'),
                                              theta_beta=op.get_attr(
                                                  'theta_beta'),
                                              theta_gamma=op.get_attr(
                                                  'theta_gamma'),
                                              backwards=True)


def _diagonal_initializer(shape, *ignored, **ignored_too):
    return np.eye(shape[0], shape[1], dtype=np.float32)


def _potts_model_initializer(shape, *ignored, **ignored_too):
    return -1 * _diagonal_initializer(shape)


class CrfLayer(tf.keras.layers.Layer):
    """ Implements the CRF-RNN layer described in:
    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera- \
        Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(
                                                       self.num_classes, self.num_classes),
                                                   initializer=_diagonal_initializer,
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(
                                                         self.num_classes, self.num_classes),
                                                     initializer=_diagonal_initializer,
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(
                                                        self.num_classes, self.num_classes),
                                                    initializer=_potts_model_initializer,
                                                    trainable=True)

        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1))
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1))

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1]
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            softmax_out = tf.nn.softmax(q_values, 0)

            # Spatial filtering
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = (tf.matmul(self.spatial_ker_weights,
                                         tf.reshape(spatial_out, (c, -1))) +
                               tf.matmul(self.bilateral_ker_weights,
                                         tf.reshape(bilateral_out, (c, -1))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w))
            q_values = unaries - pairwise

        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape


c = CrfLayer(image_dims=(256, 256),
             num_classes=1,
             theta_alpha=160.,
             theta_beta=3.,
             theta_gamma=3.,
             num_iterations=10,
             name='crfrnn')
x = tfp.distributions.Bernoulli(probs=0.2, dtype=tf.float32).sample(
    sample_shape=(16, 256, 256, 1))
print(x)

print(c(x))


"""
class CRF_Layer(tf.keras.layers.Layer):
    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5,
                 returns='logits', smoothness_weight=1, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.returns = returns

        self.smoothness_weight = smoothness_weight
        self.inv_smoothness_theta = 1 / \
            np.broadcast_to(smoothness_theta, n_spatial_dims)

    def call(self, inputs, spatial_spacings=None):
        x = inputs
        batch_size, *spatial, n_classes = x.shape
        assert len(spatial) == self.n_spatial_dims

        # binary segmentation case
        if n_classes == 1:
            x = tf.concat([x, tf.zeros(x.shape, dtype=tf.int32)], axis=1)

        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = tf.identity(inputs)

        for i in range(self.n_iter):
            # normalizing
            x = tf.nn.softmax(x, axis=1)

            # message passing
            x = self.smoothness_weight * \
                self._smoothing_filter(x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        if self.returns == 'logits':
            output = x
        elif self.returns == 'proba':
            output = tf.nn.softmax(x, axis=1)
        elif self.returns == 'log-proba':
            output = tf.nn.log_softmax(x, axis=1)
        else:
            raise ValueError(
                "Attribute ``returns`` must be 'logits', 'proba' or 'log-proba'.")

        if n_classes == 1:
            output = output[:, 0] - output[:,
                                           1] if self.returns == 'logits' else output[:, 0]
            output = tf.expand_dims(output, -1)

        return output

    def _smoothing_filter(self, x, spatial_spacings):
        return tf.stack([self._single_smoothing_filter(x[i], spatial_spacings[i]) for i in range(x.shape[0])])

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]
        # CHECK if reversed is needed here !!!
        return tf.pad(x, list(reversed(padding)))  # F.pad pads from the end

    def _single_smoothing_filter(self, x, spatial_spacing):
        x = self._pad(x, self.filter_size)
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filtering
            kernel = self._create_gaussian_kernel1d(self.inv_smoothness_theta[i], spatial_spacing[i],
                                                    self.filter_size[i]).view(1, 1, -1)
            x = tf.nn.conv1d(x, kernel)
            # reshape back to (n, *spatial)
            x = tf.transpose(tf.reshape(tf.squeeze(x, 1), (*shape_before_flatten,
                                                           x.shape[-1])), [-1, dim])

        return x

    @ staticmethod
    def _create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        distances = spacing * tf.range(-(filter_size // 2), filter_size //
                                       2 + 1)
        kernel = tf.math.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = tf.ones(filter_size)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        labels = tf.range(x.shape[1])
        compatibility_matrix = self._compatibility_function(
            labels, tf.expand_dims(labels, 1))
        return tf.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @ staticmethod
    def _compatibility_function(label1, label2):
        return tf.cast(-(label1 == label2), tf.float32)


c = CRF_Layer(2)
x = tfp.distributions.Bernoulli(probs=0.2, dtype=tf.float32).sample(
    sample_shape=(16, 256, 256, 1))
print(x)

print(c(x))
"""
