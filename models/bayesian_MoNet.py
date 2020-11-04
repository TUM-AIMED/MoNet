import joblib
import os
import segmentation_models as sm
import tensorflow.keras.layers as tf_layers
import tensorflow_probability.python.layers as tfp_layers
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tfd = tfp.distributions
kl_divergence_function = (
    lambda q, p, _: tf.reduce_mean(tfd.kl_divergence(q, p)))


def ConvBnElu(inp, filters, kernel_size=3, strides=1, dilation_rate=1):
    """ Conv-Batchnorm-Elu block
    """
    x = tfp_layers.Convolution2DFlipout(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        activation=None
    )(inp)
    x = tf_layers.BatchNormalization()(x)
    x = tf_layers.Activation("elu")(x)
    return x


def deconv(inp):
    """Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """
    num_filters = inp.get_shape().as_list()[-1]

    x = tf_layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_uniform",
    )(inp)
    x = tf_layers.BatchNormalization()(x)
    x = tf_layers.Activation("elu")(x)

    return x


def repeat_block(inp, out_filters, dropout=0.2):
    """ Reccurent conv block with decreasing kernel size. Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """
    skip = inp

    c1 = ConvBnElu(inp, out_filters, dilation_rate=4)
    c1 = tf_layers.SpatialDropout2D(dropout)(c1)
    c2 = ConvBnElu(tf_layers.add([skip, c1]), out_filters, dilation_rate=3)
    c2 = tf_layers.SpatialDropout2D(dropout)(c2)
    c3 = ConvBnElu(c2, out_filters, dilation_rate=2)
    c3 = tf_layers.SpatialDropout2D(dropout)(c3)
    c4 = ConvBnElu(tf_layers.add([c2, c3]), out_filters, dilation_rate=1)

    return tf_layers.add([skip, c4])


def getMoNet(
    input_shape=(256, 256, 1),
    output_classes=1,
    depth=2,
    n_filters_init=16,
    dropout_enc=0.2,
    dropout_dec=0.2,
):

    inputs = x = tf_layers.Input(input_shape)
    skips = []
    features = n_filters_init
    if output_classes > 1:
        activation = 'softmax'
    else:
        activation = 'sigmoid'

    # encoder
    for i in range(depth):
        x = ConvBnElu(x, features)
        x = repeat_block(x, features, dropout=dropout_enc)
        skips.append(x)
        x = ConvBnElu(x, features, kernel_size=4, strides=2)
        features *= 2

    # bottleneck
    x = ConvBnElu(x, features)
    x = repeat_block(x, features)

    # decoder
    for i in reversed(range(depth)):
        features //= 2
        x = deconv(x)
        x = tf_layers.concatenate([skips[i], x])
        x = ConvBnElu(x, features)
        x = repeat_block(x, features, dropout=dropout_dec)

    # head
    final_conv = tfp_layers.Convolution2DFlipout(
        output_classes,
        kernel_size=1,
        strides=1,
        padding="same",
    )(x)
    final_bn = tf_layers.BatchNormalization()(final_conv)
    act = tf_layers.Activation(activation)(final_bn)
    return tf.keras.models.Model(inputs, act)


"""
monet = getMoNet(output_classes=1)
monet.summary()


X_partial = joblib.load("./serialized/data/x_partial.lib")
y_partial = joblib.load("./serialized/data/y_partial.lib")
X_val = joblib.load("./serialized/data/x_val.lib")
y_val = joblib.load("./serialized/data/y_val.lib")

# one hot encoding panc + tumor labels
y_partial = tf.keras.utils.to_categorical(y_partial)
y_val = tf.keras.utils.to_categorical(y_val)

dice_l = sm.losses.bce_dice_loss
dice_c = sm.metrics.f1_score

monet.compile(loss=dice_l,
              optimizer='adam', metrics=[dice_c, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()], experimental_run_tf_function=False)

monet.fit(X_partial[..., None], y_partial,
          validation_data=(X_val[..., None], y_val),
          epochs=20,
          verbose=1)
"""
