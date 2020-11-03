import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from keras_applications import get_submodules_from_kwargs
import segmentation_models as sm
from segmentation_models.models._common_blocks import Conv2dBn
from segmentation_models.backbones.backbones_factory import Backbones


def ConvBnElu(inp, filters, kernel_size=3, strides=1, dilation_rate=1, activation='elu'):
    """ Conv-Batchnorm-Elu block
    """
    x = Conv2dBn(filters=filters, kernel_size=kernel_size,
                 strides=strides, dilation_rate=dilation_rate, activation=activation, use_batchnorm=True)(inp)
    return x


def deconv(inp):
    """Deconv upsampling of x. Doubles x and y dimension and maintains z.
    """
    num_filters = inp.get_shape().as_list()[-1]

    x = layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=4,
        strides=2,
        use_bias=False,
        padding="same",
        kernel_initializer="he_uniform",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)

    return x


def repeat_block(inp, out_filters, dropout=0.2):
    """ Reccurent conv block with decreasing kernel size. Makes use of atrous convolutions to make large kernel sizes computationally feasible

    """
    skip = inp

    c1 = ConvBnElu(inp, out_filters, dilation_rate=4)
    c1 = layers.SpatialDropout2D(dropout)(c1)
    c2 = ConvBnElu(layers.add([skip, c1]), out_filters, dilation_rate=3)
    c2 = layers.SpatialDropout2D(dropout)(c2)
    c3 = ConvBnElu(c2, out_filters, dilation_rate=2)
    c3 = layers.SpatialDropout2D(dropout)(c3)
    c4 = ConvBnElu(layers.add([c2, c3]), out_filters, dilation_rate=1)

    return layers.add([skip, c4])


def getMoNet(
    input_shape=(256, 256, 1),
    output_classes=1,
    depth=2,
    n_filters_init=16,
    dropout_enc=0.2,
    dropout_dec=0.2,
):
    backbone = Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights=encoder_weights,
        include_top=False,
        **kwargs,
    )

    input_ = backbone.input
    x = backbone.output

    inputs = x = layers.Input(input_shape)
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
        x = layers.concatenate([skips[i], x])
        x = ConvBnElu(x, features)
        x = repeat_block(x, features, dropout=dropout_dec)

    # head
    final_conv = layers.Conv2D(
        output_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
        use_bias=False,
    )(x)
    final_bn = layers.BatchNormalization()(final_conv)
    act = layers.Activation(activation)(final_bn)
    return Model(inputs, act)
