import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dropout,
    SpatialDropout2D,
    UpSampling2D,
    Input,
    concatenate,
    add,
    multiply,
    Activation,
)

# adapted from https://github.com/karolzak/keras-unet
# some changes and improvements made here were merged into the repo above


def upsample_conv(filters, kernel_size, strides, padding):
    """Deconvolutional upsampling operation
    """
    return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    """Simple upsampling operation
    """
    return UpSampling2D(strides)


def attention_concat(conv_below, skip_connection):
    """Performs concatenation of upsampled conv_below with attention gated version of skip-connection
    """
    below_filters = conv_below.get_shape().as_list()[-1]
    attention_across = attention_gate(skip_connection, conv_below, below_filters)
    return concatenate([conv_below, attention_across])


def attention_gate(inp_1, inp_2, n_intermediate_filters):
    """Attention gate. Compresses both inputs to n_intermediate_filters filters before processing.
       Implemented as proposed by Oktay et al. see https://github.com/ozan-oktay/Attention-Gated-Networks.
    """
    inp_1_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(inp_1)
    inp_2_conv = Conv2D(
        n_intermediate_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(inp_2)

    f = Activation("relu")(add([inp_1_conv, inp_2_conv]))
    g = Conv2D(
        filters=1,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_uniform",
    )(f)
    h = Activation("sigmoid")(g)
    return multiply([inp_1, h])


def conv2d_block(
    inputs,
    activation,
    kernel_initializer,
    use_batch_norm=True,
    use_spatial_dropout=True,
    dropout=0.3,
    filters=16,
    kernel_size=(3, 3),
    padding="same",
):
    """ 2D Convolutional Block 
    """

    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(inputs)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        if use_spatial_dropout:
            c = SpatialDropout2D(dropout)(c)
        else:
            c = Dropout(dropout)(c)
    c = Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
    )(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    return c


def get_custom_unet(
    input_shape,
    num_classes=1,
    use_batch_norm=True,
    use_attention=False,
    use_spatial_dropout=True,
    upsample_mode="deconv",  # 'deconv' or 'simple'
    use_dropout_on_upsampling=False,
    dropout=0.3,
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    kernel_initializer="he_uniform",
    activation_function="relu",
    output_activation="sigmoid",
):  # 'sigmoid' or 'softmax'

    if upsample_mode == "deconv":
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(
            inputs=x,
            activation=activation_function,
            filters=filters,
            use_batch_norm=use_batch_norm,
            use_spatial_dropout=use_spatial_dropout,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
        )
        down_layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        dropout += dropout_change_per_layer
        filters = filters * 2  # double the number of filters with each layer

    # bottleneck
    x = conv2d_block(
        inputs=x,
        activation=activation_function,
        filters=filters,
        use_batch_norm=use_batch_norm,
        use_spatial_dropout=use_spatial_dropout,
        dropout=dropout,
        kernel_initializer=kernel_initializer,
    )

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2  # decreasing number of filters with each layer
        dropout -= dropout_change_per_layer
        x = upsample(filters, (2, 2), strides=(2, 2), padding="same")(x)
        if use_attention:
            x = attention_concat(conv_below=x, skip_connection=conv)
        else:
            x = concatenate([x, conv])
        x = conv2d_block(
            inputs=x,
            activation=activation_function,
            filters=filters,
            use_batch_norm=use_batch_norm,
            use_spatial_dropout=use_spatial_dropout,
            dropout=dropout,
            kernel_initializer=kernel_initializer,
        )

    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
