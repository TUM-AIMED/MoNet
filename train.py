# %%
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing.data_loader import prepare_data
from models.MoNet import getMoNet
import segmentation_models as sm
import tensorflow.compat.v1 as tf

# import tensorflow as tf
import numpy as np

# %%

RES = 256
RES_Z = 16
CROP_HEIGHT = 16


b_size = 16
n_classes = 1
# %%

data_gen_args = dict(
    rotation_range=10.0,
    zoom_range=(0.8, 1.2),
    height_shift_range=0.2,
    width_shift_range=0.2,
    brightness_range=(0.7, 1.2),
    rescale=1 / 255.0,
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 42
image_generator = image_datagen.flow_from_directory(
    "../data/tf_format/train/images",
    seed=seed,
    batch_size=b_size,
    color_mode="grayscale",
    save_format="png",
)
mask_generator = mask_datagen.flow_from_directory(
    "../data/tf_format/train/masks", seed=seed, batch_size=b_size
)
train_generator = zip(image_generator, mask_generator)

val_generator = zip(
    image_datagen.flow_from_directory(
        "../data/tf_format/val/images", seed=seed, batch_size=b_size
    ),
    mask_datagen.flow_from_directory(
        "../data/tf_format/val/masks", seed=seed, batch_size=b_size
    ),
)
# %%

model = getMoNet(input_shape=(RES, RES, 1), output_classes=n_classes)
model.summary()

"""
base_model = sm.FPN(backbone_name='resnet34',
                    encoder_weights='imagenet', classes=1, activation='sigmoid')
inp = layers.Input(shape=(256, 256, 1))
l1 = layers.Conv2D(3, (1, 1))(inp)  # map N channels data to 3 channels
out = base_model(l1)
model = tf.keras.models.Model(inp, out, name=base_model.name)
model.summary()
"""
# %%

dice_l = sm.losses.bce_dice_loss
dice_c = sm.metrics.f1_score
# %%

model.compile(
    loss=dice_l,
    optimizer="adam",
    metrics=[dice_c, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    experimental_run_tf_function=False,
)
# %%

# model.load_weights("./serialized/weights/bayesian_monet.h5")
# %%

history = model.fit(
    train_generator,
    epochs=150,
    validation_data=val_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            min_delta=0.02, patience=50, restore_best_weights=True
        )
    ],
)

# %%

model.save_weights("./serialized/weights/monet.h5")
# %%

print(model.evaluate(val_generator))
# %%


# t = model.predict(X_test[..., None])
