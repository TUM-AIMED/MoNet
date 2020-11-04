from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocessing.data_loader import prepare_data
from models.bayesian_MoNet import getMoNet
import segmentation_models as sm
import tensorflow.compat.v1 as tf
#import tensorflow as tf
import numpy as np

PATH = "/home/moritz/Data/Task03_Liver"
RES = 256
RES_Z = 16
CROP_HEIGHT = 16


X_train_partial, y_train_partial, X_val, y_val, X_test, y_test = prepare_data(PATH,
                                                                              res=RES,
                                                                              res_z=RES_Z,
                                                                              crop_height=CROP_HEIGHT,
                                                                              num_samples=20)

# convert targets to floats
y_train_partial = y_train_partial.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)


b_size = 16
n_classes = 1

data_gen_args = dict(
    rotation_range=10.,
    zoom_range=(0.8, 1.2),
    height_shift_range=0.2,
    width_shift_range=0.2,
    brightness_range=(0.7, 1.2),
    rescale=1/255.)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 42
image_generator = image_datagen.flow(
    X_train_partial[..., None], seed=seed, batch_size=b_size)
mask_generator = mask_datagen.flow(
    y_train_partial[..., None], seed=seed, batch_size=b_size)
train_generator = zip(image_generator, mask_generator)

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

dice_l = sm.losses.bce_dice_loss
dice_c = sm.metrics.f1_score

model.compile(loss=dice_l,
              optimizer='adam', metrics=[dice_c, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()], experimental_run_tf_function=False)

model.load_weights("./serialized/weights/bayesian_monet.h5")
history = model.fit(train_generator,
                    epochs=150,
                    validation_data=(X_val[..., None], y_val[..., None]),
                    steps_per_epoch=X_train_partial.shape[0]//b_size,
                    callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=0.02, patience=50, restore_best_weights=True)])


model.save_weights("./serialized/weights/bayesian_monet.h5")

print(model.evaluate(X_test[..., None], y_test[..., None]))


t = model.predict(X_test[..., None])


