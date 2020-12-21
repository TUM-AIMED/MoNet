import os
import sys
import distutils

sys.path.append("../")

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

import flwr as fl
from models.MoNet import getMoNet
from models.custom_unet import get_custom_unet
import segmentation_models as sm
import joblib
import argparse

dice_loss = sm.losses.dice_loss
dice = sm.metrics.f1_score


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    # parse Args
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition", type=int, required=True, help="Data Partition index (no default)"
    )
    parser.add_argument(
        "--monet",
        type=boolean_string,
        required=True,
        help="Bool value to indicate whether to use MoNet or U-net",
        default=True,
    )
    args = parser.parse_args()
    # Load and compile model
    if args.monet == True:
        model = getMoNet()
    else:
        model = get_custom_unet((256, 256, 1), filters=16, use_attention=False)
    # print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01), loss=dice_loss, metrics=[dice]
    )
    try:
        print(f"... Loading data partition: {args.partition}")
        data_path = f"./serialized/client_{args.partition}/"
        # Load train and validation dataset
        x_train, y_train = (
            joblib.load(data_path + "x_train.lib"),
            joblib.load(data_path + "y_train.lib"),
        )
        x_val, y_val = (
            joblib.load(data_path + "x_val.lib"),
            joblib.load(data_path + "y_val.lib"),
        )
    except:
        raise FileNotFoundError(
            f"Something went wrong trying to load data for partition: {args.partition} and path: {data_path}"
        )

    # Define client
    class MSDClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=2, batch_size=12)
            return model.get_weights(), len(x_train)

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, dice = model.evaluate(x_val, y_val)
            return len(x_val), loss, dice

    # Start Flower client
    fl.client.start_numpy_client("[::]:8080", client=MSDClient())
