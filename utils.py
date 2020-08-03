import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import clear_output
import tensorflow
import time


def plot_learning_curve(
    history: dict,
    path: str,
    metrics: list = ["dice_coefficient", "val_dice_coefficient"],
    losses: list = ["loss", "val_loss"],
):
    """
        Plots metrics and losses for train and validation set over training epochs
        :param history : history dict
        :param metrics: metrics to be plotted
        :param losses: losses to be plotted
    """
    file_path = Path(path)
    metric_path = file_path / "metrics_curve.png"
    loss_path = file_path / "loss_curve.png"

    # plot metrics over training epochs
    plt.figure(figsize=(10, 8))
    for metric in metrics:
        plt.plot(history[metric], linewidth=3)
    plt.suptitle("metrics over epochs", fontsize=20)
    plt.ylabel("metric", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.legend(metrics, loc="center right", fontsize=15)
    # saving plot to file_path if not none
    if path != None:
        if file_path.exists():
            plt.savefig(metric_path.resolve(), dpi=500)
        else:
            raise FileNotFoundError()

    plt.show()

    # plot loss over training
    plt.figure(figsize=(10, 8))
    for loss in losses:
        plt.plot(history[loss], linewidth=3)
    plt.suptitle("loss over epochs", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.xlabel("epoch", fontsize=20)
    plt.legend(losses, loc="center right", fontsize=15)
    if path != None:
        if file_path.exists():
            plt.savefig(loss_path.resolve(), dpi=500)
        else:
            raise FileNotFoundError()
    plt.show()


class PlotLosses(tensorflow.keras.callbacks.Callback):
    """ Loss Plotting callback
        plots: loss, binary accuracy and dice coefficient

    """

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.dice = []
        self.val_dice = []
        self.accuracy = []
        self.val_accuracy = []

        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.dice.append(logs.get("dice_coefficient"))
        self.val_dice.append(logs.get("val_dice_coefficient"))
        self.accuracy.append(logs.get("accuracy"))
        self.val_accuracy.append(logs.get("val_accuracy"))

        self.i += 1

        plt.figure(figsize=(12, 6))
        clear_output(wait=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 5))
        plt.subplots_adjust(wspace=1, top=0.8)

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.set_title("Learning Cuve: Loss / Val-Loss")
        plt.legend()

        ax2.plot(self.x, self.accuracy, label="accuracy")
        ax2.plot(self.x, self.val_accuracy, label="val_accuracy")
        ax2.set_title("Learning Curve: Binary Accuracy")
        plt.legend()

        ax3.plot(self.x, self.dice, label="dice")
        ax3.plot(self.x, self.val_dice, label="val_dice")
        ax3.set_title("Learning Curve: Dice Score")
        plt.legend()

        plt.tight_layout()
        plt.show()


class EpochTimeHistory(tensorflow.keras.callbacks.Callback):
    """ Records and prints the time taken for one epoch (in seconds)

    """

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        epoch_time = time.time() - self.epoch_time_start
        print(f"Epoch {epoch+1} took: {epoch_time} seconds")
        self.times.append(epoch_time)
