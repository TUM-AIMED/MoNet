from functools import partial
import tensorflow
from tensorflow.keras import backend as K
from .metric import dice_coefficient, dice_coefficient_3d


def generalized_dice_loss(y_true, y_pred):

    return 1 - dice_coefficient(y_true, y_pred)


def dice_loss_3d(y_true, y_pred):
    return 1 - dice_coefficient_3d(y_true, y_pred)


def weighted_cross_entropy(y_true, y_pred, beta=100.0):
    y_pred_clipped = tensorflow.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    y_pred_logits = tensorflow.math.log(y_pred_clipped / (1 - y_pred_clipped))

    weighted_xe = tensorflow.nn.weighted_cross_entropy_with_logits(
        logits=y_pred_logits, labels=y_true, pos_weight=beta
    )

    return tensorflow.reduce_mean(weighted_xe)
