from functools import partial
import tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import (
    TruePositives,
    FalsePositives,
    TrueNegatives,
    FalseNegatives,
)


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """ 
        Dice coefficent. This metric works for 2D and 3D Labels.
        Dice coefficent = (2 * |A ∩ B|) / (|A ⋃ B|)
                = sum(|A * B|) / (sum(|A|) + sum(|B|))
        https://arxiv.org/pdf/1707.03237.pdf
        Makes no assumptions about the rank of y_true and y_pred
        :param y_true : ground truth
        :param y_pred : predictions
        :param smooth
        :return scalar
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_3d(y_true, y_pred, axis=(-4, -3, -2), smooth=1e-6):
    """ 
        Dice coefficient for 3D data, showed to be a  better and more stable aproximator for training in 3D when used as a loss function
        Assumes y_true and y_pred are of rank 5.
        :param smooth :   default = 1e-6
        :param y_true : ground truth
        :param y_pred : predictions
        :param axis :  default assumes a "channels-last" data structure;
        :return: scalar
    """
    return K.mean(
        2.0
        * (K.sum(y_true * y_pred, axis=axis) + smooth / 2)
        / (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth)
    )


def per_volume_dice_coefficient_3d(y_true, y_pred):
    """
        Calculates and returns the dice coefficent label wise and returns a list of the score and label index for each label volume.
        Assumes tensors of rank 5 for y_true and y_pred. 
        :param y_true : ground truth
        :param y_pred : predictions
    """
    dsc_scores = []
    for l in range(y_true.shape[0]):
        dsc = dice_coefficient(y_true[l], y_pred[l])
        dsc_scores.append(dsc)

    return dsc_scores


def per_volume_dice_coefficient_2d(y_true, y_pred, num_slices_per_vol=48):
    """
        Calculates the dice coefficient per volume given an array of slice predictions
        Assumes tensors of rank 4, and that slices of scan volumes are ordered
        :param num_sclices : number of slices for each volume
        :param y_true : ground truth 
        :param y_pred : predictions
    """
    scores = []
    for i in range(0, y_true.shape[0], num_slices_per_vol):
        curr_max = i + num_slices_per_vol
        pred_volume = y_pred[i:curr_max]
        true_vol = y_true[i:curr_max]
        scores.append(dice_coefficient(true_vol, pred_volume))
    return scores


def binary_dice(y_true, y_pred):
    """
        Calculates the dice coefficient for two binary volumes according to the following formula.
        Dice = 2*TP / ((2*TP) + FP + FN)
        2D and 3D compatible
        :param y_true : ground truth
        :param y_pred : soft prediction values
    """

    # threshold binarize
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    tp = TruePositives()(y_true, y_pred)
    fp = FalsePositives()(y_true, y_pred)
    fn = FalseNegatives()(y_true, y_pred)

    dice = 2 * tp / ((2 * tp) + fp + fn)
    return dice
