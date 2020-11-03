import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_metric():
    """ Metric Tests """
    def test_binary_dice():
        pass

    def test_per_volume_dice():
        pass

    def test_dice():
        # no overlap
        from metrics.loss import dice_coefficient
        t_true = tf.zeros((16, 256, 256, 3))
        t_pred = tf.ones((16, 256, 256, 3))
        assert np.round(dice_coefficient(t_true, t_pred), 2) == 0.

        # full overlap
        t_true = tf.ones((16, 256, 256, 3))
        t_pred = tf.ones((16, 256, 256, 3))
        assert np.round(dice_coefficient(t_true, t_pred), 2) == 1.

    def test_dice3d():
        pass

    test_binary_dice()
    test_per_volume_dice()
    test_dice()
    test_dice3d()


def test_models():
    """ Model Tests """
    def test_bin_MoNet():
        from models.MoNet import getMoNet
        t = tf.ones((16, 64, 64, 1))
        bin_MoNet = getMoNet(input_shape=(64, 64, 1), output_classes=1)
        t_pred = bin_MoNet.predict(t, verbose=0)
        assert t_pred.shape == (16, 64, 64, 1)

    def test_multi_classMoNet():
        from models.MoNet import getMoNet
        t = tf.ones((16, 64, 64, 1))
        bin_MoNet = getMoNet(input_shape=(64, 64, 1), output_classes=3)
        t_pred = bin_MoNet.predict(t, verbose=0)
        assert t_pred.shape == (16, 64, 64, 3)

    test_bin_MoNet()
    test_multi_classMoNet()


if __name__ == '__main__':
    test_metric()
    test_models()
    print("Passed all tests")
