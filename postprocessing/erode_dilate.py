from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
import numpy as np
from skimage.measure import label, regionprops


def erode_dilate(y_pred: np.array):
    assert len(y_pred.shape) == 4 and y_pred.dtype == np.float32
    # treshold binarize
    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.

    pred = binary_erosion(
        y_pred.astype(bool), ball(3)).astype(np.float32)

    # 2) keep only largest connected component
    labels = label(pred)
    regions = regionprops(labels)
    area_sizes = []
    for region in regions:
            area_sizes.append([region.label, region.area])
        area_sizes = np.array(area_sizes)
        tmp = np.zeros_like(pred)
        tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
        pred = tmp.copy()
        del tmp, labels, regions, area_sizes

        # 3) dilate
        pred = binary_dilation(pred.astype(bool), ball(3))

        # 4) remove small holes
        pred = remove_small_holes(pred.astype(
            bool), area_threshold=0.001*np.prod(pred.shape)).astype(np.float32)
    return pred
