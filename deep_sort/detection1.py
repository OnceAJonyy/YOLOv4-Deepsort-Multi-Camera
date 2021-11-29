# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection1(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh1, confidence1, class_name1, feature1):
        self.tlwh1 = np.asarray(tlwh1, dtype=np.float)
        self.confidence1 = float(confidence1)
        self.class_name1 = class_name1
        self.feature1 = np.asarray(feature1, dtype=np.float32)

    def get_class(self):
        return self.class_name1

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh1.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh1.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
