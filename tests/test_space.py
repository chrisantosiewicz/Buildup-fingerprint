import numpy as np

from buildup.space import _seg_dist


def test_segment_distance():
    px, py = np.array([5.0, -3.0, 14.0]), np.array([2.0, 0.0, 0.0])
    d = _seg_dist(px, py, 0.0, 0.0, 10.0, 0.0)
    assert np.allclose(d, [2.0, 3.0, 4.0])   # beside the lane, behind the passer, beyond the receiver
