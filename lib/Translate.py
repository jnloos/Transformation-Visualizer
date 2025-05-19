from lib.Transform import Transform
import numpy as np

class Translate(Transform):
    def __init__(self, *offsets: float):
        assert 2 <= len(offsets) <= 3

        dim = len(offsets)
        M = np.eye(dim + 1, dtype=float)
        M[:dim, dim] = offsets

        super().__init__(M)