from lib.Transform import Transform
import numpy as np

class Scale(Transform):
    def __init__(self, *factors: float):
        assert 2 <= len(factors) <= 3

        dim = len(factors)
        M = np.eye(dim + 1, dtype=float)
        for i, s in enumerate(factors):
            M[i, i] = s

        super().__init__(M)
