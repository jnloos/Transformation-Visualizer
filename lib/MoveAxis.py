from lib.Transform import Transform
import numpy as np

class MoveAxis(Transform):
    def __init__(self, *pivot: tuple):
        p = np.array(pivot, dtype=float)
        dim = p.shape[0]

        T = np.eye(dim + 1, dtype=float)
        T[:dim, -1] = -p

        super().__init__(T)
