from lib.Transform import Transform
import numpy as np

class ProjectOrthographic(Transform):
    def __init__(self, eliminate: str = "z"):
        assert eliminate in ('x', 'y', 'z')

        M = np.eye(4)
        if eliminate == 'x':
            M[0, 0] = 0
        elif eliminate == 'y':
            M[1, 1] = 0
        else:
            M[2, 2] = 0

        super().__init__(M)
