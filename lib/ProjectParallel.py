from lib.Transform import Transform
import numpy as np

class ProjectParallel(Transform):
    def __init__(self, direction: tuple, eliminate: str = "z"):
        assert eliminate in ('x', 'y', 'z')

        dx, dy, dz = direction
        if eliminate == 'x':
            assert dx != 0
            M = np.array([
                [0, 0, 0, 0],
                [-dy/dx, 1, 0, 0],
                [-dz/dx, 0, 1, 0],
                [0, 0, 0, 1],
            ])
        elif eliminate == 'y':
            assert dy != 0
            M = np.array([
                [1, -dx/dy, 0, 0],
                [0, 0, 0, 0],
                [0, -dz/dy, 1, 0],
                [0, 0, 0, 1],
            ])
        else:
            assert dz != 0
            M = np.array([
                [1, 0, -dx/dz, 0],
                [0, 1, -dy/dz, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ])

        super().__init__(M)