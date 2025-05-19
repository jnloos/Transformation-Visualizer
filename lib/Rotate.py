from lib.Transform import Transform
import numpy as np

class Rotate(Transform):
    def __init__(self, theta: float, axis: str|None = None):
        if axis is not None:
            axis = axis.lower()

        if axis not in (None, 'x', 'y', 'z'):
            raise ValueError(f"{axis} is an invalid axis.")

        dim = 2 if axis is None else 3
        M = np.eye(dim + 1, dtype=float)
        c, s = np.cos(theta), np.sin(theta)

        if axis is None or axis == 'z':
            M[0, 0], M[0, 1] = c, -s
            M[1, 0], M[1, 1] = s,  c
        elif axis == 'x':
            M[1, 1], M[1, 2] = c, -s
            M[2, 1], M[2, 2] = s,  c
        else:
            M[0, 0], M[0, 2] =  c, s
            M[2, 0], M[2, 2] = -s, c

        super().__init__(M)