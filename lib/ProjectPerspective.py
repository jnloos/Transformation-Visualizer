from lib.Transform import Transform
import numpy as np

class ProjectPerspective(Transform):
    def __init__(self, fov: float, aspect: float, near: float, far: float):
        assert fov > 0
        assert aspect > 0
        assert 0 < near < far

        f = 1 / np.tan(fov / 2)
        M = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ])

        super().__init__(M)