import numpy as np

class Transform:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.matrix @ points