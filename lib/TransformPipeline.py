import numpy as np
from lib.Transform import Transform
from lib.Translate import Translate
from lib.MoveAxis import MoveAxis
from lib.Scale import Scale
from lib.Rotate import Rotate

class TransformPipeline:
    def __init__(self, points: np.ndarray):
        assert isinstance(points, np.ndarray), 'Points must be a numpy array'
        assert points.ndim == 2, 'Points must be a 2D array of shape (D, N)'

        self.__dim = points.shape[0]

        ones = np.ones((1, points.shape[1]))
        homogenous = np.vstack([points, ones])
        self.init_points = homogenous.copy()
        self.__points = homogenous.copy()

        self.pipe: list[Transform] = []
        self.index: int = 0

    def reset(self):
        self.__points = self.init_points.copy()
        self.index = 0

    def translate(self, *offsets: float):
        assert len(offsets) == self.__dim, 'Invalid number of offsets'
        self.pipe.append(Translate(*offsets))
        return self

    def scale(self, *factors: float, pivot: tuple[float, ...] | None = None):
        assert len(factors) == self.__dim, 'Invalid number of factors'
        scale = Scale(*factors)
        if pivot is not None:
            self.__apply_at_pivot(scale, pivot)
        else:
            self.pipe.append(scale)
        return self

    def rotate(self, theta: float, axis: str | None = None, pivot: tuple[float, ...] | None = None):
        assert (axis is None and self.__dim == 2) or (axis is not None and self.__dim == 3)
        rotate = Rotate(theta, axis=axis)
        if pivot is not None:
            self.__apply_at_pivot(rotate, pivot)
        else:
            self.pipe.append(rotate)
        return self

    def __apply_at_pivot(self, transform: Transform, pivot: tuple[float, ...]):
        self.pipe.append(MoveAxis(*pivot))
        self.pipe.append(transform)
        self.pipe.append(MoveAxis(*(-c for c in pivot)))

    def move_axis(self, *pivot: float):
        assert len(pivot) == self.__dim, 'Invalid number of pivots'
        self.pipe.append(MoveAxis(*pivot))
        return self

    def matrix(self, M: np.ndarray):
        assert isinstance(M, np.ndarray), 'Matrix must be a numpy array'
        assert M.ndim == 2 and M.shape[0] == M.shape[1], 'Matrix must be square'
        assert M.shape[0] == self.__dim + 1, 'Matrix size must match homogeneous dimension'
        self.pipe.append(Transform(M))
        return self

    def has_next(self) -> bool:
        return self.index < len(self.init_points)

    def apply_next(self) -> np.ndarray:
        assert self.pipe, 'All transforms have been applied'
        transform = self.pipe[self.index]
        self.__points = transform.apply(self.__points)
        self.index += 1
        return self.get_points()

    def apply_all(self) -> np.ndarray:
        assert self.pipe, 'No transforms to apply'

        # Compose all transformations
        size = self.__dim + 1
        T = np.eye(size)
        for transform in self.pipe:
            T = transform.matrix @ T
        self.__points = T @ self.__points
        self.index = len(self.init_points)

        return self.get_points()

    def get_points(self) -> np.ndarray:
        return self.__points[:self.__dim, :]

    def get_dim(self) -> int:
        return self.__dim
