import numpy as np

from lib.ProjectParallel import ProjectParallel
from lib.ProjectPerspective import ProjectPerspective
from lib.ProjectOrthographic import ProjectOrthographic
from lib.Transform import Transform
from lib.Translate import Translate
from lib.MoveAxis import MoveAxis
from lib.Scale import Scale
from lib.Rotate import Rotate

class TransformPipeline:
    def __init__(self, points: np.ndarray):
        """
        Initialize pipeline with the points to be transformed.
        :param points: 2D numpy array of points
        """
        assert isinstance(points, np.ndarray), 'Points must be a numpy array'
        assert points.ndim == 2, 'The points need to be in a 2D array'

        self.__dim = points.shape[0]
        ones = np.ones((1, points.shape[1]))
        homogenous = np.vstack([points, ones])
        self.init_points = homogenous.copy()
        self.__points = homogenous.copy()
        self.pipe: list[Transform] = []
        self.index: int = 0

    def reset(self):
        """
        Reset pipeline to initial state.
        :return: self
        """
        self.__points = self.init_points.copy()
        self.index = 0
        return self

    def translate(self, *offsets: float):
        """
        Add a translation.
        :param offsets: translation offsets for each dimension
        :return: self
        """
        assert len(offsets) == self.__dim, 'Invalid number of offsets'
        self.pipe.append(Translate(*offsets))
        return self

    def scale(self, *factors: float, pivot: tuple[float, ...] | None = None):
        """
        Add a scaling, optionally about a pivot.
        :param factors: scale factors for each dimension
        :param pivot: optional pivot point for scaling
        :return: self
        """
        assert len(factors) == self.__dim, 'Invalid number of factors'
        scale = Scale(*factors)
        if pivot is not None:
            self.__apply_at_pivot(scale, pivot)
        else:
            self.pipe.append(scale)
        return self

    def rotate(self, theta: float, axis: str | None = None, pivot: tuple[float, ...] | None = None):
        """
        Add a rotation, optionally about a pivot.
        :param theta: rotation angle in radians
        :param axis: rotation axis ('x', 'y', 'z') for 3D, None for 2D
        :param pivot: optional pivot point for rotation
        :return: self
        """
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

    def perspective_projection(self, fov: float, aspect: float, near: float, far: float):
        """
        Add a 3D perspective projection.
        :param fov: vertical field of view in radians
        :param aspect: width-to-height ratio
        :param near: near clipping plane
        :param far: far clipping plane
        :return: self
        """
        assert self.__dim == 3
        self.pipe.append(ProjectPerspective(fov, aspect, near, far))
        return self

    def parallel_projection(self, direction: tuple[float, float, float], eliminate: str):
        """
        Add a 3D parallel projection.
        :param direction: projection direction vector
        :param eliminate: axis to eliminate ('x', 'y', or 'z')
        :return: self
        """
        assert self.__dim == 3
        self.pipe.append(ProjectParallel(direction, eliminate))
        return self

    def orthographic_projection(self, eliminate: str):
        """
        Add a 3D orthographic projection.
        :param eliminate: axis to eliminate ('x', 'y', or 'z')
        :return: self
        """
        assert self.__dim == 3
        self.pipe.append(ProjectOrthographic(eliminate))
        return self

    def move_axis(self, *offsets: float):
        """
        Add a translation to shift the coordinate axis.
        :param offsets: translation offsets for axis shift
        :return: self
        """
        assert len(offsets) == self.__dim, 'Invalid number of offsets'
        self.pipe.append(MoveAxis(*offsets))
        return self

    def matrix(self, M: np.ndarray):
        """
        Add a custom homogeneous transformation matrix.
        :param M: square numpy array
        :return: self
        """
        assert isinstance(M, np.ndarray), 'Matrix must be a numpy array'
        assert M.ndim == 2 and M.shape[0] == M.shape[1], 'Matrix must be square'
        assert M.shape[0] == self.__dim + 1, 'Matrix size must match homogeneous dimension'
        self.pipe.append(Transform(M))
        return self

    def has_next(self) -> bool:
        """
        Check if there are remaining transformations.
        :return: True if transformations remain, False otherwise
        """
        return self.index < len(self.init_points)

    def apply_next(self) -> np.ndarray:
        """
        Apply the next transformations in the pipeline.
        :return: transformed points array
        """
        assert self.pipe, 'All transforms have been applied'
        t = self.pipe[self.index]
        self.__points = t.apply(self.__points)
        self.index += 1
        return self.get_points()

    def apply_all(self) -> np.ndarray:
        """
        Compose and apply all transformations at once.
        :return: transformed points array
        """
        assert self.pipe, 'No transforms to apply'
        size = self.__dim + 1
        T = np.eye(size)
        for t in self.pipe:
            T = t.matrix @ T
        self.__points = T @ self.__points
        self.index = len(self.init_points)
        return self.get_points()

    def get_points(self) -> np.ndarray:
        """
        Retrieve current transformed cartesian points.
        :return: points array
        """
        return self.__points[:self.__dim, :]

    def get_dim(self) -> int:
        """
        Get dimension of the considered points.
        :return: dimension
        """
        return self.__dim
