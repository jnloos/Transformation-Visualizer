import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive animation
import matplotlib.pyplot as plt
from lib.Shape import Shape

class TransformVisualizer:
    def __init__(self, dim: int):
        assert 2 <= dim <= 3, 'Invalid dimension.'
        self.dim = dim

        self.__shapes: list[tuple[Shape, str]] = []

        self.fig = None
        self.ax = None

        self.hold_time = 1.0
        self.transition_time = 0.5

        self.__viewport = None

    def with_shape(self, shape: Shape, color: str = 'black'):
        assert shape.get_dim() == self.dim, 'Dimension mismatch.'
        self.__shapes.append((shape, color))
        return self

    def with_hold_time(self, sec: float):
        assert sec > 0, 'Invalid seconds.'
        self.hold_time = sec
        return self

    def with_transition_time(self, sec: float):
        assert sec > 0, 'Invalid seconds.'
        self.transition_time = sec
        return self

    def with_viewport(self, *viewport: tuple[float, float]):
        assert len(viewport) == self.dim, 'Invalid number of viewport tuples.'
        for lim in viewport:
            assert isinstance(lim, tuple) and len(lim) == 2, 'Each viewport limit must be a tuple[float, float]'
        self.__viewport = viewport
        return self

    def show(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': None} if self.dim == 2 else {'projection': '3d'})

        # reset shapes
        for shape, _ in self.__shapes:
            shape.reset()

        # determine max steps of pipeline
        max_steps = max((len(shape.pipe) for shape, _ in self.__shapes), default=0)

        for step in range(max_steps + 1):
            self.ax.clear()
            # reapply viewport after clear
            if self.__viewport:
                self.ax.set_xlim(*self.__viewport[0])
                self.ax.set_ylim(*self.__viewport[1])
                if self.dim == 3:
                    self.ax.set_zlim(*self.__viewport[2])
            else:
                self.ax.autoscale()

            # draw shapes
            for shape, color in self.__shapes:
                pts, segments = shape.export()
                for start, end in segments:
                    xs = [start[0], end[0]]
                    ys = [start[1], end[1]] if self.dim >= 2 else None
                    if self.dim == 2:
                        self.ax.plot(xs, ys, color=color)
                    else:
                        zs = [start[2], end[2]]
                        self.ax.plot(xs, ys, zs, color=color)
                if self.dim == 2:
                    self.ax.scatter(pts[0, :], pts[1, :], color=color, zorder=3)
                else:
                    self.ax.scatter(pts[0, :], pts[1, :], pts[2, :], color=color, zorder=3)

            # maintain equal aspect ratio
            if not self.__viewport:
                self.ax.autoscale()
            self.ax.set_aspect('equal', 'box')

            # draw and pause to show frame
            self.fig.canvas.draw()
            plt.pause(self.hold_time)

            # advance to next step
            if step < max_steps:
                for shape, _ in self.__shapes:
                    if shape.index < len(shape.pipe):
                        shape.apply_next()
                plt.pause(self.transition_time)

        plt.show()
        return self

    def save(self, filename: str, **kwargs):
        if self.fig is None:
            self.show()
        self.fig.savefig(filename, **kwargs)
        return self

