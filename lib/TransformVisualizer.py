import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lib.Shape import Shape
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

class TransformVisualizer:
    def __init__(self, dim: int):
        """
        Initialize the visualizer with given dimension.
        :param dim: Dimension (2 or 3).
        """
        assert dim in (2, 3), 'Invalid dimension.'
        self.dim = dim
        self.shapes: list[tuple[Shape, str]] = []

        # Matplotlib settings
        self.fig = None
        self.ax = None
        self.viewport = None
        self.title = None

        # Animation settings
        self.hold_time = 1.0
        self.transition_time = 0.5

        # Internal artist lists
        self.__edge_artists = []
        self.__point_artists = []
        self.__segments_counts = []

    def with_shape(self, shape: Shape, color: str = 'black'):
        """
        Add a shape to the visualizer.
        :param shape: Shape instance to visualize.
        :param color: Color to use for edges and points.
        :return: self
        """
        assert shape.get_dim() == self.dim, 'Dimension mismatch.'
        self.shapes.append((shape, color))
        return self

    def with_title(self, title: str):
        """
        Set the title for the window.
        :param title: Title text.
        :return: self
        """
        self.title = title
        return self

    def with_hold_time(self, sec: float):
        """
        Configure the duration of the transformation timeout.
        :param sec: Hold time in seconds
        :return: self
        """
        assert sec > 0, 'Invalid seconds.'
        self.hold_time = sec
        return self

    def with_transition_time(self, sec: float):
        """
        Configure the duration of transitions.
        :param sec: Transition time in seconds
        :return: self
        """
        assert sec > 0, 'Invalid seconds.'
        self.transition_time = sec
        return self

    def with_viewport(self, *limits: tuple[float, float]):
        """
        Set default axis limits.
        :param limits: Tuple of (min, max) for each axis
        :return: self
        """
        assert len(limits) == self.dim, 'Invalid number of viewport tuples.'
        for lim in limits:
            assert isinstance(lim, tuple) and len(lim) == 2, 'Invalid viewport tuple.'
        self.viewport = limits
        return self

    def show(self):
        """
        Display the animation in a Matplotlib window.
        """
        self.__init_figure()
        fps = 30
        interval = 1000 / fps

        anim = FuncAnimation(self.fig, func=self.__update_artists, init_func=self.__init_artists,
            frames=self.__frame_generator(), interval=interval, blit=False, repeat=True, cache_frame_data=False)
        plt.show()

    def save(self, filename: str, fps: int = 30, writer: str = 'pillow'):
        """
        Save the animation as a GIF file.
        :param filename: Output file path (e.g., 'animation.gif').
        :param fps: Frames per second
        :param writer: Matplotlib writer backend (e.g., 'pillow').
        """
        # Initialize figure and axes
        self.__init_figure()
        interval = 1000 / fps

        anim = FuncAnimation(self.fig, func=self.__update_artists, init_func=self.__init_artists,
            frames=self.__frame_generator(), interval=interval, blit=False, repeat=False, cache_frame_data=False)

        anim.save(filename, writer=writer, fps=fps)

    # Private helper methods below:
    def __init_figure(self):
        proj = '3d' if self.dim == 3 else None
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': proj})
        self.ax.set_aspect('equal', 'box')
        if self.viewport:
            self.__apply_viewport()
        self.__draw_axes()
        if self.title:
            self.ax.set_title(self.title)
            self.fig.canvas.manager.set_window_title(self.title)

    def __apply_viewport(self):
        if not self.viewport:
            self.ax.autoscale()
        else:
            self.ax.set_xlim(*self.viewport[0])
            self.ax.set_ylim(*self.viewport[1])
            if self.dim == 3:
                self.ax.set_zlim(*self.viewport[2])

    def __draw_axes(self):
        color, style = 'lightgray', ':'
        if self.dim == 2:
            self.ax.axhline(0, color=color, linestyle=style)
            self.ax.axvline(0, color=color, linestyle=style)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
        else:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            z0, z1 = self.ax.get_zlim()
            self.ax.plot([x0, x1], [0, 0], [0, 0], color=color, linestyle=style)
            self.ax.plot([0, 0], [y0, y1], [0, 0], color=color, linestyle=style)
            self.ax.plot([0, 0], [0, 0], [z0, z1], color=color, linestyle=style)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Z')
            self.ax.set_zlabel('Y')

    def __init_artists(self):
        self.ax.clear()
        if self.viewport:
            self.__apply_viewport()
        self.__draw_axes()

        if self.title:
            self.ax.set_title(self.title)

        frame_data = self.__get_data()
        self.__edge_artists.clear()
        self.__point_artists.clear()
        self.__segments_counts.clear()

        for data in frame_data:
            pts, segs, col = data['points'], data['segments'], data['color']
            self.__segments_counts.append(len(segs))
            # Draw edges
            for i, j in segs:
                start, end = pts[:, i], pts[:, j]
                if self.dim == 2:
                    line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], color=col)
                else:
                    line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=col)
                self.__edge_artists.append(line)
            # Draw points
            if self.dim == 2:
                scatter = self.ax.scatter(pts[0], pts[1], color=col, zorder=3)
            else:
                scatter = self.ax.scatter(pts[0], pts[1], pts[2], color=col, zorder=3)
            self.__point_artists.append(scatter)

        return self.__edge_artists + self.__point_artists

    def __get_data(self):
        data = []
        for shape, col in self.shapes:
            points = shape.get_points().copy()
            if self.dim == 3:
                points[[1, 2], :] = points[[2, 1], :]
            data.append({'points': points, 'segments': shape.segments, 'color': col})
        return data

    def __update_artists(self, frame):
        art_i = 0
        for idx, seg_count in enumerate(self.__segments_counts):
            segments = frame[idx]['segments']
            points = frame[idx]['points']
            for k in range(seg_count):
                line = self.__edge_artists[art_i + k]
                i, j = segments[k]
                s, e = points[:, i], points[:, j]
                if self.dim == 2:
                    line.set_data([s[0], e[0]], [s[1], e[1]])
                else:
                    line.set_data([s[0], e[0]], [s[1], e[1]])
                    line.set_3d_properties([s[2], e[2]])
            art_i += seg_count

        # Update points
        for idx, scatter in enumerate(self.__point_artists):
            points = frame[idx]['points']
            if self.dim == 2:
                scatter.set_offsets(np.stack([points[0], points[1]], axis=-1))
            else:
                scatter._offsets3d = (points[0], points[1], points[2])
        return self.__edge_artists + self.__point_artists

    def __frame_generator(self):
        fps = 30
        hold_steps = int(self.hold_time * fps)
        # Reset shapes
        for shape, _ in self.shapes:
            shape.reset()
        # Initial hold
        init_data = self.__get_data()
        for _ in range(hold_steps):
            yield init_data
        # Transitions
        max_steps = max((len(shape.pipe) for shape, _ in self.shapes), default=0)
        for _ in range(max_steps):
            old = self.__get_data()
            for shape, _ in self.shapes:
                if shape.index < len(shape.pipe):
                    shape.apply_next()
            new = self.__get_data()
            steps = max(int(self.transition_time * fps), 1)
            for f in range(1, steps + 1):
                alpha = f / steps
                frame = []
                for od, nd in zip(old, new):
                    points = od['points'] * (1 - alpha) + nd['points'] * alpha
                    frame.append({'points': points, 'segments': od['segments'], 'color': od['color']})
                yield frame
            # Hold after transition
            for _ in range(hold_steps):
                yield new
