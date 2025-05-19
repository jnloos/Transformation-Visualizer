import matplotlib.pyplot as plt
from lib.Shape import Shape
import numpy as np

class TransformVisualizer:
    def __init__(self, dim: int):
        assert dim in (2, 3), 'Invalid dimension.'
        self.dim = dim
        self.shapes: list[tuple[Shape, str]] = []

        # Matplot settings
        self.fig = None
        self.ax = None
        self.viewport = None
        self.title = None

        # Animation settings
        self.hold_time = 1.0
        self.transition_time = 0.5

        # Artist lists
        self.__edge_artists = []
        self.__point_artists = []
        self.__segments_counts = []

    def with_shape(self, shape: Shape, color: str = 'black'):
        assert shape.get_dim() == self.dim, 'Dimension mismatch.'
        self.shapes.append((shape, color))
        return self

    def with_title(self, title: str):
        self.title = title
        return self

    def with_hold_time(self, sec: float):
        assert sec > 0, 'Invalid seconds.'
        self.hold_time = sec
        return self

    def with_transition_time(self, sec: float):
        assert sec > 0, 'Invalid seconds.'
        self.transition_time = sec
        return self

    def with_viewport(self, *limits: tuple[float, float]):
        assert len(limits) == self.dim, 'Invalid number of viewport tuples.'
        for lim in limits:
            assert isinstance(lim, tuple) and len(lim) == 2, 'Invalid viewport tuple.'
        self.viewport = limits
        return self

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
        else:
            x0,x1 = self.ax.get_xlim()
            y0,y1 = self.ax.get_ylim()
            z0,z1 = self.ax.get_zlim()
            self.ax.plot([x0,x1],[0,0],[0,0], color=color, linestyle=style)
            self.ax.plot([0,0],[y0,y1],[0,0], color=color, linestyle=style)
            self.ax.plot([0,0],[0,0],[z0,z1], color=color, linestyle=style)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')

    def __init_artists(self):
        frame_data = self.__get_data()
        self.__edge_artists.clear()
        self.__point_artists.clear()
        self.__segments_counts.clear()

        for data in frame_data:
            points, segments, color = data['points'], data['segments'], data['color']
            self.__segments_counts.append(len(segments))
            # edges
            for i,j in segments:
                start, end = points[:,i], points[:,j]
                if self.dim == 2:
                    line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], color=color)
                else:
                    line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
                self.__edge_artists.append(line)
            # points
            if self.dim == 2:
                scatter = self.ax.scatter(points[0], points[1], color=color, zorder=3)
            else:
                scatter = self.ax.scatter(points[0], points[1], points[2], color=color, zorder=3)
            self.__point_artists.append(scatter)

    def __get_data(self):
        data = []
        for shape, color in self.shapes:
            points = shape.get_points().copy()
            if self.dim == 3:
                # Matplotlib displays the Y-axis vertical
                # => Swap Y and Z data for my favored notation.
                points[[1, 2], :] = points[[2, 1], :]
            data.append({'points': points, 'segments': shape.segments, 'color': color})
        return data

    def __update_artists(self, frame_data):
        # Update edges
        art_index = 0
        for shape_idx, seg_count in enumerate(self.__segments_counts):
            segments = frame_data[shape_idx]['segments']
            points = frame_data[shape_idx]['points']
            for seg_idx in range(seg_count):
                line = self.__edge_artists[art_index + seg_idx]
                i_idx, j_idx = segments[seg_idx]
                start, end = points[:, i_idx], points[:, j_idx]
                if self.dim == 2:
                    line.set_data([start[0], end[0]], [start[1], end[1]])
                else:
                    line.set_data([start[0], end[0]], [start[1], end[1]])
                    line.set_3d_properties([start[2], end[2]])
            art_index += seg_count

        # Update points
        for shape_idx, scatter in enumerate(self.__point_artists):
            points = frame_data[shape_idx]['points']
            if self.dim == 2:
                coords = np.stack([points[0], points[1]], axis=-1)
                scatter.set_offsets(coords)
            else:
                scatter._offsets3d = (points[0], points[1], points[2])

        self.fig.canvas.draw_idle()

    def __run_pipeline(self):
        self.ax.clear()
        self.__apply_viewport()
        self.__draw_axes()
        self.__init_artists()
        fps = 30
        plt.pause(self.hold_time)

        max_steps = max((len(shape.pipe) for shape, _ in self.shapes), default=0)
        for _ in range(max_steps):
            old_data = self.__get_data()
            # Apply transforms
            for shape, _ in self.shapes:
                if shape.index < len(shape.pipe):
                    shape.apply_next()
            new_data = self.__get_data()
            # Interpolate
            frames = max(int(self.transition_time * fps), 1)
            pause_dt = self.transition_time / frames
            for f in range(1, frames + 1):
                alpha = f / frames
                interp_frame = []
                for od, nd in zip(old_data, new_data):
                    points = od['points'] * (1 - alpha) + nd['points'] * alpha
                    interp_frame.append({'points': points, 'segments': od['segments'], 'color': od['color']})
                self.__update_artists(interp_frame)
                plt.pause(pause_dt)
            plt.pause(self.hold_time)

    def show(self):
        self.__init_figure()
        while plt.fignum_exists(self.fig.number):
            for shape, _ in self.shapes:
                shape.reset()
            self.__run_pipeline()
        return self

    def save(self, filename: str, **kwargs):
        if not self.fig:
            self.show()
        self.fig.savefig(filename, **kwargs)
        return self
