from lib.TransformPipeline import TransformPipeline
import numpy as np

class Shape(TransformPipeline):
    def __init__(self, points: np.ndarray, segments: list[tuple[int, int]]):
        super().__init__(points)
        self.segments = segments

    def export(self) -> tuple:
        points = self.get_points()
        segments = []
        for start_idx, end_idx in self.segments:
            start = tuple(points[:, start_idx])
            end = tuple(points[:, end_idx])
            segments.append([start, end])
        return points, segments
