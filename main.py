import numpy as np
from lib.Shape import Shape
from lib.TransformVisualizer import TransformVisualizer

import matplotlib
matplotlib.use('TkAgg')

def main():
    points = np.array([
        [-1, -1,  1,   1,  -1, -1,  1,  1],
        [-1,  1,  1,  -1,  -1,  1,  1, -1],
        [-1, -1, -1,  -1,   1,  1,  1,  1]
    ])
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]

    cube_a = Shape(points.copy(), edges).translate(0, 1, 2).rotate(np.pi/4, axis='x').scale(1.5, 1.5, 1.5)
    cube_b = Shape(points.copy(), edges).scale(1.5, 1.5, 1.5).rotate(np.pi/4, axis='x').translate(0, 1, 2)

    TransformVisualizer(dim=3) \
        .with_hold_time(1.0) \
        .with_transition_time(0.75) \
        .with_viewport((-3, 4), (-3, 4), (-1, 3)) \
        .with_shape(cube_a, 'blue') \
        .with_shape(cube_b, 'red') \
        .with_title('Cube Visualization') \
        .show()

if __name__ == '__main__':
    main()
