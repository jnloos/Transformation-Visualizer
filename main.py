import matplotlib
matplotlib.use('TkAgg')

def main():
    import numpy as np
    from lib.Shape import Shape
    from lib.TransformVisualizer import TransformVisualizer

    # Example 2D shape: square
    points = np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    square = Shape(points, edges)

    # Build transformation pipeline
    square.translate(1, 2) \
          .scale(0.5, 0.5, pivot=(0.5, 0.5)) \
          .rotate(np.pi / 4)

    # Visualize with animation
    TransformVisualizer(dim=2) \
        .with_hold_time(1.0) \
        .with_transition_time(0.5) \
        .with_viewport((-5, 5), (-5, 5)) \
        .with_shape(square) \
        .show()

if __name__ == "__main__":
    main()
