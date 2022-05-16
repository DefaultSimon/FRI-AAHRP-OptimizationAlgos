from random import Random
from typing import List
from itertools import product


def select_random_point(
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        random: Random,
) -> List[float]:
    """
    Generate a random point in N dimensions based on the lower and upper bounds.
    N is the length of either (both) bound lists.

    :param dimensions: Number of dimensions to generate.
    :param bounds_lower: Lower bounds for components.
    :param bounds_upper: Upper bounds for components.
    :param random: Random instance to generate the components with; ensures reproducibility.
    :return: List of values representing a point in N-dimensional space.
    """
    return [
        random.uniform(bounds_lower[index], bounds_upper[index])
        for index in range(dimensions)
    ]


def get_neighbors(
        point: List[float],
        bounds_lower: List[float],
        bounds_upper: List[float],
        step: float = 0.1
) -> List[List[float]]:
    """
    Return all neighbors of a point
    :param point:
    :param dimensions:
    :param bounds_lower:
    :param bounds_upper:
    :param step:
    :return:
    """
    offsets = list(product(*([-step, 0, step] for i in point)))
    neighbors: List[List[float]] = []

    for offset in offsets:
        neighbor = []
        for i in range(len(point)):
            val = point[i] + offset[i]

            if bounds_lower[i] <= val <= bounds_upper[i]:
                neighbor.append(val)

        if neighbor != point and len(neighbor) == len(point):
            neighbors.append(neighbor)

    return neighbors
