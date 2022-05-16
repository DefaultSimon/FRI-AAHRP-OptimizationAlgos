from random import Random
from typing import List


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
