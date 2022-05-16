from random import Random
from typing import List, Callable

from aahrp.utilities import select_random_point, get_neighbors


def hill_climbing_algorithm(
        function: Callable[..., float],
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        max_iterations: int = 1000,
        seed: float = -1,
        step: float = 1
) -> float:
    # Initialize value with seed if provided
    random: Random
    if seed == -1:
        random = Random()
    else:
        random = Random(seed)

    min_node = select_random_point(dimensions, bounds_lower, bounds_upper, random)
    min_val = function(*min_node)

    for i in range(max_iterations):
        neighbors = get_neighbors(min_node, bounds_lower, bounds_upper, step)

        next_node: List[float] = []
        next_val: float = min_val

        for neighbor in neighbors:
            if function(*neighbor) > next_val:
                next_node = neighbor
                next_val = function(*neighbor)

        if next_val > min_val:
            min_node = next_node
            min_val = next_val
        else:
            return min_val
