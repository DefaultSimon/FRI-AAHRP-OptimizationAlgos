import math
import random
from random import Random
from typing import List, Callable, Tuple

from aahrp.utilities import select_random_point, get_neighbors


def simulated_annealing(
        function: Callable[..., float],
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        step: float = 1,
        min_temperature: float = 0.001,
        max_temperature: float = 100,  # Based on best temperature from Assignment 4
        cooling_rate: float = 0.95,
        max_iterations: int = 10000,
) -> Tuple[float, List[float]]:
    # Set initial values
    current_temp = max_temperature
    min_point = select_random_point(dimensions, bounds_lower, bounds_upper, Random())
    min_value = function(*min_point)
    iterations = 0

    while current_temp > min_temperature and iterations < max_iterations:
        neighbors = get_neighbors(min_point, bounds_lower, bounds_upper, step)
        neighbor = random.choice(neighbors)
        neighbor_value = function(*neighbor)

        if neighbor_value < min_value:
            # If neighbor is better, move to it
            min_point = neighbor
            min_value = neighbor_value
        else:
            # If neighbor is worse, move to it with probability based on temperature
            probability = math.exp(-abs(neighbor_value - min_value) / current_temp)

            if probability > random.uniform(0, 1):
                min_point = neighbor
                min_value = neighbor_value

        # Decrease temperature
        current_temp *= cooling_rate
        iterations += 1

    return min_value, min_point
