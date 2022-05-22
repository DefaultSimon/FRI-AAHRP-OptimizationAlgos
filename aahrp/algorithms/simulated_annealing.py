import math
import random
from random import Random
from typing import List, Callable

from aahrp.utilities import select_random_point, get_neighbors


def simulated_annealing(
        function: Callable[..., float],
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        seed: float = -1,
        step: float = 1,
        temperature: float = 100  # Based on best temperature from Assignment 4
) -> float:
    # Set initial values
    min_temp = 0.1
    current_temp = temperature
    min_point = select_random_point(dimensions, bounds_lower, bounds_upper, Random(seed))
    min_value = function(*min_point)

    while current_temp > min_temp:
        neighbors = get_neighbors(min_point, bounds_lower, bounds_upper, step)
        neighbor = random.choice(neighbors)
        neighbor_value = function(*neighbor)

        if neighbor_value < min_value:
            # If neighbor is better, move to it
            min_point = neighbor
            min_value = neighbor_value
        else:
            # If neighbor is worse, move to it with probability based on temperature
            probability = math.exp(-(neighbor_value - min_value) / current_temp)

            if probability > random.uniform(0, 1):
                min_point = neighbor
                min_value = neighbor_value

        # Decrease temperature
        current_temp *= 0.99

    return min_value
