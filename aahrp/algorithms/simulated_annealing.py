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
    print("Random point:", min_point)
    print("Value:", min_value)
    print(f"Temperature: {current_temp}, max_temperature: {max_temperature}, min_temperature: {min_temperature}, cooling_rate: {cooling_rate}, step: {step}")

    while current_temp > min_temperature and iterations < max_iterations:
        neighbors = get_neighbors(min_point, bounds_lower, bounds_upper, step)
        neighbor = random.choice(neighbors)
        neighbor_value = function(*neighbor)
        print(f"neighbors: {neighbors}, neighbor: {neighbor}, neighbor_value: {neighbor_value}")

        if neighbor_value < min_value:
            # If neighbor is better, move to it
            min_point = neighbor
            min_value = neighbor_value
            print(f"New min_point: {min_point}, min_value: {min_value}")
        else:
            # If neighbor is worse, move to it with probability based on temperature
            #probability = math.exp(-abs(neighbor_value - min_value) / current_temp)
            diff = neighbor_value - min_value
            probability = math.exp(-diff / current_temp)

            #if probability > random.uniform(0, 1):
            if diff < 0 or random.uniform(0, 1) < probability:
                min_point = neighbor
                min_value = neighbor_value

            print(f"diff: {diff}, probability: {probability}, new min_point: {min_point}, min_value: {min_value}")

        # Decrease temperature
        current_temp *= cooling_rate
        iterations += 1
        print(f"current_temp: {current_temp}, iterations: {iterations}")

    print(f"Final min_point: {min_point}, min_value: {min_value}")
    return min_value, min_point
