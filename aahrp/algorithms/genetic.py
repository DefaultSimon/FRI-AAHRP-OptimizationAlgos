from random import Random
from typing import List, Callable, Tuple

from aahrp.utilities import select_random_point


def genetic_algorithm(
        function: Callable[..., float],
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        seed: float,
        max_generations: int,
        population_size: int = 50,
        parent_selection_count: int = 25,
        parent_suboptimal_selection_probability: float = 0.05,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.05
) -> float:
    # Independent random generator seeded with the seed parameter.
    random: Random = Random(seed)

    ## Utility functions
    def select_parents(
            parent_components: List[List[float]],
            parent_scores: List[float],
            num_parents: int,
    ) -> List[List[float]]:
        combined: List[Tuple[List[float], float]] = list(zip(parent_components, parent_scores))
        # Sorted in ascending order.
        combined_sorted: List[Tuple[List[float], float]] = sorted(combined, key=lambda p: p[1])

        selected_parents: List[List[float]] = []
        for _ in range(parent_selection_count):
            if random.random() < parent_suboptimal_selection_probability:
                # Select a random parent instead of the remaining best one.
                selected_parents.append(random.choice(combined)[0])
            else:
                # Select the current best parent and remove it from the pool.
                selected_parents.append(combined_sorted.pop(0)[0])

        return [c[0] for c in combined_sorted[:num_parents]]

    def generate_child(parent_a: List[float], parent_b: List[float]) -> List[float]:
        # 1. Perform crossover with probability.
        child_values: List[float]

        if random.random() < crossover_probability:
            # Perform crossover!
            crossover_index: int
            if dimensions == 2:
                crossover_index = 1
            else:
                crossover_index = random.randint(1, len(parent_b) - 2)

            child_values = parent_a[:crossover_index] + parent_b[crossover_index:]
        else:
            # Pick one of the parents to descend to the next generation.
            child_values = random.choice((parent_a, parent_b))

        # 2. Perform mutation with probability.
        for index in range(len(child_values)):
            if random.random() < mutation_probability:
                # Mutation occurs: value at index is mutated.
                child_values[index] = random.uniform(bounds_lower[index], bounds_upper[index])

        return child_values

    # Generate a random initial population.
    current_population: List[List[float]] = [
        select_random_point(dimensions, bounds_lower, bounds_upper, random) for _ in range(population_size)
    ]

    best_score_value: float = function(*current_population[0])

    for generation_index in range(max_generations):
        # Calculate values of each individual in current population.
        population_scores: List[float] = [function(*values) for values in current_population]

        # Update the best score if a better one has been found.
        score_value, score_candidate = min(
            zip(population_scores, current_population),
            key=lambda z: z[0]
        )

        if score_value < best_score_value:
            best_score_value = score_value

        # Select parents and generate next generation using them (crossover).
        parents: List[List[float]] = select_parents(current_population, population_scores, parent_selection_count)

        new_population: List[List[float]] = []
        for _ in range(population_size):
            first_parent: List[float] = random.choice(parents)
            second_parent: List[float] = random.choice(parents)

            child: List[float] = generate_child(first_parent, second_parent)
            new_population.append(child)

        current_population = new_population

    return best_score_value

