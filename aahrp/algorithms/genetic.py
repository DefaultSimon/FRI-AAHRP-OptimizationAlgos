from dataclasses import dataclass
from random import Random
from typing import List, Callable, Tuple

from aahrp.utilities import select_random_point


@dataclass
class GeneticSolution:
    best_value: float
    best_solution: List[float]


def genetic_algorithm(
        function: Callable[..., float],
        dimensions: int,
        bounds_lower: List[float],
        bounds_upper: List[float],
        seed: float,
        max_generations: int,
        population_size: int = 50,
        parent_selection_count: int = 25,
        mutation_probability: float = 0.05
) -> GeneticSolution:
    # Independent random generator seeded with the seed parameter.
    random: Random = Random(seed)

    ## Utility functions
    def select_parents(
            parent_components: List[List[float]],
            parent_scores: List[float],
            num_parents: int,
    ) -> List[List[float]]:
        combined: List[Tuple[List[float], float]] = list(zip(parent_components, parent_scores))
        combined_sorted: List[Tuple[List[float], float]] = sorted(combined, key=lambda p: p[1], reverse=True)
        # TODO This currently selects the top num_parents, add a bit of randomization.
        return [c[0] for c in combined_sorted[:num_parents]]

    def generate_child(parent_a: List[float], parent_b: List[float]) -> List[float]:
        # 1. Perform crossover.
        crossover_index: int = random.randint(1, len(parent_b) - 2)
        child_values: List[float] = parent_a[:crossover_index] + parent_b[crossover_index:]

        # 2. Perform random mutation.
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
    best_score_candidate: List[float] = current_population[0]

    for generation_index in range(max_generations):
        # Calculate values of each individual in current population.
        population_scores: List[float] = [function(*values) for values in current_population]

        # Update the best score if a better one has been found.
        score_value: float = min(population_scores)
        score_candidate: List[float] = current_population[population_scores.index(best_score_value)]

        if score_value < best_score_value:
            best_score_value = score_value
            best_score_candidate = score_candidate

        # Select parents and generate next generation using them (crossover).
        parents: List[List[float]] = select_parents(current_population, population_scores, parent_selection_count)

        new_population: List[List[float]] = []
        for _ in range(population_size):
            first_parent: List[float] = random.choice(parents)
            second_parent: List[float] = random.choice(parents)

            child: List[float] = generate_child(first_parent, second_parent)
            new_population.append(child)

        current_population = new_population

    return GeneticSolution(
        best_value=best_score_value,
        best_solution=best_score_candidate
    )


