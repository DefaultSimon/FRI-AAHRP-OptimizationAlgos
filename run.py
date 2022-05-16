from typing import List, Type

from aahrp.functions import \
    Function, Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor, \
    Expo, Modlangerman, EMichalewicz
from aahrp.algorithms.genetic import genetic_algorithm, GeneticSolution


OBJECTIVE_FUNCTIONS: List[Type[Function]] = [
    Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor,
    Expo, Modlangerman, EMichalewicz,
    # TODO Add remaining functions.
]

# Ten seeds for now.
SEEDS: List[float] = [781, 977, 783, 210, 826, 530, 970, 269, 133, 108]


def run_genetic_algorithm(
        function: Type[Function],
        num_runs: int = 1,
        max_generations: int = 100
) -> float:
    solutions: List[float] = []
    for run_index in range(num_runs):
        result: GeneticSolution = genetic_algorithm(
            function=function.function,
            dimensions=function.dimensions(),
            bounds_lower=function.bounds_lower(),
            bounds_upper=function.bounds_upper(),
            seed=SEEDS[run_index],
            max_generations=max_generations,
            population_size=50,
            parent_selection_count=25,
            crossover_probability=0.9,
            mutation_probability=0.05,
        )

        solutions.append(result.best_value)

    return min(solutions)
