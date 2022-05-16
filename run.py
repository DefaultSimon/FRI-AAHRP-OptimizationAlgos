from typing import List, Type

from aahrp.functions import \
    Function, Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor, \
    Expo, Modlangerman, EMichalewicz
from aahrp.algorithms.genetic import genetic_algorithm, GeneticSolution
from aahrp.timer import Timer

OBJECTIVE_FUNCTIONS: List[Type[Function]] = [
    Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor,
    Expo, Modlangerman, EMichalewicz,
    # TODO Add remaining functions.
]

# Ten seeds for now.
SEEDS: List[float] = [781, 977, 783, 210, 826, 530, 970, 269, 133, 108]

####
# Helpers for individual/combined runs
####
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
            population_size=100,
            parent_selection_count=75,
            parent_suboptimal_selection_probability=0.05,
            crossover_probability=0.9,
            mutation_probability=0.05,
        )

        solutions.append(result.best_value)

    return min(solutions)

####
# Main test functions
####
def test_genetic(
        num_runs_per_function: int = len(SEEDS),
        max_generations_per_run: int = 100,
):
    for index, function in enumerate(OBJECTIVE_FUNCTIONS):
        header: str = f"[{function.__name__:14s}|{index + 1:2d}/{len(OBJECTIVE_FUNCTIONS):2d}]"
        print(f"{header} Testing ...")

        timer: Timer = Timer()
        with timer:
            best_result: float = run_genetic_algorithm(
                function,
                num_runs=num_runs_per_function,
                max_generations=max_generations_per_run,
            )

        print(f"{header} Time to best solution: {round(timer.get_delta(), 2)}")
        print(f"{header} Best solution: {best_result}")
        print()


####
# Main
####
def main():
    print(f"{'=' * 6} GENETIC {'=' * 6}")

    GENETIC_GENERATIONS_PER_RUN = 10000

    print(f"Running genetic algorithm over {len(OBJECTIVE_FUNCTIONS)} functions "
          f"({GENETIC_GENERATIONS_PER_RUN} generations per run).")
    test_genetic(max_generations_per_run=GENETIC_GENERATIONS_PER_RUN)

    print(f"{'=' * 6}")


if __name__ == '__main__':
    main()
