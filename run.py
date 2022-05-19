from typing import List, Type

from aahrp.functions import \
    Function, Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor, \
    Expo, Modlangerman, EMichalewicz, Shekelfox5, Schwefel
from aahrp.algorithms.genetic import genetic_algorithm
from aahrp.algorithms.hillclimb import hill_climbing_algorithm
from aahrp.timer import Timer

OBJECTIVE_FUNCTIONS: List[Type[Function]] = [
    Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor,
    Expo, Modlangerman, EMichalewicz, Shekelfox5, Schwefel
]

# 100 fixed seeds.
SEEDS: List[int] = [
    4612, 10, 4620, 2575, 5137, 8210, 6170, 2090, 1067, 1066, 1583,
    4656, 49, 1593, 7225, 9277, 4162, 1092, 5703, 1608, 5200, 7251,
    9302, 1632, 3680, 2145, 2659, 5222, 2664, 7787, 108, 3693, 2162,
    6268, 7818, 8337, 2209, 171, 4268, 684, 5292, 688, 2225, 5305,
    8893, 1738, 8395, 7370, 1231, 2781, 6889, 2793, 1774, 1775, 7922,
    8437, 9481, 2314, 1801, 8982, 3874, 4904, 9522, 4918, 3916, 2919,
    4969, 4458, 6509, 7024, 3442, 3957, 4472, 3453, 895, 3465, 5514,
    2955, 1418, 7057, 6556, 6582, 440, 8633, 8127, 5571, 6098, 2515,
    5595, 1500, 4574, 8674, 8162, 8677, 8171, 5612, 4590, 9710, 5113,
    9723
]


####
# Helpers for individual/combined runs
####
def run_genetic_algorithm(
        function: Type[Function],
        number_of_runs: int = len(SEEDS),
        max_generations: int = 2500,
        population_size: int = 250,
        parents_selected: int = 25,
        random_parent_probability: float = 0.015,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.015,
) -> float:
    if number_of_runs > len(SEEDS):
        raise KeyError(f"Not enough pre-generated seeds for {number_of_runs} runs, "
                       f"generate some more at the top of the script.")

    solutions: List[float] = []

    for run_index in range(number_of_runs):
        result: float = genetic_algorithm(
            function=function.function,
            dimensions=function.dimensions(),
            bounds_lower=function.bounds_lower(),
            bounds_upper=function.bounds_upper(),
            seed=SEEDS[run_index],
            max_generations=max_generations,
            population_size=population_size,
            parents_selected=parents_selected,
            random_parent_probability=random_parent_probability,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
        )

        solutions.append(result)

    return min(solutions)


def run_hill_climb_algorithm(
        function: Type[Function],
        num_runs: int = 1
) -> float:
    solutions: List[float] = []

    for run_index in range(num_runs):
        result: float = hill_climbing_algorithm(
            function=function.function,
            dimensions=function.dimensions(),
            bounds_lower=function.bounds_lower(),
            bounds_upper=function.bounds_upper(),
            seed=SEEDS[run_index]
        )

        solutions.append(result)

    return min(solutions)


####
# Main test functions
####
def test_genetic(
        number_of_runs_per_function: int,
        max_generations_per_run: int,
        population_size: int,
        parents_selected: int,
        random_parent_probability: float,
        crossover_probability: float,
        mutation_probability: float,
):
    for index, function in enumerate(OBJECTIVE_FUNCTIONS):
        header: str = f"[{function.__name__} | {index + 1:2d} of {len(OBJECTIVE_FUNCTIONS):2d}]"
        print(f"{header} Running genetic algorithm ...")

        timer: Timer = Timer()
        with timer:
            best_result: float = run_genetic_algorithm(
                function,
                number_of_runs=number_of_runs_per_function,
                max_generations=max_generations_per_run,
                population_size=population_size,
                parents_selected=parents_selected,
                random_parent_probability=random_parent_probability,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )

        print(f"{header} Time to best solution: {round(timer.get_delta(), 2)} seconds.")
        print(f"{header} Best solution: {best_result}")
        print(f"{header} Optimal solution: {function.global_optimum()}")
        print()


def test_hill_climb(num_runs_per_function: int = len(SEEDS)):
    for index, function in enumerate(OBJECTIVE_FUNCTIONS):
        header: str = f"[{function.__name__} | {index + 1:2d} of {len(OBJECTIVE_FUNCTIONS):2d}]"
        print(f"{header} Running hill climb algorithm ...")

        timer: Timer = Timer()
        with timer:
            best_result: float = run_hill_climb_algorithm(
                function,
                num_runs=num_runs_per_function,
            )

        print(f"{header} Time to best solution: {round(timer.get_delta(), 2)} seconds.")
        print(f"{header} Best solution: {best_result}")
        print(f"{header} Optimal solution: {function.global_optimum()}")
        print()


####
# Main
####
def main():
    ## Genetic
    print(f"{'=' * 6} GENETIC {'=' * 6}")

    GENETIC_NUMBER_OF_RUNS: int = len(SEEDS)
    GENETIC_MAX_GENERATIONS: int = 2000
    GENETIC_POPULATION_SIZE: int = 250
    GENETIC_PARENTS_FOR_NEW_GENERATION: int = 25
    GENETIC_PARENT_RANDOM_SELECTION_PROBABILITY: float = 0.015
    GENETIC_CROSSOVER_PROBABILITY: float = 0.9
    GENETIC_MUTATION_PROBABILITY: float = 0.015

    print(f"Running genetic algorithm over {len(OBJECTIVE_FUNCTIONS)} functions ...\n"
          f"\tNumber of runs: {GENETIC_NUMBER_OF_RUNS}\n"
          f"\tSimulated generations: {GENETIC_MAX_GENERATIONS}\n"
          f"\tPopulation size: {GENETIC_POPULATION_SIZE}\n"
          f"\tNumber of parents for next generation: {GENETIC_PARENTS_FOR_NEW_GENERATION}\n"
          f"\tParent random selection probability: {GENETIC_PARENT_RANDOM_SELECTION_PROBABILITY}\n"
          f"\tCrossover probability: {GENETIC_CROSSOVER_PROBABILITY}\n"
          f"\tMutation probability: {GENETIC_MUTATION_PROBABILITY}")
    print()
    
    test_genetic(
        number_of_runs_per_function=GENETIC_NUMBER_OF_RUNS,
        max_generations_per_run=GENETIC_MAX_GENERATIONS,
        population_size=GENETIC_POPULATION_SIZE,
        parents_selected=GENETIC_PARENTS_FOR_NEW_GENERATION,
        random_parent_probability=GENETIC_PARENT_RANDOM_SELECTION_PROBABILITY,
        crossover_probability=GENETIC_CROSSOVER_PROBABILITY,
        mutation_probability=GENETIC_MUTATION_PROBABILITY,
    )

    print(f"{'=' * 6}")

    print()
    print()

    ## Hill climbing
    print(f"{'=' * 6} HILL CLIMBING {'=' * 6}")

    print(f"Running hill climb algorithm over {len(OBJECTIVE_FUNCTIONS)} functions ...")
    print()

    test_hill_climb()

    print(f"{'=' * 6}")


if __name__ == '__main__':
    main()
