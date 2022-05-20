import multiprocessing
from typing import List, Type, Tuple, Any, Dict, Callable

from aahrp.functions import \
    Function, Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor, \
    Expo, Modlangerman, EMichalewicz, Shekelfox5, Schwefel
from aahrp.algorithms.genetic import genetic_algorithm
from aahrp.algorithms.hillclimb import hill_climbing_algorithm
from aahrp.timer import Timer
from aahrp.parallelization import run_concurrently

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
# GENETIC ALGORITHM
####
def make_genetic_params_fn(
        f: Type[Function],
        max_generations: int,
        population_size: int,
        parents_selected: int,
        random_parent_probability: float,
        crossover_probability: float,
        mutation_probability: float,
) -> Callable[[int], Tuple]:
    # Complex, but kind of useful for this scenario.
    def _inner(seed: int):
        return (
            f.function,
            f.dimensions(),
            f.bounds_lower(),
            f.bounds_upper(),
            seed,
            max_generations,
            population_size,
            parents_selected,
            random_parent_probability,
            crossover_probability,
            mutation_probability
        )
    return _inner

# Because not every function can be optimized best the same way, we experimented and came up with
# a set of genetic algorithm parameters optimized for each individual function.
GENETIC_FUNCTION_TO_OPTIMIZED_PARAMETERS: Dict[Type[Function], Callable[[int], Tuple]] = {
    Schaffer1: make_genetic_params_fn(Schaffer1, max_generations=2000,
                                      population_size=250, parents_selected=25, random_parent_probability=0.015,
                                      crossover_probability=0.3, mutation_probability=0.015),
    Schaffer2: make_genetic_params_fn(Schaffer2, max_generations=2000,
                                      population_size=1000, parents_selected=10, random_parent_probability=0.015,
                                      crossover_probability=0.35, mutation_probability=0.015),
    Salomon: make_genetic_params_fn(Salomon, max_generations=2000,
                                    population_size=500, parents_selected=10, random_parent_probability=0.015,
                                    crossover_probability=0.55, mutation_probability=0.015),
    Griewank: make_genetic_params_fn(Griewank, max_generations=2000,
                                     population_size=1000, parents_selected=10, random_parent_probability=0.015,
                                     crossover_probability=0.9, mutation_probability=0.015),
    PriceTransistor: make_genetic_params_fn(PriceTransistor, max_generations=2000,
                                            population_size=1000, parents_selected=10, random_parent_probability=0.015,
                                            crossover_probability=0.75, mutation_probability=0.015),
    Expo: make_genetic_params_fn(Expo, max_generations=2000,
                                 population_size=1000, parents_selected=10, random_parent_probability=0.015,
                                 crossover_probability=0.8, mutation_probability=0.015),
    Modlangerman: make_genetic_params_fn(Modlangerman, max_generations=2000,
                                         population_size=1000, parents_selected=20, random_parent_probability=0.015,
                                         crossover_probability=0.7, mutation_probability=0.015),
    EMichalewicz: make_genetic_params_fn(EMichalewicz, max_generations=2000,
                                         population_size=1000, parents_selected=10, random_parent_probability=0.015,
                                         crossover_probability=0.8, mutation_probability=0.015),
    Shekelfox5: make_genetic_params_fn(Shekelfox5, max_generations=2000,
                                       population_size=1000, parents_selected=20, random_parent_probability=0.015,
                                       crossover_probability=0.8, mutation_probability=0.015),
    Schwefel: make_genetic_params_fn(Schwefel, max_generations=2000,
                                     population_size=1000, parents_selected=5, random_parent_probability=0.015,
                                     crossover_probability=0.65, mutation_probability=0.015),
}

def run_genetic_algorithm(
        function: Type[Function],
        number_of_runs: int = len(SEEDS),
        concurrency: int = multiprocessing.cpu_count()
) -> float:
    if number_of_runs > len(SEEDS):
        raise KeyError(f"Not enough pre-generated seeds for {number_of_runs} runs, "
                       f"generate some more at the top of the script.")

    concurrency: int = min(concurrency, number_of_runs)
    concurrency_arguments: List[Tuple[Any, ...]] = [
        GENETIC_FUNCTION_TO_OPTIMIZED_PARAMETERS[function](SEEDS[index])
        for index in range(number_of_runs)
    ]

    solutions: List[float] = run_concurrently(
        function=genetic_algorithm,
        list_of_argument_tuples=concurrency_arguments,
        concurrency=concurrency,
        chunk_size=2
    )

    return min(solutions)


####
# HILL CLIMB ALGORITHM
####
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
# MAIN TEST FUNCTIONS
####
def test_genetic(
        number_of_runs_per_function: int,
        concurrency: int,
):
    for index, function in enumerate(OBJECTIVE_FUNCTIONS):
        header: str = f"[{function.__name__.ljust(15)}|{index + 1:2d} of {len(OBJECTIVE_FUNCTIONS):2d}]"
        print(f"{header} Running genetic algorithm ...")

        timer: Timer = Timer()
        with timer:
            best_result: float = run_genetic_algorithm(
                function,
                number_of_runs=number_of_runs_per_function,
                concurrency=concurrency,
            )

        print(f"{header} Time to best solution: {round(timer.get_delta(), 2)} seconds.")
        print(f"{header} Best solution: {best_result}")
        print(f"{header} Optimal solution: {function.global_optimum()}")
        print(f"{header} Distance: {best_result - function.global_optimum()}")
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
    CPU_CORES: int = multiprocessing.cpu_count()

    ## Genetic
    print(f"{'=' * 6} GENETIC {'=' * 6}")
    GENETIC_NUMBER_OF_RUNS: int = len(SEEDS)

    print(f"Running genetic algorithm over {len(OBJECTIVE_FUNCTIONS)} functions ...")
    print()

    test_genetic(
        number_of_runs_per_function=GENETIC_NUMBER_OF_RUNS,
        concurrency=CPU_CORES,
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
