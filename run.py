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

# 400 seeds for now.
SEEDS: List[int] = [
    1, 4117, 26, 3456, 8220, 4136, 4144, 6192, 8244, 8261, 4174, 8273, 2133, 8278, 2136, 94, 2147, 105,
    6252, 4213, 4224, 8326, 2183, 6280, 6296, 8348, 158, 6308, 6312, 170, 8363, 2230, 182, 6328, 4282,
    191, 6336, 4288, 6341, 6346, 213, 4310, 2272, 4322, 8419, 2277, 229, 6377, 4336, 4348, 256, 4356, 2319,
    8463, 8464, 8471, 287, 8483, 2340, 2341, 6440, 4399, 2352, 307, 6452, 2360, 2368, 328, 336, 8533, 6494,
    2401, 4456, 4472, 392, 2449, 4501, 8597, 4506, 8604, 4512, 4513, 4514, 8614, 6569, 6571, 8622, 436, 2486,
    2488, 441, 444, 8650, 460, 6609, 4577, 2530, 2534, 4582, 2537, 8691, 6646, 4601, 8702, 8703, 2563,
    2567, 6675, 532, 4638, 6686, 6690, 2602, 2606, 8760, 8761, 2620, 8766, 6722, 8778, 2635, 2642, 2646,
    8794, 630, 4727, 4733, 4736, 8833, 8835, 645, 2729, 8876, 688, 8887, 2744, 2751, 4804, 4806, 728,
    2790, 4843, 751, 753, 6898, 757, 760, 6904, 6906, 8960, 8964, 8978, 4882, 2844, 806, 4907, 812, 814,
    6959, 6974, 4927, 2884, 9029, 4937, 9037, 4950, 6998, 9052, 7005, 2914, 9060, 875, 9068, 9075, 2932,
    885, 2950, 2955, 9101, 2971, 5024, 2977, 9121, 9125, 942, 947, 9141, 965, 977, 7121, 3031, 3035, 5084,
    7138, 998, 1003, 5102, 7163, 9217, 1028, 7186, 1043, 1046, 3101, 3106, 9252, 9257, 3121, 9266, 5174,
    1079, 5179, 1083, 7232, 1098, 9296, 5201, 7256, 7265, 9313, 7272, 7276, 7284, 9335, 3199, 7300,
    7306, 5259, 9359, 7316, 7317, 3223, 9373, 9379, 5283, 1212, 7358, 5312, 1217, 9419, 1228, 1230,
    3285, 7382, 7386, 1249, 7395, 1256, 5355, 9464, 5369, 9470, 7431, 9494, 1312, 7459, 3368, 7476,
    7479, 9538, 9541, 3397, 7502, 3406, 3410, 7512, 1369, 7517, 7527, 3436, 3437, 5487, 1394, 3449,
    9600, 7552, 7557, 1414, 1415, 1426, 5525, 1429, 9621, 7576, 5534, 7582, 1441, 1449, 7598, 1458,
    3507, 9656, 1466, 7613, 3523, 9668, 5574, 1480, 9679, 7631, 3539, 7641, 1498, 3552, 3553, 3554,
    3559, 5618, 9717, 5622, 5623, 7676, 1536, 9728, 7688, 1544, 9740, 7695, 9744, 9757, 7710, 5667,
    3623, 5671, 9769, 5673, 5678, 3630, 9783, 1595, 3653, 7760, 7767, 9815, 7772, 9830, 5737, 3693,
    7795, 1651, 1655, 9849, 9855, 3727, 3736, 7839, 9887, 9896, 5802, 3756, 9905, 5814, 1720, 9919,
    9930, 3788, 3789, 5848, 1755, 7929, 5884, 1801, 3861, 3867, 5932, 1839, 3887, 3890, 8005, 3913,
    8010, 8014, 8024, 1892, 1894, 1899, 3973, 1927, 8075, 8080, 6049, 1963, 6073, 4030, 4034, 8136,
    2002, 4057, 2012, 8162, 6121, 2035, 8184, 6137, 2046, 9842, 2345, 6666, 9112, 3352
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
    GENETIC_MAX_GENERATIONS: int = 2500
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
