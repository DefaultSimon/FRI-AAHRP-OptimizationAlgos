from typing import List, Type

from aahrp.functions import \
    Function, Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor, \
    Expo, Modlangerman, EMichalewicz, Shekelfox5, Schwefel
from aahrp.algorithms.genetic import genetic_algorithm
from aahrp.timer import Timer

OBJECTIVE_FUNCTIONS: List[Type[Function]] = [
    Schaffer1, Schaffer2, Salomon, Griewank, PriceTransistor,
    Expo, Modlangerman, EMichalewicz, Shekelfox5, Schwefel
]

# 200 seeds for now.
SEEDS: List[int] = [
    665, 351, 612, 998, 801, 32, 905, 989, 276, 838, 932, 905, 131, 535, 36, 551, 466, 631, 493, 180, 249, 280,
    959, 385, 134, 443, 786, 408, 704, 522, 191, 166, 143, 97, 588, 418, 785, 925, 906, 870, 669, 781, 85, 21,
    733, 140, 54, 290, 165, 289, 130, 310, 665, 30, 602, 131, 367, 181, 901, 870, 188, 705, 228, 923, 841, 530,
    117, 835, 264, 576, 581, 412, 286, 126, 693, 446, 228, 53, 231, 600, 973, 243, 601, 609, 834, 59, 731, 848,
    209, 872, 83, 852, 3, 442, 168, 367, 988, 536, 938, 818, 884, 501, 286, 729, 689, 569, 143, 853, 244, 630,
    594, 150, 779, 757, 38, 330, 683, 247, 860, 312, 41, 210, 13, 155, 310, 257, 429, 809, 632, 281, 588, 998,
    234, 358, 239, 201, 202, 32, 354, 593, 79, 918, 49, 937, 149, 210, 685, 487, 495, 372, 997, 115, 997, 250,
    380, 746, 117, 856, 506, 775, 517, 0, 660, 692, 763, 408, 244, 681, 786, 164, 703, 142, 643, 573, 818, 544,
    388, 564, 282, 748, 192, 946, 931, 123, 863, 310, 122, 75, 955, 704, 556, 989, 712, 493, 715, 354, 38, 277, 500, 27
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


####
# Main test functions
####
def test_genetic(
        num_runs_per_function: int = len(SEEDS),
        max_generations_per_run: int = 100,
):
    for index, function in enumerate(OBJECTIVE_FUNCTIONS):
        header: str = f"[{function.__name__} | {index + 1:2d} of {len(OBJECTIVE_FUNCTIONS):2d}]"
        print(f"{header} Running genetic algorithm ...")

        timer: Timer = Timer()
        with timer:
            best_result: float = run_genetic_algorithm(
                function,
                number_of_runs=num_runs_per_function,
                max_generations=max_generations_per_run,
            )

        print(f"{header} Time to best solution: {round(timer.get_delta(), 2)} seconds.")
        print(f"{header} Best solution: {best_result}")
        print()


####
# Main
####
def main():
    print(f"{'=' * 6} GENETIC {'=' * 6}")

    GENETIC_GENERATIONS_PER_RUN = 1000

    print(f"Running genetic algorithm over {len(OBJECTIVE_FUNCTIONS)} functions "
          f"({GENETIC_GENERATIONS_PER_RUN} generations per run).")
    print()
    test_genetic(max_generations_per_run=GENETIC_GENERATIONS_PER_RUN)

    print(f"{'=' * 6}")


if __name__ == '__main__':
    main()
