from multiprocessing import Pool
from typing import TypeVar, List, Callable, Tuple

T = TypeVar("T")


def run_concurrently(
        function: Callable[..., T],
        list_of_argument_tuples: List[Tuple],
        concurrency: int,
        chunk_size: int = 1,
) -> List[T]:
    pool: Pool = Pool(processes=concurrency)

    results: List[T] = pool.starmap(
        func=function,
        iterable=list_of_argument_tuples,
        chunksize=chunk_size
    )

    pool.close()
    pool.terminate()

    return results
