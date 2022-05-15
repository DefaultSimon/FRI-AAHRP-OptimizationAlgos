"""
This module contains implementations of various functions we want to find optimums for.
"""
from abc import ABCMeta
from typing import List
from math import sin, sqrt, cos, pi


# A sort of "container class" for functions and other values we need (such as the global optimum).
class Function(metaclass=ABCMeta):
    """
    Base class for all functions we want to find optimums for, useful for autocompletion.
    """
    @staticmethod
    def function(*args) -> float:
        raise NotImplementedError()

    @staticmethod
    def dimensions() -> int:
        raise NotImplementedError()

    @staticmethod
    def bounds_lower() -> List[float]:
        raise NotImplementedError()

    @staticmethod
    def bounds_upper() -> List[float]:
        raise NotImplementedError()

    @staticmethod
    def global_optimum() -> float:
        raise NotImplementedError()


####
# Specific implementations
####
class Schaffer1(Function):
    """
    Schaffer1, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L621
    """
    @staticmethod
    def function(x: float, y: float) -> float:
        num: float = sin(sqrt(x ** 2 + y ** 2)) ** 2 - 0.5
        den: float = (1 + 0.001 * (x ** 2 + y ** 2)) ** 2
        return 0.5 + num / den

    @staticmethod
    def dimensions() -> int:
        return 2

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-120, -120]

    @staticmethod
    def bounds_upper() -> List[float]:
        return [100, 100]

    @staticmethod
    def global_optimum() -> float:
        return 0


class Schaffer2(Function):
    """
    Schaffer2, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L635
    """
    @staticmethod
    def function(x: float, y: float) -> float:
        product_1: float = (x ** 2 + y ** 2) ** 0.25
        product_2: float = (50 * (x ** 2 + y ** 2)) ** 0.1
        return product_1 * (sin(sin(product_2)) + 1)

    @staticmethod
    def dimensions() -> int:
        return 2

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-120, -120]

    @staticmethod
    def bounds_upper() -> List[float]:
        return [100, 100]

    @staticmethod
    def global_optimum() -> float:
        return 0


class Salomon(Function):
    """
    Salomon, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L604
    """
    @staticmethod
    def function(*args: List[float]) -> float:
        fun_sum = sqrt(sum([args[i] ** 2 for i in range(5)]))
        return -1 * cos(2 * pi * fun_sum) + 0.1 * fun_sum + 1

    @staticmethod
    def dimensions() -> int:
        return 5

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-120] * 5

    @staticmethod
    def bounds_upper() -> List[float]:
        return [100] * 5

    @staticmethod
    def global_optimum() -> float:
        return 0


class Griewank(Function):
    """
    Griewank, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L180
    """
    @staticmethod
    def function(*args: List[float]) -> float:
        intermediate_sum: float = 0
        product: float = 0

        for i in range(10):
            intermediate_sum += args[i] ** 2
            product *= cos(args[i] / sqrt(i + 1))

        return (intermediate_sum / 4000) - product + 1

    @staticmethod
    def dimensions() -> int:
        return 10

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-550] * 10

    @staticmethod
    def bounds_upper() -> List[float]:
        return [500] * 10

    @staticmethod
    def global_optimum() -> float:
        return 0

