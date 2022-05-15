"""
This module contains implementations of various functions we want to find optimums for.
"""
from abc import ABCMeta
from typing import List
from math import sin, sqrt


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
