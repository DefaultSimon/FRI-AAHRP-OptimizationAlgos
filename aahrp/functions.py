"""
This module contains implementations of various functions we want to find optimums for.
"""
from math import sin, sqrt


# A sort of "container class" for Schaffer1's function and other values we need (such as the global optimum).
class Schaffer1:
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
    def global_optimum() -> float:
        return 0
