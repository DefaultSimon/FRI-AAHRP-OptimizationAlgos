"""
This module contains implementations of various functions we want to find optimums for.
"""
from abc import ABCMeta
from typing import List
from math import sin, sqrt, cos, pi, exp


# A sort of "container class" for functions and other values we need (such as the global optimum).
class Function(metaclass=ABCMeta):
    """
    Base class for all functions we want to find optimums for, useful for autocompletion.
    """
    @staticmethod
    def function(*args: float) -> float:
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
    def function(*args: float) -> float:
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
    def function(*args: float) -> float:
        intermediate_sum: float = 0
        product: float = 1

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


class PriceTransistor(Function):
    """
    PriceTransistor, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L545
    """
    COEFFICIENTS: List[List[float]] = [
        [0.485, 0.752, 0.869, 0.982],
        [0.369, 1.254, 0.703, 1.455],
        [5.2095, 10.0677, 22.9274, 20.2153],
        [23.3037, 101.779, 111.461, 191.267],
        [28.5132, 111.8467, 134.3884, 211.4823],
    ]

    @staticmethod
    def function(*args: float) -> float:
        coefs: List[List[float]] = PriceTransistor.COEFFICIENTS

        sqr_sums: float = 0

        for k in range(4):
            alpha: float = \
                (1 - args[0] * args[1]) \
                * args[2] \
                * (
                    exp(
                        args[4] * (
                            coefs[0][k]
                            - (0.001 * coefs[2][k] * args[6])
                            - (0.001 * args[7] * coefs[4][k])
                        )
                    )
                    - 1
                ) \
                - coefs[4][k] \
                + (coefs[3][k] * args[1])

            beta: float = \
                (1 - args[0] * args[1]) \
                * args[3] \
                * (
                    exp(
                        args[5] * (
                            coefs[0][k]
                            - coefs[1][k]
                            - (0.001 * coefs[2][k] * args[6])
                            + (coefs[3][k] * 0.001 * args[8])
                        )
                    )
                    - 1
                ) \
                - (coefs[4][k] * args[0]) \
                + coefs[3][k]

            sqr_sums += alpha ** 2 + beta ** 2

        return (args[0] * args[2] - args[1] * args[3]) ** 2 + sqr_sums

    @staticmethod
    def dimensions() -> int:
        return 9

    @staticmethod
    def bounds_lower() -> List[float]:
        return [0] * 9

    @staticmethod
    def bounds_upper() -> List[float]:
        return [10] * 9

    @staticmethod
    def global_optimum() -> float:
        return 0


class Expo(Function):
    """
    Expo, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L153
    """
    @staticmethod
    def function(*args: float) -> float:
        result: float = sum([args[j] ** 2 for j in range(10)])
        return -1 * exp(-0.5 * result)

    @staticmethod
    def dimensions() -> int:
        return 10

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-12] * 10

    @staticmethod
    def bounds_upper() -> List[float]:
        return [10] * 10

    @staticmethod
    def global_optimum() -> float:
        return -1


class Modlangerman(Function):
    """
    Modlangerman, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L407
    """
    COEFFICIENTS: List[List[float]] = [
        [9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
        [9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
        [8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
        [2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
        [8.074, 8.777, 3.467, 1.867, 6.708, 6.349, 4.534, 0.276, 7.633, 1.56],
    ]
    COEFFICIENTS2: List[float] = [0.806, 0.517, 0.1, 0.908, 0.965]

    @staticmethod
    def function(*args: float) -> float:
        coef: List[List[float]] = Modlangerman.COEFFICIENTS
        coef2: List[float] = Modlangerman.COEFFICIENTS2

        func_sum: float = 0

        for i in range(5):
            dist: float = 0
            for j in range(10):
                dist += (args[j] - coef[i][j]) ** 2

            func_sum -= coef2[i] * exp(-1 * dist / pi) * cos(pi * dist)

        return func_sum

    @staticmethod
    def dimensions() -> int:
        return 10

    @staticmethod
    def bounds_lower() -> List[float]:
        return [0] * 10

    @staticmethod
    def bounds_upper() -> List[float]:
        return [10] * 10

    @staticmethod
    def global_optimum() -> float:
        return -0.9650
