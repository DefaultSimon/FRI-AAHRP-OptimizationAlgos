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

        # TODO PriceTransistor.function(0.9, 0.45, 1, 2, 8, 8, 5, 1, 2) == 1.7881583721892765e-07
        #   but global optimum is 0. Surely this is just float inaccuracy?

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


class EMichalewicz(Function):
    """
    EMichalewicz, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L120
    """
    @staticmethod
    def function(*args: float) -> float:
        cos_t: float = cos(pi / 6)
        sin_t: float = sin(pi / 6)

        # TODO EMichalewicz.function(2.683, 0.259, 2.074, 1.023, 1.720) == -4.685305462075563
        #   but global optimum is -4.6877. Is this just float inaccuracy?

        result_y: List[float] = [0] * 10
        for j in range(0, 5 - 1, 2):
            result_y[j] = args[j] * cos_t - args[j + 1] * sin_t
            result_y[j + 1] = args[j] * sin_t + args[j + 1] * cos_t

        result_y[4] = args[4]

        func_sum: float = 0

        for k in range(0, 5):
            func_sum -= \
                sin(result_y[k]) \
                * (
                    sin(
                        (k + 1) * result_y[k] * result_y[k] / pi
                    ) ** (2 * 10)
                )

        return func_sum

    @staticmethod
    def dimensions() -> int:
        return 5

    @staticmethod
    def bounds_lower() -> List[float]:
        return [0] * 5

    @staticmethod
    def bounds_upper() -> List[float]:
        return [pi] * 5

    @staticmethod
    def global_optimum() -> float:
        return -4.6877


class Shekelfox5(Function):
    """
    Shekelfox5, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L750
    """
    COEFFICIENTS: List[List[float]] = [
        [9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
        [9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
        [8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
        [2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
        [8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567],
        [7.650, 5.658, 0.720, 2.764, 3.278, 5.283, 7.474, 6.274, 1.409, 8.208],
        [1.256, 3.605, 8.623, 6.905, 4.584, 8.133, 6.071, 6.888, 4.187, 5.448],
        [8.314, 2.261, 4.224, 1.781, 4.124, 0.932, 8.129, 8.658, 1.208, 5.762],
        [0.226, 8.858, 1.420, 0.945, 1.622, 4.698, 6.228, 9.096, 0.972, 7.637],
        [7.305, 2.228, 1.242, 5.928, 9.133, 1.826, 4.060, 5.204, 8.713, 8.247],
        [0.652, 7.027, 0.508, 4.876, 8.807, 4.632, 5.808, 6.937, 3.291, 7.016],
        [2.699, 3.516, 5.874, 4.119, 4.461, 7.496, 8.817, 0.690, 6.593, 9.789],
        [8.327, 3.897, 2.017, 9.570, 9.825, 1.150, 1.395, 3.885, 6.354, 0.109],
        [2.132, 7.006, 7.136, 2.641, 1.882, 5.943, 7.273, 7.691, 2.880, 0.564],
        [4.707, 5.579, 4.080, 0.581, 9.698, 8.542, 8.077, 8.515, 9.231, 4.670],
        [8.304, 7.559, 8.567, 0.322, 7.128, 8.392, 1.472, 8.524, 2.277, 7.826],
        [8.632, 4.409, 4.832, 5.768, 7.050, 6.715, 1.711, 4.323, 4.405, 4.591],
        [4.887, 9.112, 0.170, 8.967, 9.693, 9.867, 7.508, 7.770, 8.382, 6.740],
        [2.440, 6.686, 4.299, 1.007, 7.008, 1.427, 9.398, 8.480, 9.950, 1.675],
        [6.306, 8.583, 6.084, 1.138, 4.350, 3.134, 7.853, 6.061, 7.457, 2.258],
        [0.652, 2.343, 1.370, 0.821, 1.310, 1.063, 0.689, 8.819, 8.833, 9.070],
        [5.558, 1.272, 5.756, 9.857, 2.279, 2.764, 1.284, 1.677, 1.244, 1.234],
        [3.352, 7.549, 9.817, 9.437, 8.687, 4.167, 2.570, 6.540, 0.228, 0.027],
        [8.798, 0.880, 2.370, 0.168, 1.701, 3.680, 1.231, 2.390, 2.499, 0.064],
        [1.460, 8.057, 1.336, 7.217, 7.914, 3.615, 9.981, 9.198, 5.292, 1.224],
        [0.432, 8.645, 8.774, 0.249, 8.081, 7.461, 4.416, 0.652, 4.002, 4.644],
        [0.679, 2.800, 5.523, 3.049, 2.968, 7.225, 6.730, 4.199, 9.614, 9.229],
        [4.263, 1.074, 7.286, 5.599, 8.291, 5.200, 9.214, 8.272, 4.398, 4.506],
        [9.496, 4.830, 3.150, 8.270, 5.079, 1.231, 5.731, 9.494, 1.883, 9.732],
        [4.138, 2.562, 2.532, 9.661, 5.611, 5.500, 6.886, 2.341, 9.699, 6.500]
    ]
    COEFFICIENTS2: List[float] = [
        0.806, 0.517, 0.1, 0.908, 0.965, 0.669, 0.524, 0.902, 0.531, 0.876,
        0.462, 0.491, 0.463, 0.714, 0.352, 0.869, 0.813, 0.811, 0.828, 0.964,
        0.789, 0.360, 0.369, 0.992, 0.332, 0.817, 0.632, 0.883, 0.608, 0.326
    ]

    @staticmethod
    def function(*args: float) -> float:
        coef: List[List[float]] = Shekelfox5.COEFFICIENTS
        coef2: List[float] = Shekelfox5.COEFFICIENTS2

        func_sum: float = 0

        for j in range(30):
            sp: float = 0
            for i in range(Shekelfox5.dimensions()):
                sp += (args[i] - coef[j][i]) ** 2

            func_sum -= 1.0 / (sp + coef2[j])

        return func_sum

    @staticmethod
    def dimensions() -> int:
        return 5

    @staticmethod
    def bounds_lower() -> List[float]:
        return [0] * 5

    @staticmethod
    def bounds_upper() -> List[float]:
        return [10] * 5

    @staticmethod
    def global_optimum() -> float:
        return -10.4056


class Schwefel(Function):
    """
    Schwefel, as implemented in globalOptTests.
    See https://github.com/cran/globalOptTests/blob/master/src/objFun.c#L668
    """
    @staticmethod
    def function(*args: float) -> float:
        func_sum: float = 0

        for j in range(Schwefel.dimensions()):
            func_sum += args[j] * sin(sqrt(abs(args[j])))

        return -func_sum

    @staticmethod
    def dimensions() -> int:
        return 10

    @staticmethod
    def bounds_lower() -> List[float]:
        return [-500] * 10

    @staticmethod
    def bounds_upper() -> List[float]:
        return [500] * 10

    @staticmethod
    def global_optimum() -> float:
        return -4189.8289
