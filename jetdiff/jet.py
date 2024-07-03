from typing import Self, Union

import math
import numpy as np
import numpy.typing as npt

_LN2 = math.log(2)
_LN10 = math.log(10)


class Jet:
    def __init__(self, f: float, df: npt.NDArray[np.float64]) -> None:
        self.f = f
        self.df = df

    def __int__(self) -> int:
        return int(self.f)

    def __float__(self) -> float:
        return self.f

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return Jet(-self.f, -self.df)

    def __abs__(self) -> Self:
        # FIXME: zero point ???
        return Jet(abs(self.f), self.df * np.sign(self.f))

    def __add__(self, other: Union[Self, float]) -> Self:
        u = self.f
        du = self.df
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
            return Jet(u + v, du + dv)
        else:
            v = float(other)
            dv = 0
            return Jet(u + v, du)

    def __radd__(self, other: Union[Self, float]) -> Self:
        return self + other

    def __iadd__(self, other: Union[Self, float]) -> Self:
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
        else:
            v = float(other)
            dv = 0

        self.f += v
        self.dv += dv
        return self

    def __sub__(self, other: Union[Self, float]) -> Self:
        u = self.f
        du = self.df
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
        else:
            v = float(other)
            dv = 0

        return Jet(u - v, du - dv)

    def __rsub__(self, other: Union[Self, float]) -> Self:
        return -self + other

    def __isub__(self, other: Union[Self, float]) -> Self:
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
        else:
            v = float(other)
            dv = 0

        self.f -= v
        self.df -= dv
        return self

    def __mul__(self, other: Union[Self, float]) -> Self:
        u = self.f
        du = self.df
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
            return Jet(u * v, du * v + dv * u)
        else:
            v = float(other)
            dv = 0
            return Jet(u * v, du * v)

    def __mul__(self, other: Union[Self, float]) -> Self:
        return self * other

    def __imul__(self, other: Union[Self, float]) -> Self:
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
        else:
            v = float(other)
            dv = 0

        self.df *= v
        self.df += dv * self.f
        self.f *= v
        return self

    def reciprocal(self) -> Self:
        return Jet(np.reciprocal(self.f), -self.df * math.pow(self.f, -2))

    def __truediv__(self, other: Union[Self, float]) -> Self:
        return self * other.reciprocal()

    def __rtruediv__(self, other: Union[Self, float]) -> Self:
        return self.reciprocal() * other

    def __pow__(self, other: Union[Self, float]) -> Self:
        u = self.f
        du = self.df
        if isinstance(other, Jet):
            v = other.f
            dv = other.df
        else:
            v = float(other)
            dv = 0
        f = u**v
        df = (du * v / u + dv * math.log(u)) * f
        return Jet(f, df)

    def __rpow__(self, other: Union[Self, float]) -> Self:
        if isinstance(other, Jet):
            u = other.f
            du = other.df
        else:
            u = float(other)
            du = 0
        v = self.f
        dv = self.df
        f = u**v
        df = (du * v / u + dv * math.log(u)) * f
        return Jet(f, df)

    def __eq__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f == v

    def __ne__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f != v

    def __gt__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f > v

    def __ge__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f >= v

    def __lt__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f < v

    def __le__(self, other: Union[Self, float]) -> bool:
        v = other.f if isinstance(other, Jet) else float(other)
        return self.f <= v

    # TODO: modulo

    def exp(self) -> Self:
        f = math.exp(self.f)
        return Jet(f, self.df * f)

    def exp2(self) -> Self:
        f = math.exp2(self.f)
        return Jet(f, self.df * f * _LN2)

    def log(self) -> Self:
        return Jet(math.log(self.f), self.df / self.f)

    def log2(self) -> Self:
        return Jet(math.log2(self.f), self.df / (self.f * _LN2))

    def log10(self) -> Self:
        return Jet(math.log10(self.f), self.df / (self.f * _LN10))

    def expm1(self) -> Self:
        return Jet(math.expm1(self.f), self.df * math.exp(self.f))

    def log1p(self) -> Self:
        return Jet(math.log1p(self.f), self.df / (self.f + 1))

    def sqrt(self) -> Self:
        f = math.sqrt(self.f)
        return Jet(f, self.df / f * 0.5)

    def square(self) -> Self:
        return Jet(self.f * self.f, self.df * self.f * 2)

    def cbrt(self) -> Self:
        f = math.cbrt(self.f)
        return Jet(f, self.df / f**2)

    # TODO: gcd lcm

    # trigs
    def sin(self) -> Self:
        return Jet(math.sin(self.f), self.df * math.cos(self.f))

    def cos(self) -> Self:
        return Jet(math.cos(self.f), -self.df * math.sin(self.f))

    def tan(self) -> Self:
        sec2 = math.cos(self.f) ** -2
        return Jet(math.tan(self.f), self.df * sec2)

    def arcsin(self) -> Self:
        return Jet(math.acos(self.f), self.df / math.sqrt(-self.f**2 + 1))

    def arccos(self) -> Self:
        return Jet(math.acos(self.f), -self.df / math.sqrt(-self.f**2 + 1))

    def arctan(self) -> Self:
        return Jet(math.atan(self.f), self.df / (self.f**2 + 1))

    # TODO: arctan2

    # TODO: hypot

    # TODO: hyperbolics

    # TODO: degree, radian, ...

    # TODO: fmax, fmin
