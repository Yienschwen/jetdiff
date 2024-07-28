from typing import Self

import numpy as np

from .func import Func, SplitIn, MergeIn

from .jet import Jet


class Single(Func):
    def __init__(self, func: Func) -> None:
        if len(func.dims_in) != 1:
            raise ValueError("should be single input function")
        self._func = func
        (self._dim_in,) = func.dims_in
        self._dim_out = func.dim_out

        self._x_jet = np.array(
            [Jet.k(dim=self._dim_in, k=i) for i in range(self._dim_in)]
        )

        self._ret_jet = None
        self._ret = np.empty(self._dim_out)
        self._jac = np.empty((self._dim_out, self._dim_in))

    @property
    def dims_in(self) -> int:
        return (self._dim_in,)

    @property
    def dim_out(self) -> int:
        return self._dim_out

    @property
    def xs(self):
        return (self._x_in,)

    @xs.setter
    def xs(self, xs):
        (x_in,) = xs
        self._x_in = x_in
        for i in range(self._dim_in):
            self._x_jet[i].f = x_in[i]
        self._func.xs = (self._x_jet,)

    def compute(self) -> Self:

        self._func.compute()
        self._ret_jet = self._func.val

        for i in range(self._dim_out):
            self._ret[i] = self._ret_jet[i].f
            self._jac[i] = self._ret_jet[i].df

        return self

    @property
    def val(self):
        return self._ret

    @property
    def jac(self):
        return (self._jac,)


class Multi(SplitIn):
    def __init__(self, func: Func):
        super().__init__(Single(MergeIn(func)), func.dims_in)
