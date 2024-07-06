import numpy as np


class Func:
    @property
    def dims_in(self) -> tuple[int, ...]:
        pass

    @property
    def dim_out(self) -> int:
        pass

    @property
    def xs(self) -> tuple[np.ndarray, ...]:
        pass

    @xs.setter
    def xs(self, xs: tuple[np.ndarray]) -> None:
        pass

    def compute(self) -> None:
        pass

    @property
    def val(self) -> np.ndarray:
        pass

    @property
    def jac(self) -> tuple[np.ndarray, ...]:
        raise NotImplementedError()


class PyFunc(Func):
    def __init__(self, func, dims_in, dim_out) -> None:
        self._func = func
        self._dims_in = dims_in
        self._dim_out = dim_out
        self._xs = None
        self._ret = None

    @property
    def dims_in(self) -> tuple[int, ...]:
        return self._dims_in

    @property
    def dim_out(self) -> int:
        return self._dim_out

    @property
    def xs(self) -> tuple[np.ndarray]:
        return self._xs

    @xs.setter
    def xs(self, xs_in: tuple[np.ndarray]) -> None:
        self._xs = xs_in

    def compute(self) -> None:
        self._ret = self._func(self._xs)

    def __call__(self, *args) -> np.ndarray:
        self.xs = args
        self.compute()
        return self.val

    @property
    def val(self) -> np.ndarray:
        return self._ret


class MergeIn(Func):
    def __init__(self, func: Func) -> None:
        self._func = func
        self._dim_in = np.sum(self._func.dims_in)

        cs = np.cumsum([0] + list(self._func.dims_in))
        self._slices = [slice(st, ed) for (st, ed) in zip(cs[:-1], cs[1:])]

    @property
    def dims_in(self) -> tuple[int]:
        return (self._dim_in,)

    @property
    def dim_out(self) -> int:
        return self._func.dim_out

    @property
    def xs(self) -> tuple[np.ndarray]:
        return (self._x,)

    @xs.setter
    def xs(self, xs_in: tuple[np.ndarray]) -> None:
        (self._x,) = xs_in
        self._func.xs = [self._x[s] for s in self._slices]

    def compute(self) -> None:
        return self._func.compute()

    @property
    def val(self) -> np.ndarray:
        return self._func.val

    @property
    def jac(self) -> tuple[np.ndarray]:
        # FIXME: pre-allocate
        return (np.concatenate(self._func.jac),)


class SplitIn(Func):
    def __init__(self, func: Func, dims_in: tuple[int, ...]) -> None:
        if len(func.dims_in) != 0:
            raise ValueError("Single input function only")
        self._func = func
        self._dims_in = dims_in
        (dim_in,) = dims_in

        cs = np.cumsum([0] + list(self._dims_in))
        if cs[-1] != dim_in:
            raise ValueError("Incorrect dimensions to split")
        self._slices = [slice(st, ed) for (st, ed) in zip(cs[:-1], cs[1:])]

    @property
    def dims_in(self) -> tuple[int, ...]:
        return self._dims_in

    @property
    def dim_out(self) -> int:
        return self._func.dim_out

    @property
    def xs(self) -> tuple[np.ndarray, ...]:
        return self._xs

    @xs.setter
    def xs(self, xs_in: tuple[np.ndarray, ...]) -> None:
        self._xs = xs_in
        # FIXME: pre-allocate
        self._func.xs = np.concatenate(self._xs)

    def compute(self) -> None:
        return self._func.compute()

    @property
    def val(self) -> np.ndarray:
        return self._func.val

    @property
    def jac(self):
        (jac,) = self._func.jac
        return [jac[:, s] for s in self._slices]
