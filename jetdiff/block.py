from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from .func import Func


def _slices_from_dims(dims):
    cs = np.cumsum([0] + list(dims))
    starts = cs[:-1]
    ends = cs[1:]
    return [slice(st, ed) for st, ed in zip(starts, ends)]


class Info(NamedTuple):
    func: Func
    x_indices: tuple[int, ...]


class Block(Func):
    def __init__(self, blocks: list[Info], xs: list[np.ndarray]) -> None:
        for x in xs:
            if len(x.shape) != 1:
                raise ValueError("one-dimesional x's only")
        self._xs = np.array(xs, dtype=object)
        self._x_slices = _slices_from_dims([x.size for x in self._xs])
        self._dim_in = self._x_slices[-1].stop

        for block in blocks:
            if len(block.func.dims_in) != len(block.x_indices):
                raise ValueError("block input count mismatch with x_indices")
            xs_block = self._xs[block.x_indices]
            for x, dim in zip(xs_block):
                if x.shape != (dim,):
                    raise ValueError("incorrect x shape")
        self._blocks = blocks
        self._f_slices = _slices_from_dims(
            [block.func.dim_out for block in self._blocks]
        )
        self._dim_out = self._f_slices[-1].stop

        rows, cols = list(), list()
        for block, sf in zip(self._blocks, self._f_slices):
            range_rows = range(sf.start, sf.stop)
            for ix in block.x_indices:
                sx = self._x_slices[ix]
                range_cols = range(sx.start, sx.stop)

                rrows, ccols = np.meshgrid(range_rows, range_cols, indexing="ij")
                rows.append(rrows.reshape(-1))
                cols.append(ccols.reshape(-1))
        self._jac_rows = np.concatenate(rows)
        self._jac_cols = np.concatenate(cols)

    @property
    def xs(self):
        return (self._xs,)

    @xs.setter
    def xs(self, xs_in):
        (x_in,) = xs_in
        for x, s in zip(self._xs, self._x_slices):
            x[:] = x_in[s]

        for block in self._blocks:
            block.func.xs = self._xs[block.x_indices]  # FIXME: needed?

    def compute(self) -> None:
        for block in self._blocks:
            block.func.compute()

        # FIXME: use pre-allocated arrays
        self._val = np.concatenate([block.func.val for block in self._blocks])
        self._jac_flat = np.concatenate(
            [block.func.jac.reshape(-1) for block in self._blocks]
        )

        self._jac = sp.csr_array(
            (self._jac_flat, (self._jac_rows, self._jac_cols)),
            shape=(self._dim_out, self._dim_in),
        )

    @property
    def val(self):
        return self._val

    @property
    def jac(self):
        return self._jac
