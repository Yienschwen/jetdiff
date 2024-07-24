from typing import NamedTuple, Self, Optional

from multiprocessing.pool import Pool

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
    def __init__(
        self, blocks: list[Info], xs: list[np.ndarray], pool: Optional[Pool]
    ) -> None:
        self._pool = pool

        for x in xs:
            if len(x.shape) != 1:
                raise ValueError("one-dimesional x's only")
        self._xs = np.array(xs, dtype=object)
        self._x_slices = _slices_from_dims([x.size for x in self._xs])
        self._dim_in = self._x_slices[-1].stop

        for block in blocks:
            if len(block.func.dims_in) != len(block.x_indices):
                raise ValueError("block input count mismatch with x_indices")
            xs_block = self._xs[list(block.x_indices)]
            for x, dim in zip(xs_block, block.func.dims_in):
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
        # FIXME: don't make a copy every time
        return (np.concat(self._xs, axis=0),)

    @xs.setter
    def xs(self, xs_in):
        (x_in,) = xs_in
        for x, s in zip(self._xs, self._x_slices):
            x[:] = x_in[s]

        for block in self._blocks:
            block.func.xs = self._xs[list(block.x_indices)]  # FIXME: needed?

    @staticmethod
    def _compute(block):
        block.func.compute()
        return block

    def compute(self) -> Self:
        if self._pool is not None:
            self._blocks = self._pool.map(self._compute, self._blocks)
        else:
            self._blocks = [self._compute(block) for block in self._blocks]

        # FIXME: use pre-allocated arrays
        self._val = np.concatenate([block.func.val for block in self._blocks])
        self._jac_flat = np.concatenate(
            [jac.reshape(-1) for block in self._blocks for jac in block.func.jac]
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
