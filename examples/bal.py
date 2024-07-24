from typing import NamedTuple, Type

from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares

from jetdiff.func import PyFunc, LazySingle
from jetdiff.jetfunc import Multi as MultiJet
from jetdiff.block import Info, Block


def crossmat(vec):
    xx, yy, zz = vec
    return np.array([[0 * xx, -zz, yy], [zz, 0 * yy, -xx], [-yy, xx, 0 * zz]])


def rodrigues(rvec, eps=1e-13):
    angle = np.linalg.norm(rvec)
    if angle < eps:
        print("eps", angle)
        return np.eye(3) + crossmat(rvec)
    axis = rvec / angle
    return (
        np.eye(3) * np.cos(angle)
        + crossmat(axis) * np.sin(angle)
        + axis[:, None] @ axis[None, :] * (1 - np.cos(angle))
    )


def residual(camera, point, xy):
    rmat = rodrigues(camera[:3])
    tvec = camera[3:6]
    f, k1, k2 = camera[6:]

    p_cam = rmat @ point + tvec
    p = -p_cam / p_cam[2]

    n = np.linalg.norm(p)
    n2 = n * n
    n4 = n2 * n2
    r = 1.0 + k1 * n2 + k2 * n4

    p_ = p * f * r

    return p_[:2] - xy


class Obs(NamedTuple):
    cam_index: int
    point_index: int
    x: float
    y: float


def _load_dataset(path_txt):
    with open(path_txt, encoding="ascii") as f:
        ss = f.read().split()
        ss.reverse()

    def _load(strs: list[str], *types: Type) -> list:
        return [t(strs.pop(-1)) for t in types]

    num_cams, num_points, num_obs = _load(ss, int, int, int)

    obss: list[Obs] = []
    for _ in range(num_obs):
        obss.append(Obs(*_load(ss, int, int, float, float)))

    cams = list()
    f9 = [float] * 9
    for _ in range(num_cams):
        cams.append(np.array(_load(ss, *f9)))

    pts = list()
    for _ in range(num_points):
        pts.append(np.array(_load(ss, float, float, float)))

    assert len(ss) == 0, "dataset file is not empty yet"

    return obss, cams, pts


def _main():
    from sys import argv

    _, txt_in, txt_out = argv

    obss, cams, pts = _load_dataset(txt_in)
    xs = cams + pts
    n_cam = len(cams)
    infos = list()
    for obs in obss:
        info = Info(
            func=MultiJet(PyFunc(partial(residual, xy=(obs.x, obs.y)), (9, 3), 2)),
            x_indices=(obs.cam_index, n_cam + obs.point_index),
        )
        infos.append(info)

    with Pool(4) as pool:
        blocks = Block(infos, xs, pool)
        (x0,) = blocks.xs

        func = LazySingle(blocks)

        sol = least_squares(func, x0, jac=func.jac, x_scale="jac", verbose=2)


if __name__ == "__main__":
    _main()
