"""Offline preparation of all Armed variants. Run after skin asset updates, before serving."""
from pathlib import Path

import numpy as np

from omega_playground.policy.package import load
from omega_playground.viewer.equipment import prepare_equipment


def main():
    root = Path(__file__).resolve().parents[2]
    from omega_playground.viewer.native_grip_assets import prepare
    print('prepared SONIC grip', prepare(load(root/'artifacts/policies/sonic-v1.1-g1-cuda-batch')))
    grips = {'l': (np.zeros(3), np.array([1., 0., 0.])),
             'r': (np.zeros(3), np.array([0., 0., 1.]))}
    print('hand equipment', prepare_equipment(grips))


if __name__ == '__main__':
    main()
