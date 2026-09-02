"""Standalone script: roll out a pendulum trajectory and print canonical
bytes to stdout. Run as a fresh subprocess by test_wd7.py (TC-WD7-01)
so the same rollout can be compared across genuinely separate processes.
"""

import sys

sys.path.insert(0, sys.argv[1])

import numpy as np

from wmj.harness.serialize import canonical_serialize
from wmj.worlds import pendulum

state = np.array([0.1, 0.1, 0.0, 0.0])
action = np.array([0.3])
n_steps = 200

trajectory = np.zeros((n_steps + 1, 4))
trajectory[0] = state
for step in range(n_steps):
    state = pendulum.transition(state, action)
    trajectory[step + 1] = state

sys.stdout.buffer.write(canonical_serialize({"trajectory": trajectory}))
