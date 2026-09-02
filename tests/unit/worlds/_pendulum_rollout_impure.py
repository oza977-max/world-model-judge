"""Standalone script: the TC-WD7-02 negative/phantom-gate fixture — a
scratch copy of the rollout with a deliberately unseeded numpy.random
call injected into the step, proving the byte-identity check can
actually fail (never edits the real wmj.worlds.pendulum module).
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
    state = state + np.random.normal(0.0, 1e-9, size=state.shape)  # injected impurity
    trajectory[step + 1] = state

sys.stdout.buffer.write(canonical_serialize({"trajectory": trajectory}))
