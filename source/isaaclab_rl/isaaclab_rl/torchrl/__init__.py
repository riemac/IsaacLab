# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an environment for TorchRL library.

TorchRL is a modular, composable reinforcement learning library from PyTorch that provides
building blocks for RL algorithms rather than unified high-level runners. This integration
provides environment wrappers compatible with TorchRL's component-based design philosophy.

The following example shows how to wrap an environment for TorchRL:

.. code-block:: python

    from isaaclab_rl.torchrl import make_torchrl_env

    # Wrap the environment
    torchrl_env = make_torchrl_env(env, device="cuda:0")

    # Use with TorchRL collectors
    from torchrl.collectors import SyncDataCollector
    collector = SyncDataCollector(
        torchrl_env,
        policy,
        frames_per_batch=1000,
        total_frames=1_000_000,
    )

Note:
    TorchRL follows a component-based architecture. Unlike RL-Games or SKRL which provide
    unified runners, TorchRL offers modular loss functions, collectors, and networks that
    users compose manually. This design provides maximum flexibility at the cost of requiring
    more manual setup.

    For algorithm-specific training scripts, see:
    - PPO: ``scripts/reinforcement_learning/torchrl/train.py`` (ClipPPOLoss)
    - Future algorithms will require separate implementations with their respective loss modules.

"""

from .wrapper import TorchRLWrapper, is_torchrl_available, make_torchrl_env

__all__ = ["TorchRLWrapper", "make_torchrl_env", "is_torchrl_available"]
