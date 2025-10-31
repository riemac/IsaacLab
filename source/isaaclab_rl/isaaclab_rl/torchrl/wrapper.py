# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to integrate Isaac Lab environments with TorchRL."""

from __future__ import annotations

from typing import Any

import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

try:
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper as _TorchRLIsaacLabWrapper
except ImportError:  # pragma: no cover - handled gracefully at runtime
    _TorchRLIsaacLabWrapper = None

__all__ = ["TorchRLWrapper", "make_torchrl_env", "is_torchrl_available"]


def is_torchrl_available() -> bool:
    """Returns ``True`` when TorchRL is importable."""

    return _TorchRLIsaacLabWrapper is not None


def make_torchrl_env(
    env: ManagerBasedRLEnv | DirectRLEnv,
    *,
    device: str | torch.device | None = None,
    allow_done_after_reset: bool = True,
    categorical_action_encoding: bool = False,
    convert_actions_to_numpy: bool = False,
    **wrapper_kwargs: Any,
):
    """Wrap an Isaac Lab environment so it can be consumed by TorchRL collectors.

    Args:
        env: The environment instance to wrap. Must derive from :class:`ManagerBasedRLEnv`
            or :class:`DirectRLEnv`.
        device: Target device for TorchRL tensors. Defaults to the simulation device
            if ``None``.
        allow_done_after_reset: Forwarded to :class:`torchrl.envs.libs.isaac_lab.IsaacLabWrapper`.
        categorical_action_encoding: Forwarded to the underlying wrapper.
        convert_actions_to_numpy: Forwarded to the underlying wrapper.
        wrapper_kwargs: Additional keyword arguments passed to
            :class:`torchrl.envs.libs.isaac_lab.IsaacLabWrapper`.

    Returns:
        The wrapped environment compatible with TorchRL APIs.

    Raises:
        ImportError: If TorchRL is not installed.
        ValueError: If ``env`` is not a supported Isaac Lab RL environment.
    """

    if _TorchRLIsaacLabWrapper is None:  # pragma: no cover - exercised only when TorchRL is missing
        raise ImportError(
            "TorchRL is not installed. Please install it via 'pip install isaaclab_rl[torchrl]' "
            "or add 'torchrl>=0.10.0' to your environment before using the TorchRL integration."
        )

    base_env = env.unwrapped
    if not isinstance(base_env, (ManagerBasedRLEnv, DirectRLEnv)):
        raise ValueError(
            "The environment must inherit from ManagerBasedRLEnv or DirectRLEnv. "
            f"Received: {type(base_env)!r}."
        )

    resolved_device = _resolve_device(device, getattr(base_env, "device", None))

    return _TorchRLIsaacLabWrapper(
        env,
        allow_done_after_reset=allow_done_after_reset,
        categorical_action_encoding=categorical_action_encoding,
        convert_actions_to_numpy=convert_actions_to_numpy,
        device=resolved_device,
        **wrapper_kwargs,
    )


def _resolve_device(device: str | torch.device | None, fallback: str | torch.device | None) -> torch.device:
    """Helper to resolve the torch device for the wrapper."""

    if device is None:
        device = fallback if fallback is not None else "cuda:0"
    return torch.device(device)


# Re-export the underlying TorchRL wrapper when available for direct access.
TorchRLWrapper = _TorchRLIsaacLabWrapper
