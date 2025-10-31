# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-tests for the TorchRL Isaac Lab wrapper.
┌─────────────────────────────────────────────────────┐
│ 1. 启动 Isaac Sim (headless 模式)                  │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 2. 查找所有注册了 torchrl_cfg_entry_point 的任务   │
│    └── 从 gym.registry 中筛选                      │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 3. 对每个任务执行测试 (test_random_actions)        │
│    a. 创建 IsaacLab 环境                           │
│    b. 使用 make_torchrl_env() 包装                 │
│    c. 重置环境 → 检查观测有效性                   │
│    d. 执行 5 步随机动作 → 检查奖励有效性          │
│    e. 关闭环境                                     │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│ 4. 清理并关闭 Isaac Sim                            │
└─────────────────────────────────────────────────────┘
"""

from isaaclab.app import AppLauncher

# launch simulator headless once for the whole module
torchrl_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = torchrl_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import carb
import omni.usd
import pytest
from tensordict import TensorDictBase

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.torchrl import make_torchrl_env

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pytest.importorskip("torchrl")


@pytest.fixture(scope="module")
def registered_tasks():
    tasks: list[str] = []
    for task_spec in gym.registry.values():
        if "Isaac" in task_spec.id:
            cfg_entry_point = gym.spec(task_spec.id).kwargs.get("torchrl_cfg_entry_point")
            if cfg_entry_point is not None:
                tasks.append(task_spec.id)
    tasks.sort()

    carb.settings.get_settings().set_bool("/physics/cooking/ujitsoCollisionCooking", False)
    return tasks[:3]


def _check_values(buffer) -> bool:
    if isinstance(buffer, torch.Tensor):
        return bool(torch.isfinite(buffer).all().item())
    if isinstance(buffer, dict):
        return all(_check_values(v) for v in buffer.values())
    return True


def test_random_actions(registered_tasks):
    num_envs = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for task_name in registered_tasks:
        omni.usd.get_context().new_stage()
        carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        env = gym.make(task_name, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)  # type: ignore[arg-type]

        torchrl_env = make_torchrl_env(env, device=torch.device(device))  # type: ignore[arg-type]

        td = torchrl_env.reset()
        assert isinstance(td, TensorDictBase)
        observation = td.get("observation")
        if isinstance(observation, dict):
            assert all(_check_values(val) for val in observation.values())
        else:
            assert _check_values(observation)

        with torch.inference_mode():
            for _ in range(5):
                action_td = torchrl_env.rand_action()
                td = torchrl_env.step(action_td)
                assert _check_values(td.get("reward"))
                td = td.get("next")

        torchrl_env.close()
        env.close()


@pytest.fixture(scope="session", autouse=True)
def shutdown_app():
    yield
    simulation_app.close()
