# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 TorchRL ClipPPOLoss 在 Isaac Lab 环境中训练策略的入口脚本。

本脚本实现了基于 TorchRL 的 PPO (Proximal Policy Optimization) 算法训练流程。
TorchRL 采用组件化设计哲学，提供模块化的 Loss、Collector、Network 等组件，
由用户手动组装训练循环，而非提供统一的 Runner（与 RL-Games/SKRL 不同）。

Algorithm:
    **PPO (Proximal Policy Optimization)**
    
    - Loss Function: ``torchrl.objectives.ppo.ClipPPOLoss``
    - Advantage Estimation: GAE (Generalized Advantage Estimation)
    - Policy Constraint: Clipped importance sampling ratio
    
    数学公式:
        L_PPO = -E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
        
        其中:
        - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t): 重要性采样比率
        - Â_t: GAE 优势估计
        - ε: clip 阈值 (默认 0.2)

Features:
    - ✅ 对称 Actor-Critic: Actor 和 Critic 使用相同观测
    - ✅ 非对称 Actor-Critic: 通过 ``separate_losses=True`` 支持特权观测
    - ✅ 多环境并行训练
    - ✅ TensorBoard 日志记录
    - ✅ 模型检查点保存与加载

Asymmetric Actor-Critic:
    如需使用非对称 Actor-Critic（Critic 使用特权信息），需在环境配置中定义不同的观测组：
    
    .. code-block:: python
    
        @configclass
        class ObservationsCfg:
            @configclass
            class PolicyCfg(ObsGroup):  # Actor 观测
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                # 仅传感器可测量信息
            
            @configclass
            class CriticCfg(ObsGroup):  # Critic 观测
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                object_position = ObsTerm(func=mdp.object_position)  # 特权信息
            
            policy: PolicyCfg = PolicyCfg()
            critic: CriticCfg = CriticCfg()
    
    然后在环境注册时配置 ``obs_groups``，并在 Loss 创建时设置 ``separate_losses=True``。

Note:
    本脚本专用于 PPO 算法。如需使用其他算法（如 SAC、TD3、A2C），需要：
    
    1. 使用对应的 Loss 模块（``SACLoss``, ``TD3Loss``, ``A2CLoss`` 等）
    2. 调整数据收集器（off-policy 算法需要 Replay Buffer）
    3. 修改训练循环逻辑
    
    TorchRL 的组件化设计意味着不同算法需要不同的脚本实现。

Example:
    训练 Cartpole 环境::
    
        $ python train.py --task Isaac-Cartpole-v0 --num_envs 512 --total_frames 1000000
    
    使用检查点恢复训练::
    
        $ python train.py --task Isaac-Cartpole-v0 --checkpoint /path/to/checkpoint.pt

"""

# Launch Isaac Sim Simulator first.

import argparse
import contextlib
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="torchrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--total_frames", type=int, default=None, help="Total environment frames to collect for PPO.")
parser.add_argument("--log_interval", type=int, default=10, help="How often (in updates) to log scalars.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a TorchRL checkpoint to resume from.")
parser.add_argument(
    "--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors for manager envs."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

import omni
from tensordict import TensorDictBase
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.torchrl import is_torchrl_available, make_torchrl_env

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Relative imports must occur after package initialisation
from .common import build_actor, build_critic, flatten_size, prepare_optimizer, select_spec, split_key

# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    if not is_torchrl_available():
        raise ImportError(
            "TorchRL 未安装。请运行 `pip install isaaclab_rl[torchrl]` 或确保当前虚拟环境包含 torchrl。"
        )

    agent_cfg = agent_cfg or {}

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    seed = agent_cfg.get("seed", 0)
    if args_cli.seed is not None:
        seed = args_cli.seed
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    # determine torch device
    device = torch.device(agent_cfg.get("device", env_cfg.sim.device))

    # logging directories
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "torchrl", args_cli.task or "torchrl"))
    log_dir = os.path.join(log_root_path, run_info)
    os.makedirs(log_dir, exist_ok=True)

    # set environment log directory
    env_cfg.log_dir = log_dir

    # export IO descriptors if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        if args_cli.export_io_descriptors:
            omni.log.warn("只有 ManagerBasedRLEnv 支持导出 IO descriptors。")  # type: ignore[attr-defined]

    # dump configs for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # type: ignore[arg-type]

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    torchrl_env = make_torchrl_env(env, device=device)  # type: ignore[arg-type]
    torchrl_env.set_seed(seed)

    # 检测是否为非对称 Actor-Critic（有 policy/critic 观测组）
    obs_spec_keys = list(torchrl_env.observation_spec.keys())
    has_asymmetric_obs = "policy" in obs_spec_keys and "critic" in obs_spec_keys
    
    if has_asymmetric_obs:
        # 非对称观测：Actor 使用 policy 观测，Critic 使用 critic 观测
        policy_obs_key = split_key(agent_cfg.get("policy_obs_key", "policy"))
        critic_obs_key = split_key(agent_cfg.get("critic_obs_key", "critic"))
        obs_key = policy_obs_key  # Actor 默认使用 policy 观测
    else:
        # 对称观测：Actor 和 Critic 使用相同观测
        obs_key = split_key(agent_cfg.get("obs_key", "observation"))
        policy_obs_key = critic_obs_key = obs_key
    
    action_key = split_key(agent_cfg.get("action_key", "action"))

    policy_obs_spec = select_spec(torchrl_env.observation_spec, policy_obs_key)
    critic_obs_spec = select_spec(torchrl_env.observation_spec, critic_obs_key) if has_asymmetric_obs else policy_obs_spec
    action_spec = select_spec(torchrl_env.action_spec, action_key)
    policy_obs_dim = flatten_size(policy_obs_spec.shape)
    critic_obs_dim = flatten_size(critic_obs_spec.shape)

    policy_cfg = agent_cfg.get("policy_model", {})
    value_cfg = agent_cfg.get("value_model", {})
    policy_module = build_actor(policy_obs_key, action_key, action_spec, policy_obs_dim, policy_cfg, device)
    value_module = build_critic(critic_obs_key, critic_obs_dim, value_cfg, device)

    loss_cfg = agent_cfg.get("ppo", {})
    clip_epsilon = loss_cfg.get("clip_epsilon", 0.2)
    entropy_coef = loss_cfg.get("entropy_coef", 0.0)
    value_loss_coef = loss_cfg.get("value_loss_coef", 0.5)
    normalize_advantage = loss_cfg.get("normalize_advantage", True)

    loss_module = ClipPPOLoss(
        policy_module,
        value_module,
        clip_epsilon=clip_epsilon,
        critic_coeff=value_loss_coef,
        entropy_bonus=entropy_coef != 0.0,
        entropy_coeff=entropy_coef if entropy_coef != 0.0 else None,
        normalize_advantage=normalize_advantage,
    )
    loss_module.set_keys(action=action_key, value="state_value")
    loss_module.default_value_estimator = ValueEstimators.GAE
    loss_module.make_value_estimator(
        value_type=ValueEstimators.GAE,
        gamma=loss_cfg.get("gamma", 0.99),
        lmbda=loss_cfg.get("gae_lambda", 0.95),
    )
    loss_module._critic_coef = value_loss_coef
    loss_module.entropy_bonus = entropy_coef != 0.0
    loss_module._entropy_coef = entropy_coef
    loss_module.to(device)

    optimizer = prepare_optimizer(loss_module.parameters(), agent_cfg.get("optimizer", {}))

    collector_cfg = agent_cfg.get("collector", {})
    frames_per_batch = collector_cfg.get("frames_per_batch", 16384)
    total_frames_cfg = collector_cfg.get("total_frames", int(1e7))
    total_frames = args_cli.total_frames or total_frames_cfg
    init_random_frames = collector_cfg.get("init_random_frames", 0)
    max_frames_per_traj = collector_cfg.get("max_frames_per_traj")
    if max_frames_per_traj in (None, 0):
        max_frames_per_traj = torchrl_env.max_steps or None

    collector = SyncDataCollector(
        torchrl_env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        init_random_frames=init_random_frames,
        max_frames_per_traj=max_frames_per_traj,
    )
    collector.set_seed(seed)

    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))

    global_frames = 0
    update_idx = 0
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args_cli.checkpoint:
        _load_checkpoint(args_cli.checkpoint, policy_module, value_module, optimizer)

    with contextlib.ExitStack() as stack:
        stack.enter_context(set_exploration_type(ExplorationType.RANDOM))
        try:
            for data in collector:
                data = data.to(device)

                rollout = cast(TensorDictBase, data.flatten(0, 1))
                loss_module.value_estimator(rollout)

                minibatch_size = loss_cfg.get("mini_batch_size", 4096)
                num_epochs = loss_cfg.get("num_epochs", 4)

                metrics_accumulator = {}
                for epoch in range(num_epochs):
                    num_samples = rollout.batch_size[0]
                    for start in range(0, num_samples, minibatch_size):
                        length = min(minibatch_size, num_samples - start)
                        if length <= 0:
                            continue
                        batch = cast(TensorDictBase, rollout.narrow(0, start, length))
                        if batch.batch_size[0] == 0:
                            continue
                        optimizer.zero_grad()
                        losses = loss_module(batch)
                        total_loss = _aggregate_losses(losses)
                        total_loss.backward()
                        grad_clip = loss_cfg.get("grad_clip", 1.0)
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), grad_clip)
                        optimizer.step()
                        _accumulate_metrics(metrics_accumulator, losses)

                frames_in_batch = data.batch_size.numel()
                global_frames += frames_in_batch
                update_idx += 1

                reward_tensor = data["reward"]
                if reward_tensor.ndim > 2:
                    reward_tensor = reward_tensor.squeeze(-1)
                mean_reward = reward_tensor.sum(dim=0).mean().item()
                if update_idx % args_cli.log_interval == 0:
                    print(
                        f"[TorchRL] update={update_idx} frames={global_frames} mean_reward={mean_reward:.3f}"
                    )

                _write_metrics(writer, metrics_accumulator, global_frames)
                writer.add_scalar("train/mean_reward", mean_reward, global_frames)

                if update_idx % args_cli.log_interval == 0:
                    _save_checkpoint(
                        os.path.join(checkpoint_dir, "latest.pt"),
                        policy_module,
                        value_module,
                        optimizer,
                        global_frames,
                        agent_cfg,
                    )
        except KeyboardInterrupt:
            print("[INFO] Training interrupted by user. Saving checkpoint...")
            _save_checkpoint(
                os.path.join(checkpoint_dir, "interrupt.pt"),
                policy_module,
                value_module,
                optimizer,
                global_frames,
                agent_cfg,
            )
        finally:
            collector.shutdown()
            writer.close()
            torchrl_env.close()
            env.close()


def _aggregate_losses(loss_dict: dict) -> torch.Tensor:
    loss = torch.zeros(1, device=next(iter(loss_dict.values())).device)
    for key in ("loss_objective", "loss_entropy", "loss_critic"):
        if key in loss_dict:
            loss = loss + loss_dict[key]
    return loss


def _accumulate_metrics(container: dict, loss_dict: dict):
    for key, value in loss_dict.items():
        if key.startswith("loss") or key in {"kl_approx", "entropy"}:
            container.setdefault(key, []).append(value.detach().cpu())


def _write_metrics(writer: SummaryWriter, metrics: dict, step: int):
    for key, values in metrics.items():
        stacked = torch.stack(values)
        writer.add_scalar(f"loss/{key}", stacked.mean().item(), step)


def _load_checkpoint(path: str, policy, value, optimizer):
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["policy"])
    value.load_state_dict(checkpoint["value"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[INFO] Loaded checkpoint from {path}")


def _save_checkpoint(path: str, policy, value, optimizer, frames: int, agent_cfg: dict):
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": frames,
            "agent_cfg": agent_cfg,
        },
        path,
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()
