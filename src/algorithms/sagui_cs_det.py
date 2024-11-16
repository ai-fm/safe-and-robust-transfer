# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm."""


from __future__ import annotations

import time
from typing import Any
import numpy as np

import torch
from torch import nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from .load_guide import load_guide

from omnisafe.algorithms import registry
from omnisafe.common.lagrange import Lagrange
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.logger import Logger
from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.utils.config import Config

from safety_gymnasium.bases.base_task import BaseTask

from abc import ABC, abstractmethod

from gymnasium.spaces import Box

from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


class BaseBuffer(ABC):
    r"""Abstract base class for buffer.

    .. warning::
        The buffer only supports Box spaces.

    In base buffer, we store the following data:

    +--------+---------------------------+---------------+-----------------------------------+
    | Name   | Shape                     | Dtype         | Description                       |
    +========+===========================+===============+===================================+
    | obs    | (size, \*obs_space.shape) | torch.float32 | The observation from environment. |
    +--------+---------------------------+---------------+-----------------------------------+
    | act    | (size, \*act_space.shape) | torch.float32 | The action from agent.            |
    +--------+---------------------------+---------------+-----------------------------------+
    | reward | (size,)                   | torch.float32 | Single step reward.               |
    +--------+---------------------------+---------------+-----------------------------------+
    | cost   | (size,)                   | torch.float32 | Single step cost.                 |
    +--------+---------------------------+---------------+-----------------------------------+
    | done   | (size,)                   | torch.float32 | Whether the episode is done.      |
    +--------+---------------------------+---------------+-----------------------------------+


    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        device (torch.device): The device of the buffer. Defaults to ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`BaseBuffer`."""
        self._device: torch.device = device
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros((size, *obs_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError
        if isinstance(act_space, Box):
            act_buf = torch.zeros((size, *act_space.shape), dtype=torch.float32, device=device)
        else:
            raise NotImplementedError

        self.data: dict[str, torch.Tensor] = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros(size, dtype=torch.float32, device=device),
            'logp_guide': torch.zeros(size, dtype=torch.float32, device=device),
            'importance': torch.zeros(size, dtype=torch.float32, device=device),
            'cost': torch.zeros(size, dtype=torch.float32, device=device),
            'done': torch.zeros(size, dtype=torch.float32, device=device),
        }
        self._size: int = size

    @property
    def device(self) -> torch.device:
        """The device of the buffer."""
        return self._device

    @property
    def size(self) -> int:
        """The size of the buffer."""
        return self._size

    def __len__(self) -> int:
        """Return the length of the buffer."""
        return self._size

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros((self._size, *shape), dtype=dtype, device=self._device)

    @abstractmethod
    def store(self, **data: torch.Tensor) -> None:
        """Store a transition in the buffer.

        .. warning::
            This is an abstract method.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.store(obs=obs, act=act, reward=reward, cost=cost, done=done)

        Args:
            data (torch.Tensor): The data to store.
        """


class OffPolicyBuffer(BaseBuffer):
    r"""A ReplayBuffer for off_policy Algorithms.

    .. warning::
        The buffer only supports Box spaces.

    Compared to the base buffer, the off-policy buffer stores extra data:

    +----------+---------------------------+---------------+----------------------------------------+
    | Name     | Shape                     | Dtype         | Description                            |
    +==========+===========================+===============+========================================+
    | next_obs | (size, \*obs_space.shape) | torch.float32 | The next observation from environment. |
    +----------+---------------------------+---------------+----------------------------------------+

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data stored in the buffer.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`OffPolicyBuffer`."""
        super().__init__(obs_space, act_space, size, device)
        if isinstance(obs_space, Box):
            self.data['next_obs'] = torch.zeros(
                (size, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size

        assert (
            self._max_size > self._batch_size
        ), 'The size of the buffer must be larger than the batch size.'

    @property
    def max_size(self) -> int:
        """Return the max size of the buffer."""
        return self._max_size

    @property
    def size(self) -> int:
        """Return the current size of the buffer."""
        return self._size

    @property
    def batch_size(self) -> int:
        """Return the batch size of the buffer."""
        return self._batch_size

    def store(self, **data: torch.Tensor) -> None:
        """Store data into the buffer.

        .. hint::
            The ReplayBuffer is a circular buffer. When the buffer is full, the oldest data will be
            overwritten.

        Args:
            data (torch.Tensor): The data to be stored.
        """
        for key, value in data.items():
            self.data[key][self._ptr] = value
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        idxs = torch.randint(0, self._size, (self._batch_size,))
        return {key: value[idxs] for key, value in self.data.items()}


class VectorOffPolicyBuffer(OffPolicyBuffer):
    """Vectorized on-policy buffer.

    The vector-off-policy buffer is a vectorized version of the off-policy buffer. It stores the
    data in a single tensor, and the data of each environment is stored in a separate column.

    .. warning::
        The buffer only supports Box spaces.

    Args:
        obs_space (OmnisafeSpace): The observation space.
        act_space (OmnisafeSpace): The action space.
        size (int): The size of the buffer.
        batch_size (int): The batch size of the buffer.
        num_envs (int): The number of environments.
        device (torch.device, optional): The device of the buffer. Defaults to
            ``torch.device('cpu')``.

    Attributes:
        data (dict[str, torch.Tensor]): The data of the buffer.

    Raises:
        NotImplementedError: If the observation space or the action space is not Box.
        NotImplementedError: If the action space or the action space is not Box.
    """

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        size: int,
        batch_size: int,
        num_envs: int,
        device: torch.device = DEVICE_CPU,
    ) -> None:
        """Initialize an instance of :class:`VectorOffPolicyBuffer`."""
        self._num_envs: int = num_envs
        self._ptr: int = 0
        self._size: int = 0
        self._max_size: int = size
        self._batch_size: int = batch_size
        self._device: torch.device = device
        if isinstance(obs_space, Box):
            obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
            next_obs_buf = torch.zeros(
                (size, num_envs, *obs_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        if isinstance(act_space, Box):
            act_buf = torch.zeros(
                (size, num_envs, *act_space.shape),
                dtype=torch.float32,
                device=device,
            )
        else:
            raise NotImplementedError

        self.data = {
            'obs': obs_buf,
            'act': act_buf,
            'reward': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'logp_guide': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'importance': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'cost': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'done': torch.zeros((size, num_envs), dtype=torch.float32, device=device),
            'next_obs': next_obs_buf,
        }

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._num_envs

    def add_field(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Add a field to the buffer.

        Examples:
            >>> buffer = BaseBuffer(...)
            >>> buffer.add_field('new_field', (2, 3), torch.float32)
            >>> buffer.data['new_field'].shape
            >>> (buffer.size, 2, 3)

        Args:
            name (str): The name of the field.
            shape (tuple of int): The shape of the field.
            dtype (torch.dtype): The dtype of the field.
        """
        self.data[name] = torch.zeros(
            (self._max_size, self._num_envs, *shape),
            dtype=dtype,
            device=self._device,
        )

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample a batch of data from the buffer.

        Returns:
            The sampled batch of data.
        """
        idx = torch.randint(
            0,
            self._size,
            (self._batch_size * self._num_envs,),
            device=self._device,
        )
        env_idx = torch.arange(self._num_envs, device=self._device).repeat(self._batch_size)
        return {key: value[idx, env_idx] for key, value in self.data.items()}


class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::
        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _current_obs: torch.Tensor
    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OffPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._current_obs, _ = self.reset()
        self._max_ep_len: int = 1000
        self._reset_log()

        self.task: BaseTask = self._env.get_base_env().task

        self.env_guide, self.guide = load_guide(
            self._cfgs.transfer_cfgs.guide_save_dir,
            self._cfgs.transfer_cfgs.guide_model_name)
        self.recover = False  # Ugly but OK

        self.max_act_diff = np.linalg.norm(self.action_space.high - self.action_space.low)

    def eval_policy(  # pylint: disable=too-many-locals
        self,
        episode: int,
        agent: ConstraintActorQCritic,
        logger: Logger,
    ) -> None:
        """Rollout the environment with deterministic agent action.

        Args:
            episode (int): Number of episodes.
            agent (ConstraintActorCritic): Agent.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        for _ in range(episode):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
            obs, _ = self._eval_env.reset()
            obs = obs.to(self._device)

            done = False
            while not done:
                act = agent.step(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = self._eval_env.step(act)
                obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=self._device)
                    for x in (obs, reward, cost, terminated, truncated)
                )
                ep_ret += info.get('original_reward', reward).cpu()
                ep_cost += info.get('original_cost', cost).cpu()
                ep_len += 1
                done = bool(terminated[0].item()) or bool(truncated[0].item())

            logger.store(
                {
                    'Metrics/TestEpRet': ep_ret,
                    'Metrics/TestEpCost': ep_cost,
                    'Metrics/TestEpLen': ep_len,
                },
            )

    def _obs_student_to_guide(self, current_obs: torch.Tensor):
        student_obs_flat = current_obs.numpy().flatten()

        base_guide_env = self.env_guide.get_base_env()
        base_student_env = self._env.get_base_env()

        guide_size = self.env_guide.observation_space.shape[0]

        student_obs_dict = base_student_env.obs_space_dict
        guide_obs_dict = base_guide_env.obs_space_dict

        guide_keys = [k for k in guide_obs_dict.keys()]

        student_offset = 0
        teacher_offset = 0

        guide_obs_flat = np.zeros(guide_size, dtype=np.float32)
        for k in student_obs_dict.keys():
            obs = student_obs_dict[k]
            k_size = np.prod(obs.shape)
            if k in guide_keys:
                vals = student_obs_flat[student_offset:student_offset + k_size]
                guide_obs_flat[teacher_offset:teacher_offset + k_size] = vals
                teacher_offset += k_size
            student_offset += k_size
        return torch.tensor(guide_obs_flat, dtype=torch.float32)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Do not recover in the first step
        self.recover = False
        return super().reset()

    def rollout(  # pylint: disable=too-many-locals
        self,
        rollout_step: int,
        agent: ConstraintActorQCritic,
        buffer_guide: VectorOffPolicyBuffer,
        buffer_student: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            rollout_step (int): Number of rollout steps.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor, reward critic,
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            use_rand_action (bool): Whether to use random action.
        """

        for _ in range(rollout_step):
            # Convert student's obs to guide's obs
            obs = self._current_obs
            obs_guide = self._obs_student_to_guide(obs)

            # Compute difference between student's mean and guide's action
            with torch.no_grad():
                guide_act = self.guide.predict(obs_guide, deterministic=False)
                mean_student = agent.actor.predict(obs, deterministic=True)
                difference_normalized = torch.norm(mean_student - guide_act) / self.max_act_diff
                # If difference = 0 then similarity = 1
                # If difference = 1 then similarity = 0
                similarity = 1.0 - difference_normalized

            if self.recover or use_rand_action:
                # Use guide to recover
                act = guide_act

                # Importance sampling ratio (clamped between 0.1 and 2.0)
                importance = np.maximum(np.minimum(similarity,
                                        self._cfgs.transfer_cfgs.importance_upper), self._cfgs.transfer_cfgs.importance_lower)
            else:
                # Use student's action
                act = agent.step(obs, deterministic=False)

                # Importance sampling ratio is 1 because we sampled from the student
                importance = 1.0

            # Step the env
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            # Recover if cost is > 0
            self.recover = cost.item() > 0

            self._log_value(reward=reward, cost=cost, info=info)
            real_next_obs = next_obs.clone()
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done:
                    if 'final_observation' in info:
                        real_next_obs[idx] = info['final_observation'][idx]
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

            if abs(importance - 1.0) < 0.01:
                buffer_student.store(
                    obs=self._current_obs,
                    act=act,
                    reward=reward,
                    logp_guide=similarity,
                    importance=importance,
                    cost=cost,
                    done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                    next_obs=real_next_obs,
                )
            else:
                buffer_guide.store(
                    obs=self._current_obs,
                    act=act,
                    reward=reward,
                    logp_guide=similarity,
                    importance=importance,
                    cost=cost,
                    done=torch.logical_and(terminated, torch.logical_xor(terminated, truncated)),
                    next_obs=real_next_obs,
                )

            self._current_obs = next_obs

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0


class DDPG(BaseAlgo):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:
        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
            Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    _epoch: int

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.OffPolicyAdapter` to adapt the environment to this
        algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()

        Raises:
            AssertionError: If the number of steps per epoch is not divisible by the number of
                environments.
            AssertionError: If the total number of steps is not divisible by the number of steps per
                epoch.
        """
        self._env: OffPolicyAdapter = OffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        """Initialize the model.

        OmniSafe uses :class:`omnisafe.models.actor_critic.constraint_actor_q_critic.ConstraintActorQCritic`
        as the default model.

        User can customize the model by inheriting this method.

        Examples:
            >>> def _init_model(self) -> None:
            ...     self._actor_critic = CustomActorQCritic()
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 1
        self._actor_critic: ConstraintActorQCritic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()
        """
        self._buf_guide: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=int(self._cfgs.algo_cfgs.size * 0.5),
            batch_size=int(self._cfgs.algo_cfgs.batch_size * (1 - self._cfgs.transfer_cfgs.student_batch_portion)),
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )
        self._buf_student: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=int(self._cfgs.algo_cfgs.size * 0.5),
            batch_size=int(self._cfgs.algo_cfgs.batch_size * self._cfgs.transfer_cfgs.student_batch_portion),
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        """Log info about epoch.

        +-------------------------+----------------------------------------------------------------------+
        | Things to log           | Description                                                          |
        +=========================+======================================================================+
        | Train/Epoch             | Current epoch.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpCost          | Average cost of the epoch.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpRet           | Average return of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/EpLen           | Average length of the epoch.                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpCost      | Average cost of the evaluate epoch.                                  |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpRet       | Average return of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Metrics/TestEpLen       | Average length of the evaluate epoch.                                |
        +-------------------------+----------------------------------------------------------------------+
        | Value/reward_critic     | Average value in :meth:`rollout` (from critic network) of the epoch. |
        +-------------------------+----------------------------------------------------------------------+
        | Values/cost_critic      | Average cost in :meth:`rollout` (from critic network) of the epoch.  |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_pi            | Loss of the policy network.                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                                           |
        +-------------------------+----------------------------------------------------------------------+
        | Loss/Loss_cost_critic   | Loss of the cost critic network.                                     |
        +-------------------------+----------------------------------------------------------------------+
        | Train/LR                | Learning rate of the policy network.                                 |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/Seed               | Seed of the experiment.                                              |
        +-------------------------+----------------------------------------------------------------------+
        | Misc/TotalEnvSteps      | Total steps of the experiment.                                       |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Total              | Total time.                                                          |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Rollout            | Rollout time.                                                        |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Update             | Update time.                                                         |
        +-------------------------+----------------------------------------------------------------------+
        | Time/Evaluate           | Evaluate time.                                                       |
        +-------------------------+----------------------------------------------------------------------+
        | FPS                     | Frames per second of the epoch.                                      |
        +-------------------------+----------------------------------------------------------------------+
        """
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)

        if self._cfgs.train_cfgs.eval_episodes > 0:
            self._logger.register_key('Metrics/TestEpRet', window_length=50)
            self._logger.register_key('Metrics/TestEpCost', window_length=50)
            self._logger.register_key('Metrics/TestEpLen', window_length=50)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Evaluate')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def learn(self) -> tuple[float, float, float]:
        """This is main function for algorithm update.

        It is divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.

        Returns:
            ep_ret: average episode return in final epoch.
            ep_cost: average episode cost in final epoch.
            ep_len: average episode length in final epoch.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            self._epoch = epoch
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer_guide=self._buf_guide,
                    buffer_student=self._buf_student,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            eval_start = time.time()
            self._env.eval_policy(
                episode=self._cfgs.train_cfgs.eval_episodes,
                agent=self._actor_critic,
                logger=self._logger,
            )
            eval_time = time.time() - eval_start

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})
            self._logger.store({'Time/Evaluate': eval_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        """Update actor, critic.

        -  Get the ``data`` from buffer

        .. note::

            +----------+---------------------------------------+
            | obs      | ``observaion`` stored in buffer.      |
            +==========+=======================================+
            | act      | ``action`` stored in buffer.          |
            +----------+---------------------------------------+
            | reward   | ``reward`` stored in buffer.          |
            +----------+---------------------------------------+
            | cost     | ``cost`` stored in buffer.            |
            +----------+---------------------------------------+
            | next_obs | ``next observaion`` stored in buffer. |
            +----------+---------------------------------------+
            | done     | ``terminated`` stored in buffer.      |
            +----------+---------------------------------------+

        -  Update value net by :meth:`_update_reward_critic`.
        -  Update cost net by :meth:`_update_cost_critic`.
        -  Update policy net by :meth:`_update_actor`.

        The basic process of each update is as follows:

        #. Get the mini-batch data from buffer.
        #. Get the loss of network.
        #. Update the network by loss.
        #. Repeat steps 2, 3 until the ``update_iters`` times.
        """
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            batch_student = self._buf_student.sample_batch()
            batch_guide = self._buf_guide.sample_batch()
            data = batch_student | batch_guide  # Merge the two batches
            self._update_count += 1
            obs, act, reward, logp_guide, importance, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['logp_guide'],
                data['importance'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            lagrange: Lagrange = self._lagrange  # Inheritance is stupid
            r = reward + logp_guide * lagrange.lagrangian_multiplier  # Add bonus for this
            self._update_reward_critic(obs, act, r, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Get the TD loss of reward critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_r = self._actor_critic.target_reward_critic(next_obs, next_action)[0]
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r = self._actor_critic.reward_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q_value_r.mean().item(),
            },
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)[0]
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor.

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in DDPG is defined as:

        .. math::

            L = -Q^V (s, \pi (s))

        where :math:`Q^V` is the reward critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=True)
        return -self._actor_critic.reward_critic(obs, action)[0].mean()

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )


class SAC(DDPG):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    _log_alpha: torch.Tensor
    _alpha_optimizer: optim.Optimizer
    _target_entropy: float

    def _init_model(self) -> None:
        """Initialize the model.

        The ``num_critics`` in ``critic`` configuration must be 2.
        """
        self._cfgs.model_cfgs.critic['num_critics'] = 2
        self._actor_critic = ConstraintActorQCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        """The initialization of the algorithm.

        User can define the initialization of the algorithm by inheriting this method.

        Examples:
            >>> def _init(self) -> None:
            ...     super()._init()
            ...     self._buffer = CustomBuffer()
            ...     self._model = CustomModel()

        In SAC, we need to initialize the ``log_alpha`` and ``alpha_optimizer``.
        """
        super()._init()
        if self._cfgs.algo_cfgs.auto_alpha:
            self._target_entropy = -torch.prod(torch.Tensor(self._env.action_space.shape)).item()
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)

            assert self._cfgs.model_cfgs.critic.lr is not None
            self._alpha_optimizer = optim.Adam(
                [self._log_alpha],
                lr=self._cfgs.model_cfgs.critic.lr,
            )
        else:
            self._log_alpha = torch.log(
                torch.tensor(self._cfgs.algo_cfgs.alpha, device=self._device),
            )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Value/alpha')
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.register_key('Loss/alpha_loss')

    @property
    def _alpha(self) -> float:
        """The value of alpha."""
        return self._log_alpha.exp().item()

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Sample the target action by target actor.
        - Get the target Q value by target critic.
        - Use the minimum target Q value to update reward critic.
        - Add the entropy loss to reward critic.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=False)
            next_logp = self._actor_critic.actor.log_prob(next_action)
            next_q1_value_r, next_q2_value_r = self._actor_critic.target_reward_critic(
                next_obs,
                next_action,
            )
            next_q_value_r = torch.min(next_q1_value_r, next_q2_value_r) - next_logp * self._alpha
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r

        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_r, target_q_value_r) + nn.functional.mse_loss(
            q2_value_r,
            target_q_value_r,
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q1_value_r.mean().item(),
            },
        )

    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        """Update actor and alpha if ``auto_alpha`` is True.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
        """
        super()._update_actor(obs)

        if self._cfgs.algo_cfgs.auto_alpha:
            with torch.no_grad():
                action = self._actor_critic.actor.predict(obs, deterministic=False)
                log_prob = self._actor_critic.actor.log_prob(action)
            alpha_loss = -self._log_alpha * (log_prob + self._target_entropy).mean()

            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()
            self._logger.store(
                {
                    'Loss/alpha_loss': alpha_loss.mean().item(),
                },
            )
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SAC is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \alpha \log \pi (s)

        where :math:`Q^V` is the min value of two reward critic networks, and :math:`\pi` is the
        policy network, and :math:`\alpha` is the temperature parameter.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        q1_value_r, q2_value_r = self._actor_critic.reward_critic(obs, action)
        return (self._alpha * log_prob - torch.min(q1_value_r, q2_value_r)).mean()

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        super()._log_when_not_update()
        self._logger.store(
            {
                'Value/alpha': self._alpha,
            },
        )
        if self._cfgs.algo_cfgs.auto_alpha:
            self._logger.store(
                {
                    'Loss/alpha_loss': 0.0,
                },
            )


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SaGuiCSDet(SAC):
    """The Lagrangian version of Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the SACLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.
        """
        super()._update()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if self._epoch > self._cfgs.algo_cfgs.warmup_epochs:
            self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing ``pi/actor`` loss.

        The loss function in SACLag is defined as:

        .. math::

            L = -Q^V (s, \pi (s)) + \lambda Q^C (s, \pi (s))

        where :math:`Q^V` is the min value of two reward critic networks outputs, :math:`Q^C` is the
        value of cost critic network, and :math:`\pi` is the policy network.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1, loss_q_r_2 = self._actor_critic.reward_critic(obs, action)
        loss_r = self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)
        loss_q_c = self._actor_critic.cost_critic(obs, action)[0]
        loss_c = self._lagrange.lagrangian_multiplier.item() * loss_q_c

        return (loss_r + loss_c).mean() / (1 + self._lagrange.lagrangian_multiplier.item())

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            },
        )
