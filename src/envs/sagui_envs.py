import copy
from math import sqrt
import numpy as np
from safety_gymnasium.assets.free_geoms import Vases
from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.registration import register


sagui_env_ids = ['SafetyPointGuide1-v0',
                 'SafetyPointGuide2-v0',
                 'SafetyPointGuide3-v0',
                 'SafetyPointStudent1-v0',
                 'SafetyPointStudent2-v0',
                 'SafetyPointStudent3-v0',
                 ]


# Register sagui environments with safety_gymnasium
# https://github.com/PKU-Alignment/safety-gymnasium/blob/6c777c6f892d4db400dec4a4f30f24db0dd52fde/safety_gymnasium/__init__.py#L55
def register_sagui_envs() -> None:
    for env_id in sagui_env_ids:
        config = {'agent_name': 'Point'}
        kwargs = {'config': config, 'task_id': env_id}
        register(id=env_id, entry_point='omnisafe.envs.sagui_builder:SaguiBuilder',
                 kwargs=kwargs, max_episode_steps=1000)


# Physics coeficients for modifying the dynamics
coefs = None


def set_coef_dict(coef_dict: dict):
    global coefs  # bad practice but ok
    coefs = coef_dict


def _modify_dyn(model, coef_dict: dict):
    for name, mult in coef_dict.items():
        atr: np.ndarray = getattr(model, name)
        atr[:] *= mult


def _set_default_dyn(model):
    model.dof_damping[0] *= 1.5  # Axis X
    model.dof_damping[1] *= 1.5  # Axis Z
    # model.dof_damping[2] *= 1.0  # Steering

    if coefs != None:
        _modify_dyn(model, coefs)


def _set_adversarial_dyn(model):
    model.dof_damping *= 1.0
    model.body_mass *= 1.0

# Took it from
# https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/tasks


class GuideLevel1(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self._add_geoms(Hazards(num=1, keepout=0.75, size=0.7, locations=[(0, 0)]))
        self._add_geoms(Sigwalls(num=4, locate_factor=2.5, is_constrained=True))

        self.placements_conf.extents = [-1.75, -1.75, 1.75, 1.75]

    def calculate_reward(self):
        x0, y0, _ = self.last_robot_pos
        x, y, _ = self.agent.pos
        reward = sqrt((x - x0)**2 + (y - y0)**2)

        return reward

    def specific_reset(self):
        self.last_robot_pos = self.agent.pos
        _set_default_dyn(self.model)

    def specific_step(self):
        self.last_robot_pos = self.agent.pos

    def update_world(self):
        self.last_robot_pos = self.agent.pos

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return False  # self.dist_goal() <= self.goal.size


# Took it from
# https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/tasks
class GuideLevel2(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        # self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=8, keepout=0.5))
        self._add_geoms(Sigwalls(num=4, locate_factor=3.2, is_constrained=True))
        # self._add_free_geoms(Vases(num=1, is_constrained=False, keepout=0.18))

        # self.last_robot_pos = None
        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 5
        # self.vases.num = 8
        # self.vases.is_constrained = True

    def calculate_reward(self):
        x0, y0, _ = self.last_robot_pos
        x, y, _ = self.agent.pos
        reward = sqrt((x - x0)**2 + (y - y0)**2)

        return reward

    def specific_reset(self):
        self.last_robot_pos = self.agent.pos
        _set_default_dyn(self.model)

    def specific_step(self):
        self.last_robot_pos = self.agent.pos

    def update_world(self):
        self.last_robot_pos = self.agent.pos

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return False  # self.dist_goal() <= self.goal.size

class GuideLevel3(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        # self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=8, keepout=0.22))
        self._add_geoms(Sigwalls(num=4, locate_factor=3.5, is_constrained=True))
        self._add_free_geoms(Vases(num=1, is_constrained=False, keepout=0.18))

        self.last_robot_pos = None
        self.placements_conf.extents = [-2, -2, 2, 2]

        self.hazards.num = 8
        self.vases.num = 8
        self.vases.is_constrained = True

    def calculate_reward(self):
        x0, y0, _ = self.last_robot_pos
        x, y, _ = self.agent.pos
        reward = sqrt((x - x0)**2 + (y - y0)**2)

        return reward

    def specific_reset(self):
        self.last_robot_pos = self.agent.pos
        _set_default_dyn(self.model)

    def specific_step(self):
        self.last_robot_pos = self.agent.pos

    def update_world(self):
        self.last_robot_pos = self.agent.pos

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return False  # self.dist_goal() <= self.goal.size


class StudentLevel1(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=1, keepout=0.75, size=0.7, locations=[(0, 0)]))
        self._add_geoms(Sigwalls(num=4, locate_factor=2.5, is_constrained=True))

        self.placements_conf.extents = [-1.75, -1.75, 1.75, 1.75]

        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal
        return reward

    def specific_reset(self):
        _set_default_dyn(self.model)

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size


# Took it from
# https://github.com/PKU-Alignment/safety-gymnasium/tree/main/safety_gymnasium/tasks
class StudentLevel2(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=8, keepout=0.5))
        self._add_geoms(Sigwalls(num=4, locate_factor=3.2, is_constrained=True))

        self.placements_conf.extents = [-2, -2, 2, 2]

        self.last_dist_goal = None

        self.hazards.num = 5

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal
        return reward

    def specific_reset(self):
        _set_default_dyn(self.model)

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

class StudentLevel3(BaseTask):
    """An agent must navigate to a goal."""

    def __init__(self, config) -> None:
        super().__init__(config=config)

        self._add_geoms(Goal(keepout=0.305))
        self._add_geoms(Hazards(num=8, keepout=0.22))
        self._add_geoms(Sigwalls(num=4, locate_factor=3.5, is_constrained=True))
        self._add_free_geoms(Vases(num=1, is_constrained=False, keepout=0.18))

        self.placements_conf.extents = [-2.5, -2.5, 2.5, 2.5]

        self.last_dist_goal = None

        self.hazards.num = 8
        self.vases.num = 8
        self.vases.is_constrained = True

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal
        return reward

    def specific_reset(self):
        _set_default_dyn(self.model)
        # _set_adversarial_dyn(self.model)

    def specific_step(self):
        pass

    def update_world(self):
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

