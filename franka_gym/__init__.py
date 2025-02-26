"""
Franka Gym - A Gymnasium environment for Franka robot control
"""

from gymnasium.envs.registration import register

# Import environment classes
from franka_gym.envs.franka_base_env import FrankaBaseEnv

# Import task interface
from franka_gym.tasks import FrankaTask

# Register the environments
register(
    id='FrankaBase-v0',
    entry_point='franka_gym.envs:FrankaBaseEnv',
    max_episode_steps=1000,
)

__version__ = "0.1.0"
__all__ = ['FrankaBaseEnv', 'FrankaTask']

