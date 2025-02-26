import gymnasium as gym
import numpy as np
from gymnasium import spaces
from franka_bindings import Robot
import time
from typing import Optional, Dict, Any, Union, Type

class FrankaBaseEnv(gym.Env):
    """
    Base Franka Robot Gymnasium Environment.
    
    This class provides the core functionality for controlling a Franka robot,
    but leaves reward calculation, termination conditions, and target generation
    to be defined by task implementations provided by the user.
    """
    
    def __init__(self, task=None, ip="127.0.0.1", home_position=None):
        """
        Initialize the base Franka environment.
        
        Args:
            task: A task implementation that defines reward, termination, and target generation
            ip (str): IP address of the Franka robot
            home_position (np.ndarray, optional): Default home position for the robot
        """
        super(FrankaBaseEnv, self).__init__()
        
        # Store the task
        self.task = task
        
        # Initialize robot
        self.robot = Robot(ip)
        
        # Define action and observation spaces
        # Action space: 7 joint positions
        self.action_space = spaces.Box(
            low=np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]),
            high=np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]),
            dtype=np.float64
        )
        
        # Basic observation space: 7 joint positions + 7 joint velocities + current ee pose (16)
        # For poses we use 4x4 homogeneous transformation matrices (column-major)
        self.base_obs_space = spaces.Box(
            low=np.concatenate([
                self.action_space.low,           # joint positions
                -np.ones(7) * 2.0,              # joint velocities
                -np.ones(16) * 2.0,             # current ee pose (4x4 matrix)
            ]),
            high=np.concatenate([
                self.action_space.high,          # joint positions
                np.ones(7) * 2.0,               # joint velocities
                np.ones(16) * 2.0,              # current ee pose (4x4 matrix)
            ]),
            dtype=np.float64
        )
        
        # If task is provided, let it extend the observation space
        if self.task is not None:
            self.observation_space = self.task.extend_observation_space(self.base_obs_space)
        else:
            self.observation_space = self.base_obs_space
        
        # Home position (slightly raised neutral pose)
        if home_position is None:
            self.home_position = np.array([0.0, -0.2, 0.0, -1.5, 0.0, 1.5, 0.0])
        else:
            self.home_position = home_position
        
        # Initialize realtime control
        self.rt_control = None
        
        # Current state cache
        self.current_state = None
        
        # Set collision behavior (using conservative values)
        lower_torque_thresholds = [20.0] * 7  # Nm
        upper_torque_thresholds = [20.0] * 7  # Nm
        lower_force_thresholds = [10.0] * 6   # N (linear) and Nm (angular)
        upper_force_thresholds = [10.0] * 6   # N (linear) and Nm (angular)
        
        self.robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )
        
        # Start realtime control
        self.robot.start_realtime_control()
        self.rt_control = self.robot.get_realtime_control()
        time.sleep(0.5)  # Give time for control to initialize
    
    def _get_ee_pose(self):
        """Get end-effector pose from robot state."""
        return self.current_state.O_T_EE
    
    def _get_base_obs(self):
        """Get base observation (joint positions, velocities, and EE pose)."""
        # Update current state
        self.current_state = self.rt_control.get_current_state()
        ee_pose = self._get_ee_pose()  # This is already a 16-element array
        return np.concatenate([
            self.current_state.q,    # joint positions
            self.current_state.dq,   # joint velocities
            ee_pose,                 # current ee pose (16 elements)
        ])
    
    def _get_obs(self):
        """Get complete observation including task-specific elements."""
        base_obs = self._get_base_obs()
        
        if self.task is not None:
            return self.task.get_observation(base_obs, self)
        else:
            return base_obs
    
    def _get_info(self):
        """Get additional information."""
        info = {
            'joint_positions': self.current_state.q,
            'joint_velocities': self.current_state.dq,
            'ee_pose': self._get_ee_pose(),
        }
        
        if self.task is not None:
            task_info = self.task.get_info(self._get_obs(), self)
            info.update(task_info)
            
        return info
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Move to home position
        self.rt_control.set_target_position(self.home_position)
        
        # Wait until we reach close to home position
        while True:
            current_pos = np.array(self.rt_control.get_current_position())
            if np.linalg.norm(current_pos - self.home_position) < 0.01:
                break
            time.sleep(0.1)
        
        # Initialize task if provided
        if self.task is not None:
            self.task.reset(self)
        
        # Get observation after a short delay
        time.sleep(0.1)
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one environment step."""
        # Ensure action is within bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Send action to robot
        self.rt_control.set_target_position(action)
        
        # Wait a bit for the motion to start
        time.sleep(0.1)
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        # Get reward and termination from task if provided
        if self.task is not None:
            reward = self.task.compute_reward(observation, action, info, self)
            terminated = self.task.is_terminated(observation, info, self)
            truncated = self.task.is_truncated(observation, info, self)
        else:
            # Default values if no task is provided
            reward = 0.0
            terminated = False
            truncated = False
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources."""
        if self.task is not None:
            self.task.close()
            
        if self.robot:
            self.robot.stop()
            
    def get_current_observation(self):
        """Get the current observation without taking a step."""
        return self._get_obs() 