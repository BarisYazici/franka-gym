import gymnasium as gym
import numpy as np
from gymnasium import spaces
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Optional

class FrankaTask(ABC):
    """
    Abstract base class defining the interface for Franka robot tasks.
    
    Users should implement this interface to define custom tasks for the Franka robot,
    including reward functions, termination conditions, and task-specific observations.
    """
    
    @abstractmethod
    def extend_observation_space(self, base_observation_space: spaces.Box) -> spaces.Box:
        """
        Extend the base observation space with task-specific observations.
        
        Args:
            base_observation_space: The base observation space from the environment
                (includes joint positions, velocities, and EE pose)
                
        Returns:
            Extended observation space
        """
        pass
    
    @abstractmethod
    def get_observation(self, base_observation: np.ndarray, env) -> np.ndarray:
        """
        Get the full observation including task-specific elements.
        
        Args:
            base_observation: Base observation from the environment
                (includes joint positions, velocities, and EE pose)
            env: The environment instance
            
        Returns:
            Full observation including task-specific elements
        """
        pass
    
    @abstractmethod
    def compute_reward(self, observation: np.ndarray, action: np.ndarray, 
                      info: Dict[str, Any], env) -> float:
        """
        Compute the reward for the current state.
        
        Args:
            observation: The current observation
            action: The action that was taken
            info: Additional information
            env: The environment instance
            
        Returns:
            Scalar reward value
        """
        pass
    
    @abstractmethod
    def is_terminated(self, observation: np.ndarray, info: Dict[str, Any], env) -> bool:
        """
        Determine if the episode should terminate successfully.
        
        Args:
            observation: The current observation
            info: Additional information
            env: The environment instance
            
        Returns:
            True if episode should terminate, False otherwise
        """
        pass
    
    def is_truncated(self, observation: np.ndarray, info: Dict[str, Any], env) -> bool:
        """
        Determine if the episode should be truncated (e.g., due to time limits).
        
        Args:
            observation: The current observation
            info: Additional information
            env: The environment instance
            
        Returns:
            True if episode should be truncated, False otherwise
        """
        # Default implementation - no truncation
        return False
    
    @abstractmethod
    def reset(self, env) -> None:
        """
        Reset the task (e.g., generate a new target).
        
        Args:
            env: The environment instance
        """
        pass
    
    def get_info(self, observation: np.ndarray, env) -> Dict[str, Any]:
        """
        Get task-specific info.
        
        Args:
            observation: The current observation
            env: The environment instance
            
        Returns:
            Dictionary of task-specific information
        """
        # Default implementation - no additional info
        return {}
    
    def close(self) -> None:
        """
        Clean up task resources.
        """
        # Default implementation - no cleanup needed
        pass 