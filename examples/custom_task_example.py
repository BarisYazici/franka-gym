#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import time
import argparse
from franka_gym import FrankaBaseEnv, FrankaTask
from franka_gym.utils.robot_utils import transform_to_pose, pose_distance

class ReachingTask(FrankaTask):
    """
    Example implementation of a reaching task using the FrankaTask interface.
    
    The robot needs to reach a randomly generated target pose.
    """
    
    def __init__(self, 
                 workspace_limits=None, 
                 success_threshold=0.01,
                 position_weight=1.0,
                 orientation_weight=0.5):
        """
        Initialize the reaching task.
        
        Args:
            workspace_limits (dict, optional): Limits for target generation
            success_threshold (float): Distance threshold for success
            position_weight (float): Weight for position error in reward
            orientation_weight (float): Weight for orientation error in reward
        """
        # Workspace limits for target generation
        if workspace_limits is None:
            self.workspace_limits = {
                'x': [0.3, 0.7],    # meters
                'y': [-0.4, 0.4],   # meters
                'z': [0.2, 0.8],    # meters
            }
        else:
            self.workspace_limits = workspace_limits
            
        self.success_threshold = success_threshold
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        
        # Target pose
        self.target_pose = None
    
    def extend_observation_space(self, base_observation_space):
        """Extend observation space with target pose."""
        # Add target pose (16 elements for 4x4 transformation matrix)
        low = np.concatenate([
            base_observation_space.low, 
            -np.ones(16) * 2.0  # target ee pose (4x4 matrix)
        ])
        high = np.concatenate([
            base_observation_space.high, 
            np.ones(16) * 2.0   # target ee pose (4x4 matrix)
        ])
        
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)
    
    def get_observation(self, base_observation, env):
        """Add target pose to the observation."""
        return np.concatenate([base_observation, self.target_pose])
    
    def compute_reward(self, observation, action, info, env):
        """
        Compute reward based on distance to target.
        Reward is negative distance to target.
        """
        # Extract current and target poses
        base_obs_len = len(observation) - 16  # Subtract target pose length
        current_ee_pose = observation[base_obs_len - 16:base_obs_len] # last 16 elements of base obs
        target_ee_pose = observation[base_obs_len:]  # last 16 elements
        
        # Convert to position+quaternion format
        current_pose = transform_to_pose(current_ee_pose)
        target_pose = transform_to_pose(target_ee_pose)
        
        # Calculate distance
        distance = pose_distance(current_pose, target_pose, 
                                 position_weight=self.position_weight, 
                                 orientation_weight=self.orientation_weight)
        
        # Reward is negative distance (higher as robot gets closer to target)
        return -distance
    
    def is_terminated(self, observation, info, env):
        """Episode terminates when the distance is below threshold."""
        # Use distance from info
        return info.get('distance', float('inf')) < self.success_threshold
    
    def reset(self, env):
        """Generate a new random target."""
        self.target_pose = self._generate_target_pose()
    
    def get_info(self, observation, env):
        """Provide distance information."""
        # Extract current and target poses
        base_obs_len = len(observation) - 16  # Subtract target pose length
        current_ee_pose = observation[base_obs_len - 16:base_obs_len]
        target_ee_pose = observation[base_obs_len:]
        
        # Calculate distance
        current_pose = transform_to_pose(current_ee_pose)
        target_pose = transform_to_pose(target_ee_pose)
        distance = pose_distance(current_pose, target_pose, 
                                 position_weight=self.position_weight, 
                                 orientation_weight=self.orientation_weight)
        
        return {
            'distance': distance,
            'target_pose': transform_to_pose(self.target_pose)
        }
    
    def _generate_target_pose(self):
        """Generate a random target pose within workspace limits."""
        # Random position within workspace
        position = np.array([
            np.random.uniform(*self.workspace_limits['x']),
            np.random.uniform(*self.workspace_limits['y']),
            np.random.uniform(*self.workspace_limits['z'])
        ])
        
        # Create a homogeneous transformation matrix with downward-facing orientation
        # This creates a pose where the end-effector is pointing downward
        target_pose = np.array([
            [1.0, 0.0, 0.0, position[0]],  # First column: x-axis
            [0.0, -1.0, 0.0, position[1]], # Second column: y-axis
            [0.0, 0.0, -1.0, position[2]], # Third column: z-axis
            [0.0, 0.0, 0.0, 1.0]          # Fourth column: homogeneous part
        ])
        
        # Convert to column-major array
        return target_pose.flatten('F')


def manual_control(env, episodes=3):
    """Run the environment with manual predefined joint configurations."""
    try:
        # Reset the environment
        observation, info = env.reset()
        print_initial_state(observation, info)
        
        # Run for a few episodes
        for episode in range(episodes):
            print(f"\nEpisode {episode + 1}")
            observation, info = env.reset()
            
            # Run steps until termination
            episode_reward = 0
            step = 0
            
            # Define some example joint configurations to try
            # These are just example configurations within the joint limits
            joint_configs = [
                [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],  # Slightly lower position
                [0.5, -0.2, 0.3, -1.5, 0.2, 1.8, 0.1],  # Moved to the right
                [-0.5, -0.2, -0.3, -1.5, -0.2, 1.8, -0.1],  # Moved to the left
                [0.0, -0.5, 0.0, -1.2, 0.0, 2.0, 0.0],  # Higher position
            ]
            
            config_idx = 0
            while True:
                # Choose the next joint configuration
                action = np.array(joint_configs[config_idx])
                config_idx = (config_idx + 1) % len(joint_configs)
                
                # Take step and wait for motion to complete
                observation, reward, terminated, truncated, info = env.step(action)
                wait_for_motion(env, action)  # Wait for robot to reach target
                
                episode_reward += reward
                step += 1
                
                if step % 5 == 0:
                    print(f"\nStep {step}")
                    print(f"Current joint config: {observation[:7]}")
                    print(f"Distance to target: {info['distance']:.4f}")
                    print(f"Reward: {reward:.4f}")
                
                if terminated or truncated:
                    print(f"\nEpisode finished after {step} steps")
                    print(f"Final distance to target: {info['distance']:.4f}")
                    print(f"Total reward: {episode_reward:.4f}")
                    break
                
                time.sleep(0.1)  # Small delay between commands
                
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        env.close()


def wait_for_motion(env, target_joints, threshold=0.01):
    """Wait until the robot reaches close to target joint positions."""
    while True:
        # Access the unwrapped environment to get the current observation
        observation = env.unwrapped.get_current_observation()
        current_joints = observation[:7]
        if np.linalg.norm(current_joints - target_joints) < threshold:
            break
        time.sleep(0.1)


def print_initial_state(observation, info):
    """Print information about the initial state."""
    print("\nInitial state:")
    print(f"Joint positions: {observation[:7]}")
    print(f"Joint velocities: {observation[7:14]}")
    
    # Calculate the base observation length
    base_obs_len = len(observation) - 16  # Subtract target pose length
    
    # Convert transformation matrices to poses
    current_ee_transform = observation[base_obs_len - 16:base_obs_len]
    target_ee_transform = observation[base_obs_len:]
    
    current_ee_pose = transform_to_pose(current_ee_transform)
    target_ee_pose = transform_to_pose(target_ee_transform)
    
    print(f"Current EE pose: {current_ee_pose}")
    print(f"Target EE pose: {target_ee_pose}")
    print(f"Distance to target: {info['distance']:.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Custom Task Example with Franka Robot')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--success-threshold', type=float, default=0.01,
                        help='Distance threshold for success')
    args = parser.parse_args()
    
    # Create the custom task
    task = ReachingTask(success_threshold=args.success_threshold)
    
    # Create the environment with the custom task
    env = gym.make('FrankaBase-v0', task=task)
    
    try:
        manual_control(env, episodes=args.episodes)
    finally:
        env.close()


if __name__ == "__main__":
    main() 