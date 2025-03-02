#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import franka_gym
import time
import argparse
import os
from franka_gym.utils.robot_utils import transform_to_pose, pose_distance
from franka_gym import FrankaTask, FrankaBaseEnv

# Import Stable Baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

class ReachingTask(FrankaTask):
    """
    Reaching task implementation for the Franka robot.
    
    The robot needs to reach a randomly generated target pose.
    """
    
    def __init__(self, 
                 workspace_limits=None, 
                 success_threshold=0.05,
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
        
        # Target pose (4x4 transformation matrix in flattened column-major format)
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
        
        # Convert to position+quaternion format for distance calculation
        current_pose = transform_to_pose(current_ee_pose)
        target_pose = transform_to_pose(target_ee_pose)
        
        # Calculate distance
        distance = pose_distance(current_pose, target_pose, 
                                 position_weight=self.position_weight, 
                                 orientation_weight=self.orientation_weight)
        
        # Reward is negative distance (higher as robot gets closer to target)
        reward = -distance
        
        # Add bonus reward when reaching the target
        if distance < self.success_threshold:
            reward += 10.0
            
        return reward
    
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
            'target_pose': transform_to_pose(target_ee_pose)
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

def wait_for_motion(env, target_joints, threshold=0.01):
    """Wait until the robot reaches close to target joint positions."""
    while True:
        # Access the unwrapped environment to get the current observation
        observation = env.unwrapped.get_current_observation()
        current_joints = observation[:7]
        if np.linalg.norm(current_joints - target_joints) < threshold:
            break
        time.sleep(0.1)

# Custom callback for printing training progress
class RobotTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RobotTrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
        self.step_count = 0
        
    def _on_step(self):
        self.step_count += 1
        # Get the most recent reward (assumes single environment)
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        
        if self.step_count % 10 == 0:
            # Get the most recent action
            action = self.locals.get('actions', [np.zeros(7)])[0]
            print(f"Step {self.step_count}: Action={action}, Reward={reward:.4f}")
        
        # Check if episode is done
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            print(f"\nEpisode {self.episode_count} completed with reward: {self.current_episode_reward:.4f}")
            self.current_episode_reward = 0
            
        return True

# Custom environment wrapper to wait for robot motion
class RobotWaitWrapper(gym.Wrapper):
    def __init__(self, env, wait_threshold=0.01, debug=False):
        super(RobotWaitWrapper, self).__init__(env)
        self.wait_threshold = wait_threshold
        self.debug = debug
        
    def step(self, action):
        if self.debug:
            print(f"Executing action: {action}")
        
        # Execute the action in the environment
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Wait for the robot to reach the target position
        try:
            # Get current joint positions
            current_joints = observation[:7]
            
            # Wait until robot reaches close to target
            start_time = time.time()
            while np.linalg.norm(current_joints - action) > self.wait_threshold:
                # Get the latest observation
                latest_obs = self.env.unwrapped.get_current_observation()
                current_joints = latest_obs[:7]
                
                # Prevent infinite waiting (5 second timeout)
                if time.time() - start_time > 5.0:
                    if self.debug:
                        print("Timeout waiting for robot to reach target")
                    break
                
                time.sleep(0.05)
            
            if self.debug:
                print(f"Motion completed, distance: {np.linalg.norm(current_joints - action):.4f}")
                
        except Exception as e:
            print(f"Error during wait: {e}")
        
        return observation, reward, terminated, truncated, info

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

def train_agent(env, algorithm='PPO', total_timesteps=100000, save_path='models', use_tensorboard=True, debug=False):
    """Train an RL agent on the environment."""
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Wrap the environment to wait for robot motion
    print("Wrapping environment with RobotWaitWrapper...")
    env = RobotWaitWrapper(env, debug=debug)
    
    # Setup callbacks for saving, evaluation and custom monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,  # Save more frequently
        save_path=save_path,
        name_prefix=f"franka_{algorithm}"
    )
    
    # Custom callback for detailed progress information
    robot_callback = RobotTrainingCallback(verbose=1)
    
    # Combine callbacks
    callbacks = [checkpoint_callback, robot_callback]
    
    # Set up tensorboard log directory or disable it
    tensorboard_log = f"{save_path}/logs/" if use_tensorboard else None
    
    # Choose algorithm with appropriate hyperparameters for real robot control
    if algorithm == 'PPO':
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=tensorboard_log,
            learning_rate=3e-4,
            n_steps=2048,  # Default, but specified for clarity
            batch_size=64,  # Smaller batch size for faster updates
            n_epochs=10,    # Default
            gamma=0.99,     # Discount factor
            gae_lambda=0.95 # GAE parameter
        )
    elif algorithm == 'SAC':
        model = SAC(
            "MlpPolicy", 
            env,
            verbose=1, 
            tensorboard_log=tensorboard_log,
            learning_rate=3e-4,
            buffer_size=10000,  # Smaller buffer for faster learning
            learning_starts=1000,  # Start learning earlier
            batch_size=64,       # Smaller batch size
            tau=0.005,           # Soft update coefficient
            gamma=0.99,          # Discount factor
            train_freq=1,        # Update policy every step
            gradient_steps=1     # Gradient steps per update
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train the agent
    print(f"Training {algorithm} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # Save the final model
    model_path = f"{save_path}/franka_{algorithm}_final"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model, model_path

def evaluate_agent(env, model_path, episodes=5):
    """Evaluate a trained agent."""
    print(f"Loading model from {model_path}")
    
    # Determine the algorithm from model_path
    if "PPO" in model_path:
        model = PPO.load(model_path)
    elif "SAC" in model_path:
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Cannot determine algorithm from {model_path}")
    
    print("Evaluating model...")
    
    try:
        for episode in range(episodes):
            print(f"\nEvaluation Episode {episode + 1}")
            observation, info = env.reset()
            episode_reward = 0
            step = 0
            done = False
            
            while not done:
                # Get action from the agent
                action, _ = model.predict(observation, deterministic=True)
                
                # Execute action
                observation, reward, terminated, truncated, info = env.step(action)
                wait_for_motion(env, action)  # Wait for robot to reach target
                
                episode_reward += reward
                step += 1
                done = terminated or truncated
                
                print(f"\nStep {step}")
                print(f"Action taken: {action}")
                print(f"Distance to target: {info['distance']:.4f}")
                print(f"Reward: {reward:.4f}")
                
                if done:
                    print(f"\nEpisode finished after {step} steps")
                    print(f"Final distance to target: {info['distance']:.4f}")
                    print(f"Total reward: {episode_reward:.4f}")
                
                time.sleep(0.1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
    finally:
        env.close()

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
    parser = argparse.ArgumentParser(description='Franka Robot Reach Task with RL')
    parser.add_argument('--mode', type=str, default='manual', choices=['manual', 'train', 'evaluate'],
                        help='Execution mode: manual control, train RL agent, or evaluate trained agent')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'SAC'],
                        help='RL algorithm to use (for train mode)')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps for training (for train mode)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a trained model (for evaluate mode)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run (for manual or evaluate mode)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enable TensorBoard logging (for train mode)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable debug mode with more verbose output')
    parser.add_argument('--success-threshold', type=float, default=0.05,
                        help='Distance threshold for success (in meters)')
    args = parser.parse_args()
    
    # Create the reaching task
    task = ReachingTask(success_threshold=args.success_threshold)
    
    # Create the environment with the reaching task
    env = gym.make('FrankaBase-v0', task=task)
    
    try:
        if args.mode == 'manual':
            manual_control(env, episodes=args.episodes)
        elif args.mode == 'train':
            model, model_path = train_agent(
                env, 
                algorithm=args.algorithm, 
                total_timesteps=args.timesteps,
                use_tensorboard=args.tensorboard, 
                debug=args.debug
            )
            # Optionally evaluate after training
            evaluate_agent(env, model_path)
        elif args.mode == 'evaluate':
            if args.model is None:
                print("Error: Model path must be provided for evaluate mode")
                return
            evaluate_agent(env, args.model, episodes=args.episodes)
    finally:
        env.close()

if __name__ == "__main__":
    main() 