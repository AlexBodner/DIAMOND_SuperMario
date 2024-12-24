import pygame
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium as gym
import numpy as np
import torch
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch.nn.functional as F
from wrappers import *


def create_env(world, stage):
    """Create and return the game environment."""
    env = gym_super_mario_bros.make(
        f"SuperMarioBros-{world}-{stage}-v0",
        render_mode="human",
        apply_api_compatibility=True,
        max_episode_steps=500,
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward(GymnasiumEnvWrapper(env), None)
    env = CustomSkipFrame(env)
    env = GymnasiumEnvWrapper(env)
    return env


def detect_action(keys, combo_actions, single_key_actions, relevant_keys):
    """Detect the appropriate action based on key presses."""
    pressed_keys = frozenset(key for key in relevant_keys if keys[key])
    action = combo_actions.get(pressed_keys)
    if action is None:
        for key, mapped_action in single_key_actions.items():
            if keys[key]:
                return mapped_action
    return action if action is not None else 0  # Default to NOOP


def play_game(env,combo_actions=None, single_key_actions=None, relevant_keys=None, steps=5000):
    """
    Play the Mario game using keyboard input and optional model prediction.

    Args:
        env: The Mario environment.
        model: The trained PPO model (optional).
        combo_actions: Dictionary of key combinations and actions.
        single_key_actions: Dictionary of single key actions.
        relevant_keys: Set of relevant keys to listen for.
        steps: Maximum number of steps to run the game loop.
    """
    pygame.init()
    pygame.display.set_mode((100, 100))  # Small window for input capture

    terminated, truncated = True, True
    for step in range(steps):
        if terminated or truncated:
            obs, info = env.reset()

        pygame.event.pump()  # Update event queue
        keys = pygame.key.get_pressed()

        # Detect action
        action = detect_action(keys, combo_actions, single_key_actions, relevant_keys)

        # Debugging: Print action
        print(f"Action: {action}")

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Check if the level is completed
        if info.get("flag_get"):
            print("Level completed!")
            break

    env.close()
    pygame.quit()


def main():
    """Main function to set up and play the Mario game."""
    # Configuration
    world = 1
    stage = 1
    saved_path = "ppo_models"
    steps = 1000

    # Key mappings
    combo_actions = {frozenset([pygame.K_w, pygame.K_d]): 2}  # Right + Jump
    single_key_actions = {
        pygame.K_w: 5,  # Jump ('A')
        pygame.K_d: 1,  # Move right ('right')
        pygame.K_a: 6,  # Move left ('left')
        pygame.K_s: 0,  # No operation ('NOOP')
    }
    relevant_keys = {pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d}

    # Create environment and load model
    env = create_env(world, stage)
    obs_shape = env.observation_space.shape[0]
    action_count = len(SIMPLE_MOVEMENT)

    # Play game
    play_game(env, combo_actions, single_key_actions, relevant_keys, steps)


if __name__ == "__main__":

    main()
