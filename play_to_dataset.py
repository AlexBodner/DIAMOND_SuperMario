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
import h5py
import uuid
def action_encode(action):
    action_vector = np.array([0,0,0]) #Follows WAD, there is no S
        
    if action == 2:    # derecha y saltar
        action_vector[-1] = 1
        action_vector[0] = 1
    elif action == 3:  # derecha y correr
        action_vector[-1] = 1
    elif action==1:   # right
        action_vector[-1] = 1
    elif action==6 : # left
        action_vector[1] = 1
    elif action==5:  # jump
        action_vector[0] = 1
    # Efficiently create a one-hot encoded vector using np.eye (identity matrix)
    return action_vector
# Función para guardar en archivo hdf5
def save_to_hdf5(filename, frames_data):
    with h5py.File(filename, 'w') as f:
        for i in range(len(frames_data['frames'])):
            f.create_dataset(f'frame_{i}_x', data=frames_data['frames'][i])
            #f.create_dataset(f'frame_{i}_xaux', data=frames_data['actions'][i])
            f.create_dataset(f'frame_{i}_y', data=frames_data['actions'][i])



def create_env_rgb(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage), render_mode="human", apply_api_compatibility=True,
                                     max_episode_steps=None)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward_rgb(GymnasiumEnvWrapper(env), None)
    env = CustomSkipFrame_rgb(env)
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

import copy
def play_game(env,  combo_actions=None, single_key_actions=None, relevant_keys=None, steps=1000):
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
    pygame.display.set_mode((256, 240))  # Small window for input capture
    # Variables para almacenar datos
    frames_data = {
        'frames': [],  # Para almacenar las imágenes
        'actions': [],  # Para almacenar las acciones anteriores
        'helperarr': []  # Para almacenar las banderas de vida perdida
    }
    terminated, truncated = True, True
    for step in range(steps):
        if terminated or truncated:
            obs, info = env.reset()

        pygame.event.pump()  # Update event queue
        keys = pygame.key.get_pressed()

        # Detect action
        action = detect_action(keys, combo_actions, single_key_actions, relevant_keys)

        # Debugging: Print action
        action_enco = action_encode(action)
        print(f"Action: {action}", action_enco)
        frames_data['frames'].append(copy.deepcopy(obs))  # Imagen
        frames_data['actions'].append(action_enco.copy())  # Acción actual

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Check if the level is completed
        if info.get("flag_get"):
            print("Level completed!")
            terminated = True
    save_to_hdf5(f"generated_dataset/mario_episode_data_{str(uuid.uuid4())}.hdf5", frames_data)

    env.close()
    pygame.quit()


def main():
    """Main function to set up and play the Mario game."""
    # Configuration
    world = 1
    stage = 1
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
    env = create_env_rgb(world, stage)
    obs_shape = env.observation_space.shape[0]
    action_count = len(SIMPLE_MOVEMENT)

    # Play game
    play_game(env, combo_actions, single_key_actions, relevant_keys, steps)


if __name__ == "__main__":
    import os
    if not os.path.exists("./generated_dataset/"):
        os.mkdir("generated_dataset")
    main()
