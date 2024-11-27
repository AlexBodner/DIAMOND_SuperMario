import torch
import numpy as np
import torch.nn.functional as F
import h5py
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from src.model import PPO
from wrappers import *# CustomReward,CustomReward_rgb, CustomSkipFrame, GymnasiumEnvWrapper,process_frame
import pandas as pd
import numpy as np
import random
# Function to convert an action index to a one-hot encoded vector
def one_hot_encode(action_index, num_actions):
    # Efficiently create a one-hot encoded vector using np.eye (identity matrix)
    return np.eye(1, num_actions, action_index, dtype=int).flatten()
# Functión para predecir la acción
def predict_action(model, obs):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    # Mover la observación a la GPU si está disponible
    if torch.cuda.is_available():
        obs = obs.cuda()
    logits, value = model(obs)
    policy = F.softmax(logits, dim=1)
    action = torch.argmax(policy).item()
    return action
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Crear entorno
def create_env(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage), render_mode=None, apply_api_compatibility=True,
                                     max_episode_steps=1000)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward(GymnasiumEnvWrapper(env), None)
    env = CustomSkipFrame(env)
    env = GymnasiumEnvWrapper(env)
    return env
    
def create_env_rgb(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage), render_mode=None, apply_api_compatibility=True,
                                     max_episode_steps=1000)

    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward_rgb(GymnasiumEnvWrapper(env), None)
    env = CustomSkipFrame_rgb(env)
    env = GymnasiumEnvWrapper(env)
    return env

# Función para guardar en archivo hdf5
def save_to_hdf5(filename, frames_data):
    with h5py.File(filename, 'w') as f:
        for i in range(len(frames_data['frames'])):
            f.create_dataset(f'frame_{i}_x', data=frames_data['frames'][i])
            f.create_dataset(f'frame_{i}_xaux', data=frames_data['actions'][i])
            f.create_dataset(f'frame_{i}_y', data=frames_data['target_actions'][i])
            f.create_dataset(f'frame_{i}_helperarr', data=frames_data['helperarr'][i])



def main(n_episodes = 60, epsilons = [0.1,0.15,0.3,0.5,0.7,0.8,0.9]):
    # Inicialización
    world = 1
    stage = 1
    saved_path = "ppo_models"
    env = create_env(world, stage)

    actions = SIMPLE_MOVEMENT
    model = PPO(env.observation_space.shape[0], len(actions))

    # Cargar modelo
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage), map_location=lambda storage, loc: storage))
        model.eval()

    for episode in range(n_episodes):
        epsilon = random.choice(epsilons)

        print("Started episode",episode, "with epsilon",epsilon)
        env = create_env(world, stage)
        env_rgb =  create_env_rgb(world, stage)
        # Variables para almacenar datos
        frames_data = {
            'frames': [],  # Para almacenar las imágenes
            'actions': [],  # Para almacenar las acciones anteriores
            'target_actions': [],  # Para almacenar las acciones de destino
            'helperarr': []  # Para almacenar las banderas de vida perdida
        }
        # Empezar a jugar el juego y recolectar datos
        terminated, truncated = True, True
        previous_lives = None  # Variable para trackear las vidas anteriores
        prev_action = one_hot_encode(3,len(actions) ) #NOOP
        reward = 0
        for step in range(1000):  # 1000 frames (~1 minuto)
            if terminated or truncated:
                obs, info = env.reset(seed = 42)
                obs_rgb, _ = env_rgb.reset(seed = 42)
                #print("rgb shape",obs_rgb.shape)
            # Verificar si se ha perdido una vida
            current_lives = info.get("life", 2)  # Supongamos que 3 es el número inicial de vidas, si no está disponible, se asume 2
            life_lost = False
            if previous_lives is not None and current_lives < previous_lives:
                life_lost = True
            previous_lives = current_lives
            frames_data['helperarr'].append([reward,int(life_lost)])  # 1 si se perdió una vida, 0 si no
            # Predecir la acción
            if random.random()<epsilon:
                action =env.action_space.sample()
            else:
                action = model.predict_action(obs)

            # Almacenar los datos de cada frame
            frames_data['frames'].append(obs_rgb)  # Imagen
            frames_data['actions'].append(prev_action)  # Acción anterior
            #print(one_hot_encode(action,len(actions) ))
            obs_rgb, reward_r, terminated, truncated, info = env_rgb.step(action)

            obs, reward, terminated, truncated, info = env.step(action)
            if reward!=reward_r:
                print("differnce in rewards between played and saved")
            action = one_hot_encode(action,len(actions) )
            frames_data['target_actions'].append(action)  # Aquí irían las acciones de destino (debe adaptarse según el formato requerido)

            prev_action =action
            # Mostrar el estado del juego
            #env.render()
            if info["flag_get"]:
                print(f"World {world} stage {stage} completed")
                break

        # Guardar los datos recolectados en un archivo .hdf5
        save_to_hdf5(f"diamond/dataset_files/mario_episode_data_{episode}.hdf5", frames_data)

        env.close()
        env_rgb.close()

if __name__ =="__main__":
    main()
