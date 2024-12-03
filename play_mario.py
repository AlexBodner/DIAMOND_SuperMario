from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gymnasium as gym

#agents from https://github.com/vietnh1009/Super-mario-bros-PPO-pytorch
import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F
from wrappers import * 

def predict_action(model, obs):
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    # Move obs to GPU if available
    if torch.cuda.is_available():
        obs = obs.cuda()
    logits, value = model(obs)
    policy = F.softmax(logits, dim=1)
    print(policy)
    action = torch.argmax(policy).item()
    return action

def create_env(world, stage):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage), render_mode='human', apply_api_compatibility=True,max_episode_steps= 500)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward(GymnasiumEnvWrapper(env), None)
    env = CustomSkipFrame(env)
    env = GymnasiumEnvWrapper(env)
    return env

world = 1
stage = 4

saved_path= "ppo_models"
env = create_env(world,stage)

# Load model
actions = SIMPLE_MOVEMENT
model = PPO(env.observation_space.shape[0], len(actions))
print("size",env.observation_space.shape[0])
if torch.cuda.is_available():
    model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, 1)))
    model.cuda()
else:
    model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, 1),
                                         map_location=lambda storage, loc: storage))
    model.eval()

#Play game
terminated, truncated = True,True
for step in range(5000):
    if terminated or truncated:
        obs,info = env.reset()

    #print(obs.shape)
    #action=env.action_space.sample() # For random sampling
    action = 0#model.predict_action( obs)
    print(action)
    obs, reward, terminated, truncated , info  = env.step(action)
    print(obs.dtype)
    #print(obs.shape)
    print(info)
    env.render()
    if info["flag_get"]:
        print("World {} stage {} completed".format(world, stage))
        break

env.close()

