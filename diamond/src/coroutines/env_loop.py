<<<<<<< HEAD
import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from . import coroutine
from envs import TorchEnv, WorldModelEnv


@coroutine
def make_env_loop(
    env: Union[TorchEnv, WorldModelEnv], model: nn.Module, epsilon: float = 0.3
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield


    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            #act = Categorical(logits=logits_act).sample()
            
            # Optional: epsilon-greedy exploration
            if random.random() < epsilon:
                act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)
            else:
                
                act = model.predict_action(obs)

            next_obs, rew, end, trunc, info = env.step(act)


            # Store transition

            dead = torch.logical_or(end, trunc)

            #if dead.any():
            #    with torch.no_grad():
            #        _, val_final_obs, _ = model.predict_action(info["final_observation"])

            all_.append([obs, torch.tensor([act.item()]), rew, end, trunc])
            infos.append(info)


            obs = next_obs
            n += 1


        # Stack all transitions
        #print("all",all_)
        all_obs, act, rew, end, trunc = (torch.stack(x, dim=1) for x in zip(*all_))

=======
import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from . import coroutine
from envs import TorchEnv, WorldModelEnv


@coroutine
def make_env_loop(
    env: Union[TorchEnv, WorldModelEnv], model: nn.Module, epsilon: float = 0.3
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield


    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            #act = Categorical(logits=logits_act).sample()
            
            # Optional: epsilon-greedy exploration
            if random.random() < epsilon:
                act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)
            else:
                
                act = model.predict_action(obs)

            next_obs, rew, end, trunc, info = env.step(act)


            # Store transition

            dead = torch.logical_or(end, trunc)

            #if dead.any():
            #    with torch.no_grad():
            #        _, val_final_obs, _ = model.predict_action(info["final_observation"])

            all_.append([obs, torch.tensor([act.item()]), rew, end, trunc])
            infos.append(info)


            obs = next_obs
            n += 1


        # Stack all transitions
        #print("all",all_)
        all_obs, act, rew, end, trunc = (torch.stack(x, dim=1) for x in zip(*all_))

>>>>>>> origin/master
        num_steps = yield all_obs, act, rew, end, trunc, infos