from gymnasium import *
import numpy as np
from gymnasium.spaces import Box
import cv2
import subprocess as sp
import gymnasium as gym
class CustomReward(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, terminated ,truncated, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        #reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if terminated or truncated:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward, terminated,truncated, info

    def reset(self,**kwargs):
        self.curr_score = 0
        obs, info= self.env.reset(**kwargs)
        return process_frame(obs),info
class CustomReward_rgb(Wrapper):
    def __init__(self, env=None, monitor=None):
        super(CustomReward_rgb, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, terminated ,truncated, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        #state = process_frame(state)
        #reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if terminated or truncated:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        return state, reward, terminated,truncated, info

    def reset(self,**kwargs):
        self.curr_score = 0
        obs, info= self.env.reset(**kwargs)
        return obs,info

        #return process_frame(obs),info



class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, terminated,truncated, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if terminated or truncated:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, terminated,truncated, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, terminated,truncated, info

    def reset(self,**kwargs):
        state,info = self.env.reset(**kwargs)
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)
class CustomSkipFrame_rgb(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame_rgb, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, terminated,truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, terminated,truncated, info

        return state, total_reward, terminated,truncated, info

    def reset(self,**kwargs):
        state,info = self.env.reset(**kwargs)
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return state
    
def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))
class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        # try:
        self.pipe.stdin.write(image_array.tostring())
        # except IOError as e:
        #     if e.errno == errno.EPIPE:
        #         pass
class GymnasiumEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        if not type(observation)== tuple:
            return (observation,{})  # gymnasium expects a tuple (obs, info)
        #print("o",observation)
        return observation
    def step(self, action):
        observation, reward, termianted, truncated, info = self.env.step(action)
        return observation, reward, termianted, truncated, info  # gymnasium expects 5 values (obs, reward,     , truncated, info)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()