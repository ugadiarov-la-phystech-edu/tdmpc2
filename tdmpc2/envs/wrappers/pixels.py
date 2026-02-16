from collections import deque

import gymnasium as gym
import numpy as np
import torch


class Pixels(gym.Wrapper):
	def __init__(self, env, cfg, num_frames=3):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self._size = self.cfg.obs_image_size
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, self._size, self._size), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)

	def _get_obs(self, is_reset=False):
		frame = self.env.render(width=self._size, height=self._size).transpose(2, 0, 1)
		num_frames = self._frames.maxlen if is_reset else 1
		for _ in range(num_frames):
			self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames))

	def reset(self):
		self.env.reset()
		return self._get_obs(is_reset=True)

	def step(self, action):
		_, reward, done, info = self.env.step(action)
		return self._get_obs(), reward, done, info

	def close(self):
		self.env.close()