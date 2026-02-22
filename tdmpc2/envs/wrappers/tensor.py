from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""
	Wrapper for converting numpy arrays to torch tensors.
	"""

	def __init__(self, env):
		super().__init__(env)
		self._wrapped_vectorized = env.__class__.__name__ == 'Vectorized'
	
	def rand_act(self):
		if self._wrapped_vectorized:
			return self.env.rand_act()
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None, **kwargs):
		if self._wrapped_vectorized:
			obs = self.env.reset(**kwargs)
		else:
			obs = self.env.reset()
		return self._obs_to_tensor(obs)

	def reset_wait(self):
		return self._obs_to_tensor(self.env.reset_wait())

	def _wrap_into_tensor(self, step_result):
		obs, reward, terminated, truncated, info = step_result
		reward = torch.tensor(reward, dtype=torch.float32)
		terminated = torch.tensor(terminated)
		truncated = torch.tensor(truncated)
		done = terminated | truncated
		data = {k: torch.zeros_like(reward) for k in ('success', 'terminated', 'truncated')}
		if 'success' not in info:
			info['success'] = data['success']

		for env_id in range(done.shape[0]):
			if self._wrapped_vectorized and not self.env.is_eval and done[env_id].item():
				final_info = info['final_info'][env_id]
				final_info['success'] = torch.tensor(final_info.get('success', False), dtype=torch.float32)
				final_info['terminated'] = torch.tensor(final_info['terminated'], dtype=torch.float32)
				final_info['truncated'] = torch.tensor(final_info['truncated'], dtype=torch.float32)
				for key in data:
					data[key][env_id] = final_info[key]
			else:
				for key in data:
					data[key][env_id] = torch.tensor(info[key][env_id], dtype=torch.float32)

		info.update(data)

		return self._obs_to_tensor(obs), reward, done, info

	def step_wait(self):
		step_result = self.env.step_wait()
		return self._wrap_into_tensor(step_result)

	def step(self, action, **kwargs):
		if self._wrapped_vectorized:
			step_result = self.env.step(action.numpy(), **kwargs)
		else:
			step_result = self.env.step(action.numpy())

		return self._wrap_into_tensor(step_result)

	def render(self, **kwargs):
		if self._wrapped_vectorized:
			return self.env.render(**kwargs)
		else:
			return self.env.render()
