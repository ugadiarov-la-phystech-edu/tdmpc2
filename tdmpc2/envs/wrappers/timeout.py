import gymnasium as gym


class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		if truncated:
			raise ValueError(f"Time out must happen only inside Timeout wrapper, but received truncated={truncated} after t={self._t} steps")
		self._t += 1
		truncated = self._t >= self.max_episode_steps
		info['terminated'] = terminated
		info['truncated'] = truncated
		return obs, reward, terminated, truncated, info
