import gym
import numpy as np
import torch

from envs.wrappers.vectorized import Vectorized
from ocr.tools import SlotExtractor


class SlotVectorized(Vectorized):
	def __init__(self, cfg, env_fn, slot_extractor: SlotExtractor, is_eval=False):
		super().__init__(cfg, env_fn, is_eval)
		self._slot_extractor = slot_extractor
		self._prev_slots = None
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(self.cfg.ocr_frame_stack, *self._slot_extractor.get_slots_dim()), dtype=np.float32
		)
		self._frame_stack = torch.zeros(self.cfg.num_envs, *self.observation_space.shape, dtype=torch.float32)

	def reset_wait(self, env_ids=None):
		if env_ids is None:
			env_ids = range(self.cfg.num_envs)

		obss = super().reset_wait(env_ids)
		slots = self._slot_extractor(obss, None)
		self._prev_slots = slots
		self._frame_stack[env_ids, :, :, :] = slots.unsqueeze(1)

		return self._frame_stack[env_ids].clone()

	def step_wait(self):
		assert self._prev_slots is not None, f'Previous slots {self._prev_slots} are None'
		obss, rews, terms, truncs, infos = super().step_wait()
		if 'final_observation' in infos:
			dones = terms | truncs
			new_episode_obss = obss[dones].clone()
			reset_env_ids = dones.nonzero(as_tuple=True)[0].tolist()
			obss[dones] = torch.stack([infos['final_observation'][env_id] for env_id in reset_env_ids])
			slots = self._slot_extractor(obss, self._prev_slots)
			self._frame_stack = torch.roll(self._frame_stack, shifts=-1, dims=1)
			self._frame_stack[:, -1] = slots
			for env_id in reset_env_ids:
				infos['final_observation'][env_id] = self._frame_stack[env_id].clone()

			slots[dones] = self._slot_extractor(new_episode_obss, None)
			self._frame_stack[dones, :, :, :] = slots[dones].unsqueeze(1)
		else:
			slots = self._slot_extractor(obss, self._prev_slots)
			self._frame_stack = torch.roll(self._frame_stack, shifts=-1, dims=1)
			self._frame_stack[:, -1] = slots

		self._prev_slots = slots

		return self._frame_stack.clone(), rews, terms, truncs, infos
