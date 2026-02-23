import os
from collections import defaultdict
from time import time

import torch
from tensordict.tensordict import TensorDict

from common.utils import stop_watch
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._resume = False
		if self.cfg.get('checkpoint', None):
			path = os.path.join(self.cfg.checkpoint, 'checkpoint.pt')
			print(f'Loading checkpoint: {path}')
			state_dict = torch.load(path)
			self.agent.load(state_dict)
			self._step = state_dict['step']
			self._ep_idx = state_dict['episode']
			self.buffer.num_eps = state_dict['episode']
			self._start_time = time() - state_dict['total_time']
			self._resume = True

		self._prev_step = self._step
		self._prev_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		tm = time()
		steps_per_second = (self._step - self._prev_step) / (tm - self._prev_time)
		self._prev_step = self._step
		self._prev_time = tm
		total_time = tm - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=total_time,
			steps_per_second=steps_per_second,
			mean_steps_per_second=self._step / total_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		current_episode_steps = torch.zeros(self.cfg.num_envs, dtype=torch.long)
		current_episode_rewards = torch.zeros(self.cfg.num_envs, dtype=torch.float)
		n_episodes = torch.zeros(self.cfg.num_envs, dtype=torch.long)
		episode_videos = [[] for _ in range(self.cfg.num_envs)]

		# call reset() just to get obs of correct dimension
		obs = self.eval_env.reset()
		done = torch.ones(self.cfg.num_envs, dtype=torch.bool)
		first_step = True
		log_video_env_ids = set(list(range(self.cfg.num_videos)))

		def _need_render():
			return self.cfg.save_video and torch.any(n_episodes[:self.cfg.num_videos] == 0).item()

		while n_episodes.sum() < self.cfg.eval_episodes or _need_render():
			if done.any().item():
				if not first_step:
					ep_rewards.extend(current_episode_rewards[done].tolist())
					current_episode_rewards[done] = 0.0
					ep_successes.extend(info['success'][done].tolist())
					ep_lengths.extend(current_episode_steps[done].tolist())
					current_episode_steps[done] = 0
					n_episodes[done] += 1

				env_ids = done.nonzero(as_tuple=True)[0].tolist()
				obs[done] = self.eval_env.reset(env_ids=env_ids)
				if _need_render():
					render_env_ids = list(log_video_env_ids.intersection(env_ids))
					images = self.eval_env.render(env_ids=render_env_ids)
					for env_id, image in zip(render_env_ids, images):
						episode_videos[env_id].append([image])

			first_step = False
			torch.compiler.cudagraph_mark_step_begin()
			action = self.agent.act(obs, t0=done.to(self.agent.device), eval_mode=True)
			obs, reward, done, info = self.eval_env.step(action)
			current_episode_rewards += reward
			current_episode_steps += 1

			if _need_render():
				render_env_ids = list(log_video_env_ids)
				images = self.eval_env.render(env_ids=render_env_ids)
				for env_id, image in zip(render_env_ids, images):
					episode_videos[env_id][-1].append(image)

		if self.cfg.save_video:
			for env_id in log_video_env_ids:
				self.logger.log_video(episode_videos[env_id][0], f"env-{env_id}", self._step)

		episode_rewards = ep_rewards[:self.cfg.eval_episodes]
		episode_successes = ep_successes[:self.cfg.eval_episodes]
		episode_lengths = ep_lengths[:self.cfg.eval_episodes]
		return dict(
			episode_reward=torch.tensor(episode_rewards, dtype=torch.float32).mean().item(),
			episode_success=torch.tensor(episode_successes, dtype=torch.float32).mean().item(),
			episode_length= torch.tensor(episode_lengths, dtype=torch.float32).mean().item(),
			episode_rewards=episode_rewards,
			episode_successes=episode_successes,
			episode_lengths=episode_lengths,
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act()[0], float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def _obs_buffer(self, obs):
		if self.cfg.obs == 'rgb':
			return obs[-3:]

		return obs

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done = {}, torch.ones(self.cfg.num_envs, dtype=torch.bool)
		profiling_statistics = defaultdict(list)
		self._tds = [None for _ in range(self.cfg.num_envs)]
		first_step = True
		do_pretrain = self._step < self.cfg.seed_steps
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if not (self._resume and first_step) and self.cfg.eval_freq > 0 and self._step % self.cfg.eval_freq == 0:
				eval_metrics = self.eval()
				eval_metrics.update(self.common_metrics())
				self.logger.log(eval_metrics, 'eval', flush=True)

			if not first_step and self.cfg.save_freq > 0 and self._step % self.cfg.save_freq == 0:
				self.logger.save_agent(self.agent, statistics=self.common_metrics(), identifier='checkpoint', buffer=self.buffer)

			# Reset environment
			if done.any().item():
				env_ids = done.nonzero(as_tuple=True)[0].tolist()
				reset_obs = self.env.reset(env_ids=env_ids)
				if first_step:
					assert done.all().item()
					obs = reset_obs
				else:
					if self.cfg.check_termination and not self.cfg.episodic and info['terminated'].any().item():
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					episode_rewards, episode_successes, episode_lengths, episode_terminations = [], [], [], []
					for env_id in env_ids:
						tds = torch.cat(self._tds[env_id])
						episode_rewards.append(tds['reward'].nansum(0).item())
						episode_successes.append(info['success'][env_id].nanmean().item())
						episode_lengths.append(len(self._tds[env_id]))
						episode_terminations.append(info['terminated'][env_id].nanmean().item())
						# Do not add too short trajectories
						if len(tds) > self.cfg.horizon:
							self._ep_idx, buffer_add_time = stop_watch(self.buffer.add, tds)
							profiling_statistics['buffer_add_time'].append(buffer_add_time)

					train_metrics.update(
						episode_rewards=episode_rewards,
						episode_successes=episode_successes,
						episode_lengths=episode_lengths,
						episode_terminations=episode_terminations,
						episode_reward=torch.tensor(episode_rewards, dtype=torch.float32).mean().item(),
						episode_success=torch.tensor(episode_successes, dtype=torch.float32).mean().item(),
						episode_length=torch.tensor(episode_lengths, dtype=torch.float32).mean().item(),
						episode_terminated=torch.tensor(episode_terminations, dtype=torch.float32).mean().item(),
					)
					train_metrics.update(self.common_metrics())
					train_metrics.update({k: sum(v) / len(v) for k, v in profiling_statistics.items()})
					self.logger.log(train_metrics, 'train')
					train_metrics = {}
					profiling_statistics = defaultdict(list)
					obs[done] = reset_obs

				for env_id in env_ids:
					self._tds[env_id] = [self.to_td(self._obs_buffer(obs[env_id]))] * self.cfg.frame_stack

				if first_step:
					first_step = False
					self.buffer.init(self._tds[0])

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action, act_time = stop_watch(self.agent.act, obs, t0=done.to(self.agent.device))
				profiling_statistics['act_time'].append(act_time)
			else:
				action = self.env.rand_act()
			(obs, reward, done, info), step_time = stop_watch(self.env.step, action)
			profiling_statistics['env_step_time'].append(step_time)
			for env_id in range(self.cfg.num_envs):
				self._tds[env_id].append(self.to_td(self._obs_buffer(obs[env_id]), action[env_id], reward[env_id], info['terminated'][env_id]))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if do_pretrain:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...', flush=True)
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				for i in range(1, num_updates + 1):
					_train_metrics, update_time = stop_watch(self.agent.update, self.buffer)
					_train_metrics = {k: v.item() for k, v in _train_metrics.items()}
					buffer_sample_time = _train_metrics.pop('buffer_sample_time')
					profiling_statistics['update_time'].append(update_time - buffer_sample_time)
					profiling_statistics['buffer_sample_time'].append(buffer_sample_time)
					if do_pretrain and i % self.cfg.log_every_pretraining == 0:
						train_metrics.update(_train_metrics)
						train_metrics.update(self.common_metrics())
						train_metrics.update({k: sum(v) / len(v) for k, v in profiling_statistics.items()})
						self.logger.log(train_metrics, 'train')
						train_metrics = {}
						profiling_statistics = defaultdict(list)
						
				train_metrics.update(_train_metrics)
				if do_pretrain:
					print('Pretraining complete.', flush=True)
					do_pretrain = False

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent, statistics=self.common_metrics(), identifier='checkpoint', buffer=self.buffer)
