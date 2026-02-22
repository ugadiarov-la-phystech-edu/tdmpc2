import collections
import functools
import time
from copy import deepcopy

import cloudpickle
import portal
import numpy as np
import torch


class Vectorized:
	"""
	Vectorized environment for TD-MPC2 online training.
	"""
	def __init__(self, cfg, env_fn, is_eval=False):
		self.cfg = cfg
		self.parallel = cfg.parallel
		self.is_eval = is_eval
		self.backend = None
		self._need_to_wait = False
		self._step_results = None
		self._reset_results = None

		def _make(rank):
			_cfg = deepcopy(cfg)
			_cfg.num_envs = 1
			_cfg.seed = cfg.seed + rank
			return env_fn(_cfg, autoreset=not is_eval)

		fns = [functools.partial(_make, i) for i in range(int(is_eval) * self.cfg.num_envs, (int(is_eval) + 1) * self.cfg.num_envs)]
		if self.parallel:
			import multiprocessing as mp
			context = mp.get_context()
			self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.cfg.num_envs)])
			self.stop = context.Event()
			fns = [cloudpickle.dumps(fn) for fn in fns]
			self.procs = [
				portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
				for i, (fn, pipe) in enumerate(zip(fns, pipes))]
			self.pipes[0].send(('action_space',))
			self.action_space = self._receive(self.pipes[0])
			self.pipes[0].send(('observation_space',))
			self.observation_space = self._receive(self.pipes[0])
			self.pipes[0].send(('max_episode_steps',))
			self.max_episode_steps = self._receive(self.pipes[0])
		else:
			self.envs = [fn() for fn in fns]
			self.action_space = self.envs[0].action_space
			self.observation_space = self.envs[0].observation_space
			self.max_episode_steps = self.envs[0].max_episode_steps

	def _receive(self, pipe):
		try:
			msg, arg = pipe.recv()
			if msg == 'error':
				raise RuntimeError(arg)
			assert msg == 'result'
			return arg
		except Exception:
			print('Terminating workers due to an exception.')
			[proc.kill() for proc in self.procs]
			raise

	@staticmethod
	def _env_server(stop, envid, pipe, ctor):
		try:
			ctor = cloudpickle.loads(ctor)
			env = ctor()
			while not stop.is_set():
				if not pipe.poll(0.1):
					time.sleep(0.1)
					continue
				try:
					msg, *args = pipe.recv()
				except EOFError:
					return
				if msg == 'step':
					assert len(args) == 1
					act = args[0]
					step_result = env.step(act)
					pipe.send(('result', step_result))
				elif msg == 'reset':
					assert len(args) == 0
					reset_result = env.reset()
					pipe.send(('result', reset_result))
				elif msg == 'render':
					assert len(args) == 0
					image = env.render()
					pipe.send(('result', image))
				elif msg == 'observation_space':
					assert len(args) == 0
					pipe.send(('result', env.observation_space))
				elif msg == 'action_space':
					assert len(args) == 0
					pipe.send(('result', env.action_space))
				elif msg == 'max_episode_steps':
					assert len(args) == 0
					pipe.send(('result', env.max_episode_steps))
				else:
					raise ValueError(f'Invalid message {msg}')
		except ConnectionResetError:
			print('Connection to driver lost')
		except Exception as e:
			pipe.send(('error', e))
			raise
		finally:
			try:
				env.close()
			except Exception:
				pass
			pipe.close()

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1

	def _set_backend(self, data):
		if self.backend is None:
			if isinstance(data, np.ndarray):
				self.backend = np
			elif isinstance(data, torch.Tensor):
				self.backend = torch
			else:
				raise ValueError('Invalid type:', type(data))

	def _stack_obs(self, observations):
		self._set_backend(observations[0])
		return self.backend.stack(observations)

	def step(self, acts):
		self.step_async(acts)
		return self.step_wait()

	def step_async(self, acts):
		if self._need_to_wait:
			raise ValueError('Asynchronous step is being executed. Wait for results using step_wait().')

		self._need_to_wait = True
		if self.parallel:
			[pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
		else:
			self._step_results = [env.step(act) for env, act in zip(self.envs, acts)]

	@staticmethod
	def _merge_dicts(dicts):
		result = collections.defaultdict(list)
		all_keys = set()

		for d in dicts:
			all_keys.update(d.keys())

		for key in all_keys:
			for d in dicts:
				result[key].append(d.get(key, None))

		return result

	def step_wait(self):
		if not self._need_to_wait:
			raise ValueError('Must call step_wait() after calling step_async()')

		self._need_to_wait = False
		if self.parallel:
			step_results = [self._receive(pipe) for pipe in self.pipes]
		else:
			step_results = self._step_results
			self._step_results = None

		obss, rews, terms, truncs, infos = zip(*step_results)
		infos = self._merge_dicts(infos)
		return self._stack_obs(obss), np.stack(rews), np.stack(terms), np.stack(truncs), infos

	def reset(self, env_ids=None):
		self.reset_async(env_ids)
		return self.reset_wait(env_ids)

	def reset_async(self, env_ids=None):
		if self._need_to_wait:
			raise ValueError('Asynchronous reset is being executed. Wait for results using reset_wait().')

		self._need_to_wait = True
		if env_ids is None:
			env_ids = range(self.cfg.num_envs)

		if self.parallel:
			[self.pipes[i].send(('reset',)) for i in env_ids]
		else:
			self._reset_results = [self.envs[i].reset() for i in env_ids]

	def reset_wait(self, env_ids=None):
		if not self._need_to_wait:
			raise ValueError('Must call reset_wait() after calling reset_async()')

		self._need_to_wait = False
		if env_ids is None:
			env_ids = range(self.cfg.num_envs)

		if self.parallel:
			reset_results = [self._receive(self.pipes[i]) for i in env_ids]
		else:
			reset_results = self._reset_results
			self._reset_results = None

		if isinstance(reset_results[0], tuple):
			obss, infos = zip(*reset_results)
		else:
			obss = reset_results
		return self._stack_obs(obss)

	def render(self, env_ids=None):
		if env_ids is None:
			env_ids = range(self.cfg.num_envs)

		if self.parallel:
			[self.pipes[i].send(('render',)) for i in env_ids]
			images = [self._receive(self.pipes[i]) for i in env_ids]
		else:
			images = [self.envs[i].render() for i in env_ids]

		return images

	def close(self):
		if self.parallel:
			[proc.kill() for proc in self.procs]
		else:
			[env.close() for env in self.envs]
