from copy import deepcopy
import warnings

import gymnasium as gym

from common.utils import InvalidTaskException, MissingDependencyException
# from envs.wrappers.multitask import MultitaskWrapper
# from envs.wrappers.slot_vectorized import SlotVectorized
# from envs.wrappers.tensor import TensorWrapper
# from envs.wrappers.vectorized import Vectorized
# from ocr.tools import build_ocr_model, SlotExtractor


def missing_dependencies(suite, exception):
	def _cast_missing_dependencies_exception(*args, **kwargs):
		raise MissingDependencyException(suite, exception)

	return _cast_missing_dependencies_exception

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except Exception as e:
	make_dm_control_env = missing_dependencies('dmcontrol', e)
try:
	from envs.maniskill import make_env as make_maniskill_env
except Exception as e:
	make_maniskill_env = missing_dependencies('maniskill', e)
try:
	from envs.metaworld import make_env as make_metaworld_env
except Exception as e:
	make_metaworld_env = missing_dependencies('metaworld', e)
try:
	from envs.myosuite import make_env as make_myosuite_env
except Exception as e:
	make_myosuite_env = missing_dependencies('myosuite', e)
try:
	from envs.mujoco import make_env as make_mujoco_env
except Exception as e:
	make_mujoco_env = missing_dependencies('mujoco', e)
try:
	from envs.robosuite_env import make_env as make_robosuite_env
except Exception as e:
	make_robosuite_env = missing_dependencies('robosuite_env', e)
try:
	from envs.maniskill3 import make_env as make_maniskill3_env
except Exception as e:
	make_maniskill3_env = missing_dependencies('maniskill3', e)
try:
	from envs.cw_envs.target import make_env as make_causalworld_env
except Exception as e:
	make_causalworld_env = missing_dependencies('causalworld', e)


warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env


def make_env(cfg, is_eval=False):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)
	else:
		env = None
		for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env,
				   make_robosuite_env, make_maniskill3_env, make_causalworld_env]:
			try:
				env = fn(cfg)
				break
			except InvalidTaskException as e:
				print(e)
			except MissingDependencyException as e:
				print(e)
		if env is None:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')

		if 'ocr_model' in cfg and cfg.ocr_model is not None:
			ocr_model = build_ocr_model(cfg)
			slot_extractor = SlotExtractor(model=ocr_model, device=cfg.ocr_device)
			env = SlotVectorized(cfg, fn, slot_extractor, is_eval=False)
		else:
			env = Vectorized(cfg, fn, is_eval)

		env = TensorWrapper(env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = cfg.get("seed_steps", max(1000, 5*cfg.episode_length) * cfg.num_envs)
	return env
