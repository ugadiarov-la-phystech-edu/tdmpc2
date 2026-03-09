import argparse
import json
import os
import time
from pprint import pprint
from types import SimpleNamespace

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

import torch
from termcolor import colored

from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True


def log_video(videos_dir, images, tag, fps):
	clip = ImageSequenceClip(images, fps)
	path = os.path.join(videos_dir, f"{tag}.mp4")
	clip.write_videofile(path)


def evaluate(cfg):
	"""
	Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task evaluation)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`checkpoint`: path to model checkpoint to load
		`eval_episodes`: number of episodes to evaluate on per task (default: 10)
		`save_video`: whether to save a video of the evaluation (default: True)
		`seed`: random seed (default: 1)
	
	See config.yaml for a full list of args.

	Example usage:
	````
		$ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
		$ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
		$ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	# cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	if not cfg.multitask and ('mt80' in cfg.checkpoint or 'mt30' in cfg.checkpoint):
		print(colored('Warning: single-task evaluation of multi-task models is not currently supported.', 'red', attrs=['bold']))
		print(colored('To evaluate a multi-task model, use task=mt80 or task=mt30.', 'red', attrs=['bold']))

	start = time.perf_counter()

	# Make environment
	eval_env = make_env(cfg, is_eval=True)

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	
	# Evaluate
	if cfg.multitask:
		print(colored(f'Evaluating agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating agent on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)

	ep_rewards, ep_successes, ep_lengths = [], [], []
	current_episode_steps = torch.zeros(cfg.num_envs, dtype=torch.long)
	current_episode_rewards = torch.zeros(cfg.num_envs, dtype=torch.float)
	n_episodes = torch.zeros(cfg.num_envs, dtype=torch.long)
	episode_videos = [[] for _ in range(cfg.num_envs)]

	# call reset() just to get obs of correct dimension
	obs = eval_env.reset()
	done = torch.ones(cfg.num_envs, dtype=torch.bool)
	first_step = True
	log_video_env_ids = set(list(range(cfg.num_videos)))

	def _need_render():
		return cfg.save_video and torch.any(n_episodes[:cfg.num_videos] == 0).item()

	while n_episodes.sum() < cfg.eval_episodes or _need_render():
		if done.any().item():
			if not first_step:
				ep_rewards.extend(current_episode_rewards[done].tolist())
				current_episode_rewards[done] = 0.0
				ep_successes.extend(info['success'][done].tolist())
				ep_lengths.extend(current_episode_steps[done].tolist())
				current_episode_steps[done] = 0
				n_episodes[done] += 1

			env_ids = done.nonzero(as_tuple=True)[0].tolist()
			obs[done] = eval_env.reset(env_ids=env_ids)
			if _need_render():
				render_env_ids = list(log_video_env_ids.intersection(env_ids))
				images = eval_env.render(env_ids=render_env_ids)
				for env_id, image in zip(render_env_ids, images):
					episode_videos[env_id].append([image])

		first_step = False
		torch.compiler.cudagraph_mark_step_begin()
		action = agent.act(obs, t0=done.to(agent.device), eval_mode=True)
		obs, reward, done, info = eval_env.step(action)
		current_episode_rewards += reward
		current_episode_steps += 1

		if _need_render():
			render_env_ids = list(log_video_env_ids)
			images = eval_env.render(env_ids=render_env_ids)
			for env_id, image in zip(render_env_ids, images):
				episode_videos[env_id][-1].append(image)

	print('Time elapsed:', time.perf_counter() - start)
	if cfg.save_video:
		for env_id in log_video_env_ids:
			log_video(video_dir, episode_videos[env_id][0], f"env-{env_id}", cfg.fps)

	episode_rewards = ep_rewards[:cfg.eval_episodes]
	episode_successes = ep_successes[:cfg.eval_episodes]
	episode_lengths = ep_lengths[:cfg.eval_episodes]
	results = dict(
		episode_reward=torch.tensor(episode_rewards, dtype=torch.float32).mean().item(),
		episode_success=torch.tensor(episode_successes, dtype=torch.float32).mean().item(),
		episode_length=torch.tensor(episode_lengths, dtype=torch.float32).mean().item(),
		episode_rewards=episode_rewards,
		episode_successes=episode_successes,
		episode_lengths=episode_lengths,
	)
	pprint(results)
	with open(os.path.join(cfg.work_dir, 'results.txt'), 'w') as f:
		pprint(results, stream=f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', type=str, required=True)
	parser.add_argument('--work_dir', type=str, required=True)
	parser.add_argument('--task', type=str, required=True)
	args = parser.parse_args()
	with open(args.config_path, 'r') as f:
		cfg_dict = json.load(f)

	cfg_dict['work_dir'] = args.work_dir
	cfg_dict['compile_plan'] = True
	cfg_dict['num_envs'] = 30
	cfg_dict['task'] = args.task
	cfg_dict['num_videos'] = 30
	cfg = SimpleNamespace(**cfg_dict)
	cfg.get = cfg_dict.get
	os.makedirs(cfg.work_dir, exist_ok=True)
	evaluate(cfg)
