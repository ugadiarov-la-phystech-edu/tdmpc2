import time
import pickle
from pathlib import Path

import yaml
from matplotlib import pyplot

import isaacgym
from tdmpc2.envs.isaac.isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper, IsaacPandaPushWrapper
from tdmpc2.envs.isaac.isaac_panda_push_env import IsaacPandaPush


def create_env(device):
	config = yaml.safe_load(Path(f'config/generalization_sort_push/Config.yaml').read_text())
	isaac_env_cfg = yaml.safe_load(Path(f'config/generalization_sort_push/IsaacPandaPushConfig.yaml').read_text())
	isaac_env_cfg['env']['numObjects'] = 3
	isaac_env_cfg['env']['numColors'] = 3
	isaac_env_cfg['env']['cameraRes'] = 128
	env = IsaacPandaPush(
		cfg=isaac_env_cfg,
		rl_device=device,
		sim_device=device,
		graphics_device_id=0,
		headless=True,
		virtual_screen_capture=False,
		force_render=False,
	)
	env = IsaacPandaPushGoalSB3Wrapper(
		env=env,
		obs_mode='raw',
		n_views=1,
		latent_rep_model=None,
		latent_classifier=None,
		reward_cfg=config['Reward']['GT'],
		smorl=(config['Model']['method'] == 'SMORL'),
	)
	env = IsaacPandaPushWrapper(env)

	return env


def show(img):
	pyplot.imshow(img)
	pyplot.show()


if __name__ == '__main__':
	env = create_env(device='cpu')
	# warm up
	o = env.reset()
	done = False
	from PIL import Image
	# Image.fromarray(env.env.goal[0][0].transpose(1, 2, 0)).save('/tmp/goal.png')

	observations = []
	infos = []
	while not done:
		o, r, done, i = env.step(env.action_space.sample())
		show(o)
		infos.append({k: v for k, v in i.items() if k in ('eff_pos', 'current_cube_pos')})
		observations.append(o)

	for i, (obs, info) in enumerate(zip(observations, infos)):
		pass
		from PIL import Image
		# Image.fromarray(obs).save(f'/tmp/img_{i:03d}.png')
		# with open(f'/tmp/data_{i:03d}.pkl', 'wb') as file_obj:
		#     pickle.dump(info, file_obj)
