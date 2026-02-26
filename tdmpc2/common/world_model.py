from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs = self._build_components(cfg)
		if not cfg.episodic:
			self._termination = None

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	@staticmethod
	def _build_components(cfg):
		if 'ocr_model' in cfg:
			encoder = nn.Identity()
			num_slot = cfg.ocr_config['num_slot']
			slot_size = cfg.ocr_config['slot_size']
			frame_stack = cfg.ocr_frame_stack
			dynamics = OCDynamicsModel(frame_stack, num_slot, slot_size, cfg.mlp_dim, cfg.action_dim)
			reward = OCRewardModel(frame_stack, num_slot, slot_size, cfg.mlp_dim, cfg.action_dim, max(cfg.num_bins, 1))
			termination = OCTermination(frame_stack, num_slot, slot_size, cfg.mlp_dim)
			pi = OCPolicy(frame_stack, num_slot, slot_size, cfg.mlp_dim, cfg.action_dim)
			Qs = layers.Ensemble([OCRewardModel(frame_stack, num_slot, slot_size, cfg.mlp_dim, cfg.action_dim,
												max(cfg.num_bins, 1)) for _ in range(cfg.num_q)])

			for model in encoder, dynamics, reward, termination, pi, Qs:
				model.apply(init.weight_init)

			init.zero_([reward.mlp.weight, Qs.params['mlp', 'weight']])
		else:
			encoder = layers.enc(cfg)
			dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
										cfg.latent_dim, act=layers.SimNorm(cfg))
			dynamics.loss = nn.functional.mse_loss
			reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))
			termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim],1)
			pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
			Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
												   max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
			for model in encoder, dynamics, reward, termination, pi, Qs:
				model.apply(init.weight_init)

			init.zero_([reward.module[-1].weight, Qs.params["module", "2", "weight"]])

		return encoder, dynamics, reward, termination, pi, Qs

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and 'ocr_model' in self.cfg:
			# obs.shape = (num_envs, frame_stack, num_slots, slot_dim) -> (num_envs, num_slots, frame_stack, slot_dim)
			# obs.shape = (num_envs, num_slots, frame_stack, slot_dim) -> (num_envs, num_slots * frame_stack * slot_dim)
			return obs.movedim(source=-3, destination=-2).flatten(start_dim=-3)
		elif self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])

		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._dynamics((z, a))

	def dynamics_loss(self, input_z, target_z):
		return self._dynamics.loss(input_z, target_z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._reward((z, a))
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet((z, a))

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2


class OCDynamicsModel(nn.Module):
	def __init__(self, frame_stack, num_slots, slot_size, hidden_dim, action_dim, use_interactions=True):
		super().__init__()
		self._frame_stack = frame_stack
		self._slot_size = slot_size
		input_dim = self._frame_stack * self._slot_size
		self.gnn = layers.GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=self._slot_size,
							  action_dim=action_dim, num_objects=num_slots, ignore_action=False,
							  copy_action=True, edge_actions=True, use_interactions=use_interactions)

	def forward(self, slots_action):
		# predicts the next state and makes frame stack using values from the previous step
		slots, action = slots_action
		batch_shape = slots.shape[:-1]
		slots = slots.reshape(*batch_shape, self.gnn.num_objects, self.gnn.input_dim)
		output = self.gnn(slots.flatten(end_dim=-3), action.flatten(end_dim=-2))
		output = output.unflatten(dim=0, sizes=batch_shape)
		slots = slots.unflatten(dim=-1, sizes=(self._frame_stack, self._slot_size))
		output = torch.cat((slots[..., 1:, :], output.unsqueeze(-2)), dim=-2)
		return output.reshape(*batch_shape, -1)

	def loss(self, input_slots, target_slots):
		# ignore old frames from stack when computing loss
		input_slots = input_slots.unflatten(dim=-1, sizes=(self.gnn.num_objects, self._frame_stack, self._slot_size))
		target_slots = target_slots.unflatten(dim=-1, sizes=(self.gnn.num_objects, self._frame_stack, self._slot_size))
		return nn.functional.mse_loss(input_slots[..., -1, :], target_slots[..., -1, :])


class OCRewardModel(nn.Module):
	def __init__(self, frame_stack, num_slots, slot_size, hidden_dim, action_dim, num_bins, use_interactions=True):
		super().__init__()
		self.act = nn.ReLU(inplace=True)
		input_dim = frame_stack * slot_size
		self.gnn = layers.GNN(input_dim=input_dim, hidden_dim=hidden_dim,
							  action_dim=action_dim, num_objects=num_slots + 1, ignore_action=False,
							  copy_action=True, edge_actions=True, use_interactions=use_interactions)
		self.learnable_embedding = nn.Parameter(torch.randn(1, input_dim))
		self.mlp = nn.Linear(in_features=input_dim, out_features=max(num_bins, 1))
		with torch.no_grad():
			limit = (6.0 / (1 + input_dim)) ** 0.5
			torch.nn.init.uniform_(self.learnable_embedding, -limit, limit)

	def forward(self, slots_action):
		slots, action = slots_action
		batch_shape = slots.shape[:-1]
		slots = slots.reshape(*batch_shape, self.gnn.num_objects - 1, self.gnn.input_dim)
		embedding = self.learnable_embedding.reshape((1,) * len(batch_shape) + self.learnable_embedding.shape)
		slots = torch.cat([slots, embedding.expand((*batch_shape, -1, -1))], dim=-2)
		x = self.gnn(slots.flatten(end_dim=-3), action.flatten(end_dim=-2))[:, -1]
		x = x.reshape(*batch_shape, -1)
		return self.mlp(x)


class OCTermination(nn.Module):
	def __init__(self, frame_stack, num_slots, slot_size, hidden_dim, use_interactions=True):
		super().__init__()
		self.act = nn.ReLU(inplace=True)
		input_dim = frame_stack * slot_size
		self.gnn = layers.GNN(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=0,
							  num_objects=num_slots + 1, ignore_action=True, copy_action=False, edge_actions=False,
							  use_interactions=use_interactions)
		self.learnable_embedding = nn.Parameter(torch.randn(1, input_dim))
		self.mlp = nn.Linear(in_features=input_dim, out_features=1)
		with torch.no_grad():
			limit = (6.0 / (1 + input_dim)) ** 0.5
			torch.nn.init.uniform_(self.learnable_embedding, -limit, limit)

	def forward(self, slots):
		batch_shape = slots.shape[:-1]
		slots = slots.reshape(*batch_shape, self.gnn.num_objects - 1, self.gnn.input_dim)
		embedding = self.learnable_embedding.reshape((1,) * len(batch_shape) + self.learnable_embedding.shape)
		slots = torch.cat([slots, embedding.expand((*batch_shape, -1, -1))], dim=-2)
		x = self.gnn(slots.flatten(end_dim=-3), action=None)[:, -1]
		x = x.reshape(*batch_shape, -1)
		return self.mlp(x)


class OCPolicy(nn.Module):
	def __init__(self, frame_stack, num_slots, slot_size, hidden_dim, action_dim, use_interactions=True):
		super().__init__()
		self.act = nn.ReLU(inplace=True)
		input_dim = frame_stack * slot_size
		self.gnn = layers.GNN(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=0,
							  num_objects=num_slots + 1, ignore_action=True, copy_action=False, edge_actions=False,
							  use_interactions=use_interactions)
		self.learnable_embedding = nn.Parameter(torch.randn(1, input_dim))
		self.mlp = nn.Linear(in_features=input_dim, out_features=2 * action_dim)
		with torch.no_grad():
			limit = (6.0 / (1 + input_dim)) ** 0.5
			torch.nn.init.uniform_(self.learnable_embedding, -limit, limit)

	def forward(self, slots):
		batch_shape = slots.shape[:-1]
		slots = slots.reshape(*batch_shape, self.gnn.num_objects - 1, self.gnn.input_dim)
		embedding = self.learnable_embedding.reshape((1,) * len(batch_shape) + self.learnable_embedding.shape)
		slots = torch.cat([slots, embedding.expand((*batch_shape, -1, -1))], dim=-2)
		x = self.gnn(slots.flatten(end_dim=-3), action=None)[:, -1]
		x = x.reshape(*batch_shape, -1)
		return self.mlp(x)
