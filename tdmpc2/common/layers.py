import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy

from common import utils
from common.utils import concatenate_input_wrapper


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	model = nn.Sequential(*mlp)
	model = concatenate_input_wrapper(model)
	return model


def get_out_size(input_size, kernel_sizes, strides):
	assert len(kernel_sizes) == len(strides)
	size = input_size
	for kernel_size, stride in zip(kernel_sizes, strides):
		size = math.floor((size - kernel_size) / stride + 1)

	return size


def conv(in_shape, num_channels, latent_dim, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	kernel_sizes = [7, 5, 3, 3]
	strides = [2, 2, 2, 1]
	layers = [ShiftAug(), PixelPreprocess()]
	size = in_shape[0]
	for kernel_size, stride in zip(kernel_sizes, strides):
		layers.append(nn.Conv2d(size, num_channels, kernel_size, stride=stride))
		layers.append(nn.ReLU(inplace=True))
		size = num_channels

	out_size = get_out_size(in_shape[-1], kernel_sizes, strides)
	out_features = out_size * out_size * num_channels
	if out_features == latent_dim:
		layers = layers[:-1]
		layers.append(nn.Flatten())
	else:
		layers.append(nn.Flatten())
		layers.append(nn.Linear(out_features, latent_dim))

	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			out[k] = conv(cfg.obs_shape[k], cfg.num_channels, cfg.latent_dim, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict


class GNN(torch.nn.Module):

	def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, copy_action=False,
				 act_fn='relu', layer_norm=True, num_layers=3, use_interactions=True, edge_actions=False,
				 output_dim=None):
		super(GNN, self).__init__()

		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		if self.output_dim is None:
			self.output_dim = self.input_dim

		self.num_objects = num_objects
		self.ignore_action = ignore_action
		self.copy_action = copy_action
		self.use_interactions = use_interactions
		self.edge_actions = edge_actions
		self.num_layers = num_layers

		if self.ignore_action:
			self.action_dim = 0
		else:
			self.action_dim = action_dim

		tmp_action_dim = self.action_dim
		edge_mlp_input_size = self.input_dim * 2 + int(self.edge_actions) * tmp_action_dim

		if self.use_interactions:
			self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
				edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
			))

		if self.num_objects == 1 or not self.use_interactions:
			node_input_dim = self.input_dim + tmp_action_dim
		else:
			node_input_dim = hidden_dim + self.input_dim + tmp_action_dim

		self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
			node_input_dim, self.output_dim, act_fn, layer_norm
		))

		self.edge_list = None
		self.batch_size = 0

	def _edge_model(self, source, target, action=None):
		if action is None:
			x = [source, target]
		else:
			x = [source, target, action]

		out = torch.cat(x, dim=1)
		return self.edge_mlp(out)

	def _node_model(self, node_attr, edge_index, edge_attr):
		if edge_attr is not None:
			row, col = edge_index
			agg = utils.unsorted_segment_sum(
				edge_attr, row, num_segments=node_attr.size(0))
			out = torch.cat([node_attr, agg], dim=1)
		else:
			out = node_attr
		return self.node_mlp(out)

	def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
		# Only re-evaluate if necessary (e.g. if batch size changed).
		if self.edge_list is None or self.batch_size != batch_size:
			self.batch_size = batch_size

			# Create fully-connected adjacency matrix for single sample.
			adj_full = torch.ones(num_objects, num_objects)

			# Remove diagonal.
			adj_full -= torch.eye(num_objects)
			self.edge_list = adj_full.nonzero()

			# Copy `batch_size` times and add offset.
			self.edge_list = self.edge_list.repeat(batch_size, 1)
			offset = torch.arange(
				0, batch_size * num_objects, num_objects).unsqueeze(-1)
			offset = offset.expand(batch_size, num_objects * (num_objects - 1))
			offset = offset.contiguous().view(-1)
			self.edge_list += offset.unsqueeze(-1)

			# Transpose to COO format -> Shape: [2, num_edges].
			self.edge_list = self.edge_list.transpose(0, 1)
			self.edge_list = self.edge_list.to(device)

		return self.edge_list

	def process_action_(self, action):
		if self.copy_action:
			if len(action.shape) == 1:
				# action is an integer
				action_vec = utils.to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
			else:
				# action is a vector
				action_vec = action.repeat(1, self.num_objects)

			# mix node and batch dimension
			action_vec = action_vec.reshape(-1, self.action_dim).float()
		else:
			# we have a separate action for each node
			if len(action.shape) == 1:
				# index for both object and action
				action_vec = utils.to_one_hot(action, self.action_dim * self.num_objects)
				action_vec = action_vec.reshape(-1, self.action_dim)
			else:
				action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
				action_vec = action_vec.reshape(-1, self.action_dim)

		return action_vec

	def forward(self, states, action):

		device = states.device
		batch_size = states.size(0)
		num_nodes = states.size(1)

		# states: [batch_size (B), num_objects, embedding_dim]
		# node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
		node_attr = states.reshape(-1, self.input_dim)

		action_vec = None
		if not self.ignore_action:
			action_vec = self.process_action_(action)

		edge_attr = None
		edge_index = None

		if num_nodes > 1 and self.use_interactions:
			# edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
			edge_index = self._get_edge_list_fully_connected(
				batch_size, num_nodes, device)

			row, col = edge_index
			edge_attr = self._edge_model(node_attr[row], node_attr[col], action_vec[row] if self.edge_actions else None)

		if not self.ignore_action:
			# Attach action to each state
			node_attr = torch.cat([node_attr, action_vec], dim=-1)

		node_attr = self._node_model(
			node_attr, edge_index, edge_attr)

		# [batch_size, num_nodes, hidden_dim]
		node_attr = node_attr.view(batch_size, num_nodes, -1)

		return node_attr

	def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):
		return utils.make_node_mlp_layers(self.num_layers, input_dim, self.hidden_dim, output_dim, act_fn, layer_norm)
