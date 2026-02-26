import os
import time

import torch
from torch import nn


class InvalidTaskException(Exception):
	def __init__(self, task, suite):
		self.task = task
		self.suite = suite
		super().__init__(f"Cannot find task {self.task} in suite {self.suite}")


class MissingDependencyException(Exception):
	def __init__(self, suite, exception):
		self.suite = suite
		super().__init__(f'Missing dependencies for suite {self.suite}; install dependencies to use this environment. Error: {exception}')


def stop_watch(function, *args, **kwargs):
	start = time.perf_counter()
	result = function(*args, **kwargs)
	return result, time.perf_counter() - start


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def get_act_fn(act_fn):
	if act_fn == 'relu':
		return nn.ReLU()
	elif act_fn == 'leaky_relu':
		return nn.LeakyReLU()
	elif act_fn == 'elu':
		return nn.ELU()
	elif act_fn == 'sigmoid':
		return nn.Sigmoid()
	elif act_fn == 'softplus':
		return nn.Softplus()
	else:
		raise ValueError('Invalid argument for `act_fn`.')


def make_node_mlp_layers(num_layers, input_dim, hidden_dim, output_dim, act_fn, layer_norm):
	layers = []

	for idx in range(num_layers):

		if idx == 0:
			# first layer, input_dim => hidden_dim
			layers.append(nn.Linear(input_dim, hidden_dim))
			layers.append(get_act_fn(act_fn))
		elif idx == num_layers - 2:
			# layer before the last, add layer norm
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			if layer_norm:
				layers.append(nn.LayerNorm(hidden_dim))
			layers.append(get_act_fn(act_fn))
		elif idx == num_layers - 1:
			# last layer, hidden_dim => output_dim and no activation
			layers.append(nn.Linear(hidden_dim, output_dim))
		else:
			# all other layers, hidden_dim => hidden_dim
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(get_act_fn(act_fn))

	return layers


def to_one_hot(indices, max_index):
	"""Get one-hot encoding of index tensors."""
	zeros = torch.zeros(
		indices.size()[0], max_index, dtype=torch.float32,
		device=indices.device)
	return zeros.scatter_(1, indices.unsqueeze(1), 1)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
	"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
	result_shape = (num_segments, tensor.size(1))
	result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
	segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
	result.scatter_add_(0, segment_ids, tensor)
	return result


def concatenate_input_wrapper(module: nn.Module):
	class ConcatWrapper(nn.Module):
		def __init__(self, module: nn.Module):
			super().__init__()
			self.module = module

		def forward(self, inputs):
			if isinstance(inputs, (tuple, list)):
				inputs = torch.cat(inputs, dim=-1)

			return self.module(inputs)

	return ConcatWrapper(module)
