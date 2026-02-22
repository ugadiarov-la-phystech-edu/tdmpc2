import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs.transforms import Transform

from common.utils import make_dir


STORAGE = {'lazy_tensor': LazyTensorStorage, 'lazy_memmap': LazyMemmapStorage}


class BatchTransform(Transform):
	def __init__(self, device, batch_size, segment_length, frame_stack):
		super(BatchTransform, self).__init__()
		self._device = device
		self._batch_size = batch_size
		self._segment_length = segment_length
		self._frame_stack = frame_stack

	def _get_batch_view(self, tensor):
		return tensor.view(self._batch_size, self._segment_length, *tensor.shape[1:])

	def _cut_prefix(self, tensor):
		return self._get_batch_view(tensor)[:, self._frame_stack - 1:].flatten(end_dim=1)

	def forward(self, td: TensorDict) -> TensorDict:
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		batch_view = self._get_batch_view(td.get('obs'))
		stacked_batch_view = batch_view.unfold(dimension=1, size=self._frame_stack, step=1).movedim(-1, 2).flatten(start_dim=2, end_dim=3)
		stacked_obs = stacked_batch_view.flatten(end_dim=1).contiguous()
		action = self._cut_prefix(td.get('action')).contiguous()
		reward = self._cut_prefix(td.get('reward').unsqueeze(-1)).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = self._cut_prefix(td.get('terminated').unsqueeze(-1)).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return TensorDict(
			obs=stacked_obs, action=action, reward=reward, terminated=terminated, task=task, batch_size=stacked_obs.shape[0]
		)


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(self.cfg.device)
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self.segment_length = cfg.horizon + 1
		if self.cfg.obs == 'rgb':
			self.segment_length += self.cfg.frame_stack - 1

		self._batch_size = cfg.batch_size * self.segment_length
		self._num_eps = 0
		self._buffer = None
		self._buffer_dir = make_dir(cfg.buffer_dir)
		self._buffer_path = self._buffer_dir / 'checkpoint.buf'
		self._buffer_state_dict_path = self._buffer_dir / 'replay_buffer.pt'
		if self.cfg.buffer_storage_type not in STORAGE:
			raise ValueError(f'Unsupported storage type: {self.cfg.buffer_storage_type}')

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	@num_eps.setter
	def num_eps(self, num_eps):
		self._num_eps = num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=self.cfg.num_envs,
			batch_size=self._batch_size,
			transform=BatchTransform(self._device, self.cfg.batch_size, self.segment_length, self.cfg.frame_stack),
		)

	def init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		assert self._buffer is None, "Buffer is already initialized."
		print(f'Buffer capacity: {self._capacity:,}')
		storage_device = self.cfg.get('buffer_storage_device', None)
		if storage_device is None:
			mem_free, _ = torch.cuda.mem_get_info(self._device)
			bytes_per_step = sum([
				(v.numel() * v.element_size() if not isinstance(v, TensorDict) \
					 else sum([x.numel() * x.element_size() for x in v.values()])) \
				for v in tds.values()
			]) / len(tds)
			total_bytes = bytes_per_step * self._capacity
			print(f'Storage required: {total_bytes / 1e9:.2f} GB')
			# Heuristic: decide whether to use CUDA or CPU memory
			storage_device = self._device if 2.5 * total_bytes < mem_free else 'cpu'

		print(f'Using {storage_device.upper()} memory for storage.')
		cls = STORAGE[self.cfg.buffer_storage_type]
		kwargs = dict(max_size=self._capacity, device=torch.device(storage_device))
		if self.cfg.buffer_storage_type == 'lazy_memmap':
			kwargs.update(dict(existsok=True, scratch_dir=self._buffer_path))

		if self._buffer_path.exists():
			print(f'Loading buffer data from from {self._buffer_path}', flush=True)

		storage = cls(**kwargs)
		buffer = self._reserve_buffer(storage)
		if self.cfg.buffer_storage_type == 'lazy_tensor' and self._buffer_path.exists():
			buffer.loads(self._buffer_path)
		elif self.cfg.buffer_storage_type == 'lazy_memmap' and self._buffer_state_dict_path.exists():
			print(f'Loading buffer state dict from from {self._buffer_state_dict_path}', flush=True)
			buffer.load_state_dict(torch.load(self._buffer_state_dict_path, weights_only=False))

		self._buffer = buffer

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		d = td.to_dict()
		return d.get('obs'), d.get('action')[1:], d.get('reward')[1:], d.get('terminated')[1:], d.get('task')

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

	def dumps(self):
		if self.cfg.buffer_storage_type == 'lazy_tensor':
			self._buffer.dumps(self._buffer_path)
		else:
			torch.save(self._buffer.state_dict(), self._buffer_state_dict_path)
