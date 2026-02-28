import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, LazyMemmapStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs.transforms import Transform

from common.utils import make_dir


STORAGE = {'lazy_tensor': LazyTensorStorage, 'lazy_memmap': LazyMemmapStorage}


class BatchTransform(Transform):
	def __init__(self, device):
		super(BatchTransform, self).__init__()
		self._device = device

	def forward(self, td: TensorDict) -> TensorDict:
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action').contiguous()
		reward = td.get('reward').unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated').unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return TensorDict(
			obs=obs, action=action, reward=reward, terminated=terminated, task=task, batch_size=td.batch_size
		)


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		if self.cfg.buffer_storage_type not in STORAGE:
			raise ValueError(f'Unsupported storage type: {self.cfg.buffer_storage_type}')

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
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		self._buffer = None
		self._buffer_dir = make_dir(cfg.buffer_dir)
		if self.cfg.buffer_storage_type == 'lazy_memmap':
			self._buffer_storage_path = make_dir(self._buffer_dir / 'storage')

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
			transform=BatchTransform(self._device),
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
			kwargs.update(dict(existsok=True, scratch_dir=self._buffer_storage_path))

		storage = cls(**kwargs)
		buffer = self._reserve_buffer(storage)
		if self.cfg.get('resume', False):
			checkpoint_buffer = self._buffer_dir
		else:
			checkpoint_buffer = self.cfg.get('checkpoint_buffer', None)

		if checkpoint_buffer:
			print(f'Loading buffer data from from {checkpoint_buffer}', flush=True)
			buffer.loads(checkpoint_buffer)

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
		self._buffer.dumps(self._buffer_dir)
