import os
import time


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
