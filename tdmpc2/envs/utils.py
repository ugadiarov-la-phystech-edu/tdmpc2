class InvalidTaskException(Exception):
	def __init__(self, task, suite):
		self.task = task
		self.suite = suite
		super().__init__(f"Cannot find task {self.task} in suite {self.suite}")


class MissingDependencyException(Exception):
	def __init__(self, suite):
		self.suite = suite
		super().__init__(f'Missing dependencies for suite {self.suite}; install dependencies to use this environment.')
