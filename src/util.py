import time
from functools import wraps
from typing import Awaitable


class Timer:
	def __enter__(self):
		self.start_time = time.perf_counter()
		return self

	def __exit__(self, *exc):
		self.time = time.perf_counter() - self.start_time
		return False


def awaitable_args_decorator(func):
	@wraps(func)
	async def wrapper(*args, **kwargs):
		cargs = []
		ckwargs = dict()
		for arg in args:
			if isinstance(arg, Awaitable):
				cargs.append(await arg)
			else:
				cargs.append(arg)
		for key, arg in kwargs.items():
			if isinstance(arg, Awaitable):
				ckwargs[key] = await arg
			else:
				ckwargs[key] = arg
		result = await func(*cargs, **ckwargs)
		return result

	return wrapper
