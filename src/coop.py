import asyncio
from typing import Any, Awaitable, Callable, Coroutine, TypeVar

from src.eff import Handler, Operation, coroutine_decorator

_T = TypeVar("T")
_R = TypeVar("R")

async_: Callable[[Coroutine[Any, Any, _T], Callable[[_T], _R]], Awaitable[_R]] = (
	Operation()
)
await_: Callable[[Awaitable[_T]], _T] = Operation()


class AsyncHandler(Handler):
	"""
	handles: async, await

	forward:
	"""

	def __init__(self):
		super().__init__()
		self.register(async_, self.async_)
		self.register(await_, self.await_)

	def __enter__(self):
		super().__enter__()
		self.loop = asyncio.new_event_loop()
		self.futures = []
		return self

	def __exit__(self, *exc):
		super().__exit__(*exc)
		for future in self.futures:
			self.loop.run_until_complete(future)
		self.loop.close()
		return False

	def async_(
		self, coro: Coroutine[Any, Any, _T], post_fn: Callable[[_T], _R] = lambda x: x
	) -> Awaitable[_R]:
		@coroutine_decorator
		async def aux():
			result = await coro
			return post_fn(result)

		future = self.loop.create_task(aux())
		self.futures.append(future)
		return future

	def await_(self, future: Awaitable[_T]) -> _T:
		self.loop.run_until_complete(future)
		return future.result()

	def wrap_future_object(future, *attrs):
		def make(attr):
			def f(self, *args, **kwargs):
				if isinstance(self.future, Awaitable):
					self.future = await_(self.future)
				return self.future.__getattribute__(attr)(*args, **kwargs)

			return f

		dict = {"future": future}
		for attr in attrs:
			dict[attr] = make(attr)
		return type("wrap_future_object", (), dict)()


class AsyncSeqLikeHandler(Handler):
	"""
	handles: async

	forward: async, await
	"""

	def __init__(self):
		super().__init__()
		self.register(async_, self.async_)

	def __enter__(self):
		super().__enter__()
		self.init_event = asyncio.Event()
		self.prev_event = self.init_event
		self.init_event.set()
		return self

	def __exit__(self, *exc):
		super().__exit__(*exc)
		await_(async_(self.prev_event.wait()))
		return False

	def async_(
		self, coro: Coroutine[Any, Any, _T], post_fn: Callable[[_T], _R] = lambda x: x
	) -> Awaitable[_R]:
		@coroutine_decorator
		async def aux(prev_event, next_event):
			result = await coro
			await prev_event.wait()
			result = post_fn(result)
			next_event.set()
			return result

		next_event = asyncio.Event()
		future = async_(aux(self.prev_event, next_event))
		self.prev_event = next_event
		return future
