from functools import wraps


class _Store:
	def __init__(self):
		self.handlers = {}

	def __setitem__(self, op, fn):
		self.handlers[op.id] = fn

	def __getitem__(self, op):
		found_opty = None
		found_fn = None
		for opid, fn in self.handlers.items():
			if isinstance(op.id, type(opid)):
				if found_opty is None or issubclass(type(opid), found_opty):
					found_opty = type(opid)
					found_fn = fn
		if found_opty is not None:
			return found_fn
		return None


class _Stack:
	def empty():
		return _StackNil()

	def push(self, hd):
		return _StackCons(hd, self)

	def top(self):
		raise NotImplementedError

	def pop(self):
		raise NotImplementedError

	def is_empty(self):
		raise NotImplementedError


class _StackNil(_Stack):
	def __init__(self):
		super().__init__()

	def top(self):
		raise IndexError

	def pop(self):
		raise IndexError

	def is_empty(self):
		return True


class _StackCons(_Stack):
	def __init__(self, hd, tl):
		super().__init__()
		self.hd = hd
		self.tl = tl

	def top(self):
		return self.hd

	def pop(self):
		return self.tl

	def is_empty(self):
		return False


_HANDLER_STACK = _Stack.empty()


class _SaveStack:
	def __init__(self, stack):
		self.stack = stack

	def __enter__(self):
		global _HANDLER_STACK
		self.saved_stack = _HANDLER_STACK
		_HANDLER_STACK = self.stack
		return self

	def __exit__(self, *exc):
		global _HANDLER_STACK
		_HANDLER_STACK = self.saved_stack
		return False


class Handler:
	def __init__(self):
		self.store = _Store()

	def __enter__(self):
		global _HANDLER_STACK
		_HANDLER_STACK = _HANDLER_STACK.push(self)
		return self

	def __exit__(self, *exc):
		global _HANDLER_STACK
		_HANDLER_STACK = _HANDLER_STACK.pop()
		return False

	def register(self, op, fn):
		self.store[op] = fn

	def handle(self, stack, op, *args, **kwargs):
		h = self.store[op]
		if h is not None:
			with _SaveStack(stack):
				result = h(*args, **kwargs)
			return True, result
		return False, None


class Operation:
	def __init__(self):
		self.id = type("operation", (), {})()

	def __call__(self, *args, **kwargs):
		global _HANDLER_STACK
		st = _HANDLER_STACK
		while not st.is_empty():
			handled, result = st.top().handle(st.pop(), self, *args, **kwargs)
			if handled:
				return result
			else:
				st = st.pop()
		raise KeyError()

	def sub(parent):
		inst = Operation.__new__(Operation)
		inst.id = type("operation", (type(parent.id),), {})()
		return inst


def coroutine_decorator(func):
	global _HANDLER_STACK
	st = _HANDLER_STACK

	@wraps(func)
	async def wrapper(*args, **kwargs):
		with _SaveStack(st):
			result = await func(*args, **kwargs)
		return result

	return wrapper
