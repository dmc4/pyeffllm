import asyncio
import time
from collections import defaultdict, deque
from typing import Awaitable, Callable, TypeVar

import backoff
from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel

from src.coop import async_
from src.eff import Handler, Operation, coroutine_decorator

_M = TypeVar("M", bound=BaseModel)

complete_kw: Callable[[str, dict], ChatCompletion | Awaitable[ChatCompletion]] = (
	Operation()
)
parse_kw: Callable[
	[str, dict, type[_M]],
	ParsedChatCompletion[_M] | Awaitable[ParsedChatCompletion[_M]],
] = Operation()

reset: Callable[[], None] = Operation()
addmsg: Callable[[str, str, dict], None] = Operation()
complete: Callable[[str, str, dict], ChatCompletion | Awaitable[ChatCompletion]] = (
	Operation()
)
parse: Callable[
	[str, str, type[_M], dict],
	ParsedChatCompletion[_M] | Awaitable[ParsedChatCompletion[_M]],
] = Operation()


class LLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward:
	"""

	def __init__(self, **client_kwargs):
		super().__init__()
		self.client = OpenAI(**client_kwargs)
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	@backoff.on_exception(backoff.expo, RateLimitError)
	def complete_kw(self, id: str, chat_kwargs: dict) -> ChatCompletion:
		response = self.client.chat.completions.create(**chat_kwargs)
		return response

	@backoff.on_exception(backoff.expo, RateLimitError)
	def parse_kw(
		self, id: str, chat_kwargs: dict, response_format: type[_M]
	) -> ParsedChatCompletion[_M]:
		response = self.client.beta.chat.completions.parse(
			response_format=response_format, **chat_kwargs
		)
		return response


class AsyncLLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward: async
	"""

	def __init__(self, **client_kwargs):
		super().__init__()
		self.client = AsyncOpenAI(**client_kwargs)
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	def complete_kw(self, id: str, chat_kwargs: dict) -> Awaitable[ChatCompletion]:
		@backoff.on_exception(backoff.expo, RateLimitError)
		@coroutine_decorator
		async def aux():
			response = await self.client.chat.completions.create(**chat_kwargs)
			return response

		return async_(aux())

	def parse_kw(
		self, id: str, chat_kwargs: dict, response_format: type[_M]
	) -> Awaitable[ParsedChatCompletion[_M]]:
		@backoff.on_exception(backoff.expo, RateLimitError)
		@coroutine_decorator
		async def aux():
			response = await self.client.beta.chat.completions.parse(
				response_format=response_format, **chat_kwargs
			)
			return response

		return async_(aux())


class TraceLLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward: complete_kw, parse_kw
	"""

	def __init__(self):
		super().__init__()
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	def __enter__(self):
		super().__enter__()
		self.trace = defaultdict(deque)
		return self.trace

	def complete_kw(self, id: str, chat_kwargs: dict) -> ChatCompletion:
		start_time = time.perf_counter()
		response = complete_kw(id, chat_kwargs)
		response_time = time.perf_counter() - start_time
		self.trace[id].append((response.model_dump_json(), response_time))
		return response

	def parse_kw(
		self, id: str, chat_kwargs: dict, schema: type[_M]
	) -> ParsedChatCompletion[_M]:
		start_time = time.perf_counter()
		response = parse_kw(id, chat_kwargs, schema)
		response_time = time.perf_counter() - start_time
		self.trace[id].append((response.model_dump_json(), response_time))
		return response


class AsyncTraceLLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward: complete_kw, parse_kw, async
	"""

	def __init__(self):
		super().__init__()
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	def __enter__(self):
		super().__enter__()
		self.trace = defaultdict(deque)
		return self.trace

	def complete_kw(self, id: str, chat_kwargs: dict) -> Awaitable[ChatCompletion]:
		@coroutine_decorator
		async def aux():
			start_time = time.perf_counter()
			response = await complete_kw(id, chat_kwargs)
			response_time = time.perf_counter() - start_time
			self.trace[id].append((response.model_dump_json(), response_time))
			return response

		return async_(aux())

	def parse_kw(
		self, id: str, chat_kwargs: dict, schema: type[_M]
	) -> Awaitable[ParsedChatCompletion[_M]]:
		@coroutine_decorator
		async def aux():
			start_time = time.perf_counter()
			response = await parse_kw(id, chat_kwargs, schema)
			response_time = time.perf_counter() - start_time
			self.trace[id].append((response.model_dump_json(), response_time))
			return response

		return async_(aux())


class ReplayLLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward:
	"""

	def __init__(self, trace):
		super().__init__()
		self.trace = trace
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	def complete_kw(self, id: str, chat_kwargs: dict) -> ChatCompletion:
		response, response_time = self.trace[id].popleft()
		time.sleep(response_time)
		return ChatCompletion.model_validate_json(response)

	def parse_kw(
		self, id: str, chat_kwargs: dict, schema: type[_M]
	) -> ParsedChatCompletion[_M]:
		response, response_time = self.trace[id].popleft()
		time.sleep(response_time)
		return ParsedChatCompletion[schema].model_validate_json(response)


class AsyncReplayLLMHandler(Handler):
	"""
	handles: complete_kw, parse_kw

	forward: async
	"""

	def __init__(self, trace):
		super().__init__()
		self.trace = trace
		self.register(complete_kw, self.complete_kw)
		self.register(parse_kw, self.parse_kw)

	def complete_kw(self, id: str, chat_kwargs: dict) -> Awaitable[ChatCompletion]:
		@coroutine_decorator
		async def aux():
			response, response_time = self.trace[id].popleft()
			await asyncio.sleep(response_time)
			return ChatCompletion.model_validate_json(response)

		return async_(aux())

	def parse_kw(
		self, id: str, chat_kwargs: dict, schema: type[_M]
	) -> Awaitable[ParsedChatCompletion[_M]]:
		@coroutine_decorator
		async def aux():
			response, response_time = self.trace[id].popleft()
			await asyncio.sleep(response_time)
			return ParsedChatCompletion[schema].model_validate_json(response)

		return async_(aux())


class OneRoundChatHandler(Handler):
	"""
	handles: complete, parse

	forward: complete_kw, parse_kw
	"""

	def __init__(self, instruction="You are a helpful assistant.", **chat_kwargs):
		super().__init__()
		self.instruction = instruction
		self.chat_kwargs = chat_kwargs
		self.register(complete, self.complete)
		self.register(parse, self.parse)

	def complete(
		self, id: str, prompt: str, override_kwargs: dict = {}
	) -> ChatCompletion:
		messages = [
			{"role": "system", "content": self.instruction},
			{"role": "user", "content": prompt},
		]
		response = complete_kw(
			id, dict(messages=messages, **self.chat_kwargs, **override_kwargs)
		)
		return response

	def parse(
		self, id: str, prompt: str, schema: type[_M], override_kwargs: dict = {}
	) -> ParsedChatCompletion[_M]:
		messages = [
			{"role": "system", "content": self.instruction},
			{
				"role": "user",
				"content": f"{prompt}\nPlease respond with respect to the JSON schema {schema.model_json_schema()}.",
			},
		]
		response = parse_kw(
			id, dict(messages=messages, **self.chat_kwargs, **override_kwargs), schema
		)
		return response


class AsyncOneRoundChatHandler(Handler):
	"""
	handles: complete, parse

	forward: complete_kw, parse_kw, async
	"""

	def __init__(self, instruction="You are a helpful assistant.", **chat_kwargs):
		super().__init__()
		self.instruction = instruction
		self.chat_kwargs = chat_kwargs
		self.register(complete, self.complete)
		self.register(parse, self.parse)

	def complete(
		self, id: str, prompt: str, override_kwargs: dict = {}
	) -> Awaitable[ChatCompletion]:
		@coroutine_decorator
		async def aux():
			messages = [
				{"role": "system", "content": self.instruction},
				{"role": "user", "content": prompt},
			]
			response = await complete_kw(
				id, dict(messages=messages, **self.chat_kwargs, **override_kwargs)
			)
			return response

		return async_(aux())

	def parse(
		self, id: str, prompt: str, schema: type[_M], override_kwargs: dict = {}
	) -> Awaitable[ParsedChatCompletion[_M]]:
		@coroutine_decorator
		async def aux():
			messages = [
				{"role": "system", "content": self.instruction},
				{
					"role": "user",
					"content": f"{prompt}\nPlease respond with respect to the JSON schema {schema.model_json_schema()}.",
				},
			]
			response = await parse_kw(
				id,
				dict(messages=messages, **self.chat_kwargs, **override_kwargs),
				schema,
			)
			return response

		return async_(aux())


class ChatHandler(Handler):
	"""
	handles: reset, addmsg, complete, parse

	forward: complete_kw, parse_kw
	"""

	def __init__(self, instruction="You are a helpful assistant.", **chat_kwargs):
		super().__init__()
		self.instruction = instruction
		self.chat_kwargs = chat_kwargs
		self.register(reset, self.reset)
		self.register(addmsg, self.addmsg)
		self.register(complete, self.complete)
		self.register(parse, self.parse)

	def __enter__(self):
		super().__enter__()
		self.messages = [{"role": "system", "content": self.instruction}]
		return self

	def reset(self):
		self.messages = [{"role": "system", "content": self.instruction}]

	def addmsg(self, role: str, content: str, kwargs: dict = {}):
		self.messages.append({"role": role, "content": content, **kwargs})

	def complete(
		self, id: str, prompt: str, override_kwargs: dict = {}
	) -> ChatCompletion:
		self.messages.append({"role": "user", "content": prompt})
		response = complete_kw(
			id, dict(messages=self.messages, **self.chat_kwargs, **override_kwargs)
		)
		self.messages.append(response.choices[0].message)
		return response

	def parse(
		self, id: str, prompt: str, schema: type[_M], override_kwargs: dict = {}
	) -> ParsedChatCompletion[_M]:
		self.messages.append(
			{
				"role": "user",
				"content": f"{prompt}\nPlease respond with respect to the JSON schema {schema.model_json_schema()}.",
			}
		)
		response = parse_kw(
			id,
			dict(messages=self.messages, **self.chat_kwargs, **override_kwargs),
			schema,
		)
		self.messages.append(response.choices[0].message)
		return response


class AsyncChatHandler(Handler):
	"""
	handles: reset, addmsg, complete, parse

	forward: complete_kw, parse_kw, async
	"""

	def __init__(self, instruction="You are a helpful assistant.", **chat_kwargs):
		super().__init__()
		self.instruction = instruction
		self.chat_kwargs = chat_kwargs
		self.register(reset, self.reset)
		self.register(addmsg, self.addmsg)
		self.register(complete, self.complete)
		self.register(parse, self.parse)

	def __enter__(self):
		super().__enter__()
		self.messages = [{"role": "system", "content": self.instruction}]
		return self

	def reset(self):
		self.messages = [{"role": "system", "content": self.instruction}]

	def addmsg(self, role: str, content: str, kwargs: dict = {}):
		self.messages.append({"role": role, "content": content, **kwargs})

	def complete(
		self, id: str, prompt: str, override_kwargs: dict = {}
	) -> Awaitable[ChatCompletion]:
		@coroutine_decorator
		async def aux():
			self.messages.append({"role": "user", "content": prompt})
			response = await complete_kw(
				id, dict(messages=self.messages, **self.chat_kwargs, **override_kwargs)
			)
			self.messages.append(response.choices[0].message)
			return response

		return async_(aux())

	def parse(
		self, id: str, prompt: str, schema: type[_M], override_kwargs: dict = {}
	) -> Awaitable[ParsedChatCompletion[_M]]:
		@coroutine_decorator
		async def aux():
			self.messages.append(
				{
					"role": "user",
					"content": f"{prompt}\nPlease respond with respect to the JSON schema {schema.model_json_schema()}.",
				}
			)
			response = await parse_kw(
				id,
				dict(messages=self.messages, **self.chat_kwargs, **override_kwargs),
				schema,
			)
			self.messages.append(response.choices[0].message)
			return response

		return async_(aux())
