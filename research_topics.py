import os
import pickle

from dotenv import load_dotenv
from pydantic import BaseModel

from src.coop import AsyncHandler, AsyncSeqLikeHandler, async_
from src.eff import Handler, Operation, coroutine_decorator
from src.llm import (
	AsyncOneRoundChatHandler,
	AsyncReplayLLMHandler,
	LLMHandler,
	OneRoundChatHandler,
	ReplayLLMHandler,
	TraceLLMHandler,
	complete,
	parse,
)
from src.util import Timer, awaitable_args_decorator

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "ali/qwen-turbo-latest"
TEMPERATURE = 1.0
MAX_TOKENS = 1000

PROMPT_TOPICS = "Give a list of topics in the research area: {area}."
PROMPT_DESCRIPTION = "Give a short description about the research topic: {topic}."


get_topics = Operation()
get_description = Operation()
log = Operation()


class ResearchArea(BaseModel):
	topics: list[str]


class ResearchTopicsHandler(Handler):
	"""
	handles: get_topics, get_description, log

	forward: parse, complete
	"""

	def __init__(self):
		super().__init__()
		self.register(get_topics, self.get_topics)
		self.register(get_description, self.get_description)
		self.register(log, print)

	def get_topics(self, area):
		response = parse("topics", PROMPT_TOPICS.format(area=area), ResearchArea)
		return response.choices[0].message.parsed.topics

	def get_description(self, topic):
		response = complete(f"desc_{topic}", PROMPT_DESCRIPTION.format(topic=topic))
		return response.choices[0].message.content


class AsyncResearchTopicsHandler(Handler):
	"""
	handles: get_topics, get_description, log

	forward: parse, complete, async, await
	"""

	def __init__(self):
		super().__init__()
		self.register(get_topics, self.get_topics)
		self.register(get_description, self.get_description)
		self.register(log, self.log)

	def get_topics(self, area):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(area):
			response = await parse(
				"topics", PROMPT_TOPICS.format(area=area), ResearchArea
			)
			return response.choices[0].message.parsed.topics

		return AsyncHandler.wrap_future_object(async_(aux(area)), "__iter__")

	def get_description(self, topic):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(topic):
			response = await complete(
				f"desc_{topic}", PROMPT_DESCRIPTION.format(topic=topic)
			)
			return response.choices[0].message.content

		return async_(aux(topic))

	def log(self, msg):
		@awaitable_args_decorator
		@coroutine_decorator
		async def aux(msg):
			return msg

		return async_(aux(msg), print)


def main():
	topics = get_topics("PL techniques for LLM applications")
	for topic in topics:
		log(topic)
		descrption = get_description(topic)
		log(descrption)


if __name__ == "__main__":
	if not os.path.exists("research_topics.trace"):
		with (
			LLMHandler(base_url=BASE_URL, api_key=API_KEY),
			TraceLLMHandler() as trace,
			OneRoundChatHandler(
				model=MODEL,
				temperature=TEMPERATURE,
				max_tokens=MAX_TOKENS,
			),
			ResearchTopicsHandler(),
		):
			main()
		with open("research_topics.trace", "wb") as f:
			pickle.dump(trace, f)

	with open("research_topics.trace", "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_sync:
		with ReplayLLMHandler(trace), OneRoundChatHandler(), ResearchTopicsHandler():
			main()
	print(f"sync time: {t_sync.time}")

	with open("research_topics.trace", "rb") as f:
		trace = pickle.load(f)
	with Timer() as t_async:
		with (
			AsyncHandler(),
			AsyncReplayLLMHandler(trace),
			AsyncOneRoundChatHandler(),
			AsyncSeqLikeHandler(),
			AsyncResearchTopicsHandler(),
		):
			main()
	print(f"async time: {t_async.time}")

	print(f"speedup: {t_sync.time / t_async.time}x")
