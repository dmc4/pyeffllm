import os
from typing import List, Literal

from dotenv import load_dotenv
from openai import pydantic_function_tool
from pydantic import BaseModel, Field

from src.coop import AsyncHandler, async_, coroutine_decorator
from src.llm import (
	AsyncLLMHandler,
	AsyncOneRoundChatHandler,
	ChatHandler,
	LLMHandler,
	OneRoundChatHandler,
	addmsg,
	complete,
	parse,
)

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = "ali/qwen-plus-latest"
TEMPERATURE = 1.0
MAX_TOKENS = 1000


def prompt_chaining():
	def generate_joke(topic):
		return (
			complete("generate", f"Write a short joke about {topic}")
			.choices[0]
			.message.content
		)

	def check_punchline(joke):
		return "?" in joke or "!" in joke

	def improve_joke(joke):
		return (
			complete("improve", f"Make this joke funnier by adding wordplay: {joke}")
			.choices[0]
			.message.content
		)

	def polish_joke(joke):
		return (
			complete("polish", f"Add a surprising twist to this joke: {joke}")
			.choices[0]
			.message.content
		)

	joke = generate_joke("cats")
	while not check_punchline(joke):
		joke = improve_joke(joke)
	joke = polish_joke(joke)
	print(joke)


def parallelization():
	def call(req, topic):
		@coroutine_decorator
		async def aux():
			response = await complete(f"write_{req}", f"Write a {req} about {topic}")
			return response.choices[0].message.content

		return AsyncHandler.wrap_future_object(async_(aux()), "__str__")

	def aggregate(topic, joke, story, poem):
		combined = f"Here's a story, joke, and poem about {topic}!\n\n"
		combined += f"STORY:\n{story}\n\n"
		combined += f"JOKE:\n{joke}\n\n"
		combined += f"POEM:\n{poem}"
		return combined

	topic = "cats"
	joke = call("joke", topic)
	story = call("story", topic)
	poem = call("poem", topic)
	result = aggregate(topic, joke, story, poem)
	print(result)


def routing():
	class Route(BaseModel):
		step: Literal["poem", "story", "joke"] = Field(
			None, description="The next step in the routing process"
		)

	def router(input):
		response = parse(
			"router",
			f"Route the following input to story, joke, or poem: {input}",
			Route,
		)
		return response.choices[0].message.parsed.step

	def call_story(input):
		response = complete("story", f"You are a wonderful story teller. {input}")
		return response.choices[0].message.content

	def call_joke(input):
		response = complete("story", f"You are a funny joke creater. {input}")
		return response.choices[0].message.content

	def call_poem(input):
		response = complete("story", f"You are an elegant poem composer. {input}")
		return response.choices[0].message.content

	input = "Write me a joke about cats"
	decision = router(input)
	if decision == "story":
		print(call_story(input))
	elif decision == "joke":
		print(call_joke(input))
	elif decision == "poem":
		print(call_poem(input))
	else:
		assert False


def orchestrator_worker():
	class Section(BaseModel):
		name: str = Field(
			description="Name for this section of the report.",
		)
		description: str = Field(
			description="Brief overview of the main topics and concepts to be covered in this section.",
		)

	class Sections(BaseModel):
		sections: List[Section] = Field(
			description="Sections of the report.",
		)

	def orchestrate(topic):
		@coroutine_decorator
		async def aux():
			response = await parse(
				"orchestrate",
				f"Generate a plan for the report with topic {topic}",
				Sections,
			)
			return response.choices[0].message.parsed.sections

		return AsyncHandler.wrap_future_object(async_(aux()), "__iter__")

	def call(section):
		@coroutine_decorator
		async def aux():
			response = await complete(
				f"write_{section.name}",
				f"Write a report section, whose name is {section.name} and description is {section.description}",
			)
			return response.choices[0].message.content

		return AsyncHandler.wrap_future_object(async_(aux()), "__str__")

	def synthesize(reports):
		return "\n\n---\n\n".join(map(str, reports))

	sections = orchestrate("Create a report on LLM scaling laws")
	reports = [call(section) for section in sections]
	final_report = synthesize(reports)
	print(final_report)


def evaluator_optimizer():
	class Feedback(BaseModel):
		grade: Literal["funny", "not funny"] = Field(
			description="Decide if the joke is funny or not.",
		)
		feedback: str = Field(
			description="If the joke is not funny, provide feedback on how to improve it.",
		)

	def generate(topic, feedback=None):
		if feedback is None:
			return (
				complete("generate", f"Write a joke about {topic}")
				.choices[0]
				.message.content
			)
		else:
			return (
				complete(
					"refine",
					f"Write a joke about {topic} but take into account the feedback: {feedback}",
				)
				.choices[0]
				.message.content
			)

	def evaluate(joke):
		return (
			parse("evaluate", f"Grade the joke {joke}", Feedback)
			.choices[0]
			.message.parsed
		)

	topic = "cats"
	feedback = None
	while True:
		joke = generate(topic, feedback)
		feedback = evaluate(joke)
		if feedback.grade == "funny":
			break
	print(joke)


def agent():
	class Multiply(BaseModel):
		a: int
		b: int

	class Add(BaseModel):
		a: int
		b: int

	class Divide(BaseModel):
		a: int
		b: int

	def call_tool(tool_call):
		name = tool_call.function.name
		arguments = tool_call.function.arguments
		if name == "Add":
			args = Add.model_validate_json(arguments)
			return args.a + args.b
		elif name == "Multiply":
			args = Multiply.model_validate_json(arguments)
			return args.a * args.b
		elif name == "Divide":
			args = Divide.model_validate_json(arguments)
			return args.a / args.b
		else:
			assert False

	tools = [
		pydantic_function_tool(Add),
		pydantic_function_tool(Multiply),
		pydantic_function_tool(Divide),
	]
	prompt = "Perform arithmetic on a set of inputs: Add 30 divided by 6 and the product of 4 and 5."
	message = (
		complete(
			"agent",
			prompt,
			dict(tools=tools),
		)
		.choices[0]
		.message
	)
	while True:
		if message.tool_calls is None:
			break
		for tool_call in message.tool_calls:
			addmsg("tool", str(call_tool(tool_call)), dict(tool_call_id=tool_call.id))
		message = complete("agent", prompt, dict(tools=tools)).choices[0].message
	print(message.content)


if __name__ == "__main__":
	print("==================================")
	print("  Prompt Chaining")
	print("==================================")
	with (
		LLMHandler(base_url=BASE_URL, api_key=API_KEY),
		OneRoundChatHandler(
			model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
		),
	):
		prompt_chaining()

	print()
	print("==================================")
	print("  Parallelization")
	print("==================================")
	with (
		AsyncHandler(),
		AsyncLLMHandler(base_url=BASE_URL, api_key=API_KEY),
		AsyncOneRoundChatHandler(
			model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
		),
	):
		parallelization()

	print()
	print("==================================")
	print("  Routing")
	print("==================================")
	with (
		LLMHandler(base_url=BASE_URL, api_key=API_KEY),
		OneRoundChatHandler(
			model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
		),
	):
		routing()

	print()
	print("==================================")
	print("  Orchestrator-Worker")
	print("==================================")
	with (
		AsyncHandler(),
		AsyncLLMHandler(base_url=BASE_URL, api_key=API_KEY),
		AsyncOneRoundChatHandler(
			model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
		),
	):
		orchestrator_worker()

	print()
	print("==================================")
	print("  Evaluator-Optimizer")
	print("==================================")
	with (
		LLMHandler(base_url=BASE_URL, api_key=API_KEY),
		OneRoundChatHandler(
			model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
		),
	):
		evaluator_optimizer()

	print()
	print("==================================")
	print("  Agent")
	print("==================================")
	with (
		LLMHandler(base_url=BASE_URL, api_key=API_KEY),
		ChatHandler(model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS),
	):
		agent()
