from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, AsyncOpenAI
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv()


provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/"
    )
model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash"
    )

client = RunConfig(
    model=model,
    tracing_disabled=True,
    model_provider=provider
)

agent = Agent(
    name = "AI Assistant",
    instructions="If you want to know about Agentic AI to ask me."
)

result = Runner.run_sync(
    agent,
    input="If you want to know about Agentic AI.",
    run_config=client
)

print("Result---->>>", result.final_output)