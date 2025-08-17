from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI, ModelSettings
import os
from dotenv import load_dotenv
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
    model_provider=provider,
    # max_tokens=1296
)
first_agent = Agent(
    name="Coder_Assistant",
    instructions="You are a expert in Python programming.",
    model_settings=ModelSettings(temperature=0.5)
)
shelo_agent = first_agent.clone(
    name="Creative_Assistant",
    instructions="You are a creative writing assistant. Always respond with vivid, imaginative language.",
    model_settings=ModelSettings(temperature=0.9)
)
Result = Runner.run_sync(
    shelo_agent,
    run_config=client,
    input="Write a short story about a dragon who discovers a hidden talent for painting."
)
print("Result-->>",first_agent)