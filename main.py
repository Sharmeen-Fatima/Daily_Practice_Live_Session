from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig,AsyncOpenAI, function_tool
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client_provider = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", 
    openai_client=client_provider)
# set_tracing_disabled(True)

config = RunConfig(model = model,
                   model_provider = client_provider,
                   tracing_disabled = True)

@function_tool
def sdk_docs_reader(url: str) -> str:
    """fetches the contect of a url and returns the text contant."""
    print("TOOL CALLED")
    # URL = "https://openai.github.io/openai-agents-python/index.html"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n", strip = True)
    else:
        return f"fail to retrieve contant from {url} status code {response.status_code}"

Openai_Agent_SDK = Agent(
    name = "OpenAi Agent SDK",
    instructions = (
        "You are an expert in the OpenAI Agent SDK. Use the tool to read the official SDK documentation and answer user questions accurately."
        "You MUST use the `sdk_docs_reader` tool to fetch the latest content and respond accordingly."
        "If the user asks about another topic Say 'I Designe to answer about OpenAI Agent SDK' 'Good Bye!' and end the conversation. And use Atitude Emojis in your response."
        "Always use the Emojis in your response to make it more engaging."
    ),
    tools = [sdk_docs_reader]
)

agentic_Developer = Agent(
    name="Agentic Developer",
    instructions=(
        "You are a professional OpenAI Agent SDK Developer 👨‍💻.\n"
        "When the user asks for a new agent, FIRST fetch and analyze the latest OpenAI Agent SDK code from the official docs using the sdk_docs_reader tool 🧠.\n"
        "THEN build a new custom agent using the same structure, style, and patterns from the SDK documentation ⚙️, but DON'T reuse or copy-paste example code word-for-word 🚫.\n"
        "Instead, generate a FRESH and UNIQUE agent according to user needs that follows the SDK format (like using Agent, function_tool, RunConfig, etc.) ✅.\n"
        "If the user asks something outside OpenAI Agent SDK development, say '❌ I only generate OpenAI Agent SDK-based agents. Goodbye!' and end the conversation 💨.\n"
        "Always use cool emojis in your replies to keep it fun 😎🔥.\n"
    ),
    tools=[sdk_docs_reader],
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="You are a triage agent. Your job is to determine whether the user needs help with the OpenAI Agent SDK or if they need a new agent built. If they need help with the SDK, use the OpenAI Agent SDK agent. If they need a new agent built, use the Agentic Developer agent. If they ask about anything else, say 'I only handle OpenAI Agent SDK-related queries. Goodbye!' and end the conversation.",
    handoffs=[Openai_Agent_SDK, agentic_Developer]
)

prompt = f"""Fetch the latest documentation from https://openai.github.io/openai-agents-python/index.html 
and answer this question:\n\n{input("Enter your question about the OpenAI Agent SDK: ")}"""

while True:
    user_input = input("Enter your question about the OpenAI Agent SDK (Type 'exit' to quit):")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye! see you next time!!...👋")
        break

result = Runner.run_sync(
    triage_agent,
    prompt,
    run_config = config
)

print(result.final_output)