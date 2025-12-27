import chainlit as cl
from agents import (
    Runner,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    AsyncOpenAI,
)
from openai.types.responses import ResponseTextDeltaEvent
from agents.tool import function_tool
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINY_API_KEY")

# 1st Step : Initialize the OpenAI provider with Gemini model

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2nd Step: Create the OpenAIChatCompletionsModel with the provider

model = OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=provider)

# 3rd Step: Create the RunConfig with the model and provider
run_config = RunConfig(model=model, model_provider=provider, tracing_disabled=True)

@function_tool("get_weather")
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Dummy implementation for demonstration purposes
    return f"The current weather in {location} is sunny with a temperature of 25°C."



# 4th Step: Create the Agent
agent1 = Agent(
    instructions="You are a helpful assistant that answer the questions. ",
    name="Panaversity Support Agent",
     tools=[get_weather],
)


@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content="Hello! I am your Personal Support Agent. How can I assist you today?"
    ).send()


# 5th Step: Define the Chainlit message handler to use the runner


@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")

    msg = cl.Message(content="")
    await msg.send()

    history.append({"role": "user", "content": message.content})
    result = Runner.run_streamed(
        agent1,
        input=history,
        run_config=run_config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)

    # await cl.Message(content=result.final_output).send()
