import asyncio
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


async def stream_chat_message(message):
    model = ChatOpenAI(temperature=0, streaming=True)
    events = []
    async for event in model.astream_events(message, version="v1"):
        events.append(event)
    return events


def main():
    message = "I am a streamed message"
    events = asyncio.run(stream_chat_message(message))
    print(events)

    event_types = {event["event"] for event in events}
    print("Unique event types:", event_types)


if __name__ == "__main__":
    main()
