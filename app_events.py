from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk

load_dotenv()

model = ChatOpenAI(temperature=0, streaming=True)
app = FastAPI()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def serialize_aimessagechunk(chunk):
    """
    Custom serializer for AIMessageChunk objects.
    Convert the AIMessageChunk object to a serializable format.
    """
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    elif isinstance(chunk, dict) and "content" in chunk:
        return chunk["content"]
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    async def generate_chat_events():
        async for event in model.astream_events(message, version="v1"):
            if event["event"] == "on_chat_model_stream":
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                chunk_content_html = chunk_content.replace("\n", "<br>")
                yield f"data: {chunk_content_html}\n\n"
            elif event["event"] == "on_chat_model_end":
                print("Chat model has completed its response.")

    return StreamingResponse(generate_chat_events(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
