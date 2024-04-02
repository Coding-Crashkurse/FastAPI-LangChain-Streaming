from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI


model = ChatOpenAI(temperature=0, streaming=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat_stream/{message}")
async def chat_stream(message: str):
    async def generate_chat_responses():
        async for chunk in model.astream(message):
            content = chunk.content.replace("\n", "<br>")
            yield f"data: {content}\n\n"

    return StreamingResponse(generate_chat_responses(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
