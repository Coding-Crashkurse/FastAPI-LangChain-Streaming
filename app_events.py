from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessageChunk


load_dotenv()

embedding_function = OpenAIEmbeddings()

docs = [
    Document(
        page_content="the dog loves to eat pizza", metadata={"source": "animal.txt"}
    ),
    Document(
        page_content="the cat loves to eat lasagna", metadata={"source": "animal.txt"}
    ),
]

db = Chroma.from_documents(docs, embedding_function)
retriever = db.as_retriever()


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI(temperature=0, streaming=True)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

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
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )

async def generate_chat_events(message):
    async for event in model.astream_events(message, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
            chunk_content_html = chunk_content.replace("\n", "<br>")
            yield f"data: {chunk_content_html}\n\n"
        elif event["event"] == "on_chat_model_end":
            print("Chat model has completed its response.")


@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(generate_chat_events(message), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
