from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chain import get_chat_response

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


@app.get("/")
def root():
    return {"message": "RAG server is running"}

@app.post("/messages")
def chat(req: ChatRequest):
    answer = get_chat_response(req.query, req.history)
    return {"answer": answer}