from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import graph

app = FastAPI()

class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Doctor AI Agent is running"

    }

@app.post("/chat")
def chat(request: ChatRequest):

    result = graph.invoke({
         "messages": [
              HumanMessage(content=request.message)
         ]
    })
   

    return {
        "response": result["messages"][-1].content
    }