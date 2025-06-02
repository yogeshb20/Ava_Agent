from fastapi import FastAPI
from fastapi.responses import JSONResponse,StreamingResponse
import asyncio
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware
from graphflow import build_workflow
from tools import customer_information
from fastapi import Query
import json


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str  # User's message
    customer_data: Optional[dict] = None  # Customer data (optional, can be None)


class ChatRequestMemory(BaseModel):
    message: str
    thread_id: Optional[str] = "default-thread-id"  # Thread ID for conversation context


# Build the workflow
workflow = build_workflow().compile()
# --- Globals ---
chat_initialized = False
chat_history = []
stored_customer_data = None  # ← NEW: Hold customer data between requests



@app.get("/start_chat")
def start_chat(client_id: int = Query(...), client_name: str = Query(...)):
    global chat_initialized, stored_customer_data
    try:
        stored_customer_data = customer_information(client_id=client_id, client_name=client_name)
        chat_initialized = True
        return JSONResponse(content={"data": jsonable_encoder(stored_customer_data)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    global chat_initialized, chat_history, stored_customer_data

    if not chat_initialized:
        return JSONResponse(status_code=400, content={"error": "Chat not initialized. Call /start_chat first."})

    customer_data = stored_customer_data or request.customer_data
    if not customer_data:
        return JSONResponse(status_code=400, content={"error": "Customer data missing. Please start chat again."})

    # Case: empty message (first load)
    if request.message.strip() == "":
        chat_history.clear()
        chat_history.append(HumanMessage(content="__start__"))
    else:
        chat_history.append(HumanMessage(content=request.message))

    state = {
        "messages": chat_history,
        "last_user_message": request.message if request.message.strip() else "__start__",
        "conversation_ended": False,
        "conversation_completed": False,
        "feedback_given": False,
        "customer_data": customer_data,
    }

    try:
        state = workflow.invoke(state)
        chat_history[:] = state["messages"]
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        full_response = ai_messages[-1].content if ai_messages else "Sorry, no response."

        async def event_stream():
            for i in range(1, len(full_response) + 1):
                yield f"data: {json.dumps({'response': full_response[:i]})}\n\n"
                await asyncio.sleep(0.015)  # simulate "typing" speed

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})