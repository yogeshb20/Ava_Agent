from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

from agent import *
from graphflow import build_workflow
from tools import customer_information



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
def start_chat():
    global chat_initialized, stored_customer_data
    try:
        stored_customer_data = customer_information(client_id=1, client_name='mark')  # <- Fetch and store
        chat_initialized = True
        return JSONResponse(content={"data": jsonable_encoder(stored_customer_data)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})




@app.post("/chat")
async def chat(request: ChatRequest):
    global chat_history, chat_initialized, stored_customer_data

    if not chat_initialized:
        return JSONResponse(status_code=400, content={"error": "Chat not initialized. Please call /start_chat first."})

    # Get customer data
    customer_data = request.customer_data or stored_customer_data
    last_user_msg = request.message.strip() if request.message else ""
    if not customer_data:
        return JSONResponse(status_code=400, content={"error": "Customer data is required."})


    # Append only the latest user message to full chat history
    chat_history.append(HumanMessage(content=last_user_msg + str(customer_data)))

    # Prepare trimmed context: last 4 messages + current one (max 5 total)
    recent_context = chat_history[-4:] if len(chat_history) >= 4 else chat_history[:]
    recent_context.append(HumanMessage(content=last_user_msg))

    # Call workflow with only trimmed context
    try:
        result_state = workflow.invoke({
            "messages": recent_context,
            "last_user_message": last_user_msg,
            "customer_data": customer_data
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    ai_messages = [msg for msg in result_state["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        response = ai_messages[-1].content
        chat_history.append(AIMessage(content=response))  # Keep full chat history for future reference
    else:
        response = "Sorry, I didn't understand that."

    # Reset state if conversation ends
    if result_state.get("conversation_ended", False):
        chat_history = []
        chat_initialized = False
        stored_customer_data = None

    return response


























