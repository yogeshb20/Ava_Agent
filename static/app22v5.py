from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
from tester1 import init_model_and_tools, build_workflow
from tools import customer_information
from openai import APIConnectionError
import uuid
import json
import requests
import logging
import asyncio
from datetime import datetime
from mailsent2v2 import (
    send_email,
    format_sentiment_output,
    format_chat_thread,
    analysis_instructions,
    model as sentiment_model
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model, tools = init_model_and_tools()
workflow = build_workflow().compile()

# ✅ Session-based state
chat_sessions = {}  # session_id: {chat_history, customer_data}
session_last_active = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    customer_data: Optional[dict] = None


@app.get("/start_chat")
def start_chat(client_id: int = Query(...), client_name: str = Query(...)):
    try:
        customer_data = customer_information(client_id=client_id, client_name=client_name)
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            "chat_history": [],
            "customer_data": customer_data
        }
        session_last_active[session_id] = datetime.utcnow()
        return JSONResponse(content={
            "session_id": session_id,
            "data": jsonable_encoder(customer_data)
        })

    except requests.exceptions.ConnectionError:
        logger.exception("Network connection error during customer info retrieval")
        return JSONResponse(status_code=503, content={"error": "Unable to connect to the customer data service."})
    except requests.exceptions.Timeout:
        logger.exception("Timeout while fetching customer info")
        return JSONResponse(status_code=504, content={"error": "Request to customer data service timed out."})
    except Exception as e:
        logger.exception("Unexpected error in start_chat")
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    session_id = request.session_id
    if not session_id or session_id not in chat_sessions:
        return JSONResponse(status_code=400, content={"error": "Invalid or missing session_id. Please start chat again."})

    session_data = chat_sessions[session_id]
    chat_history = session_data["chat_history"]
    customer_data = session_data["customer_data"]

    if not customer_data:
        return JSONResponse(status_code=400, content={"error": "Customer data missing. Please start chat again."})

    if request.message is None or request.message.strip() == "":
        chat_history.clear()
        chat_history.append(HumanMessage(content="__start__"))
    else:
        chat_history.append(HumanMessage(content=request.message.strip()))

    session_last_active[session_id] = datetime.utcnow()

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
        chat_sessions[session_id]["chat_history"] = state["messages"]
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        full_response = ai_messages[-1].content if ai_messages else "Sorry, no response."

        async def event_stream():
            for i in range(1, len(full_response) + 1):
                yield f"data: {json.dumps({'response': full_response[:i]})}\n\n"
                await asyncio.sleep(0.015)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except APIConnectionError:
        logger.exception("API connection error: Unable to reach OpenAI service.")
        return JSONResponse(status_code=503, content={"error": "Connection lost. Try again."})
    except Exception as e:
        logger.exception("Unexpected error in chat_stream")
        return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})


@app.post("/end_chat")
async def end_chat(request: ChatRequest):
    session_id = request.session_id
    if not session_id or session_id not in chat_sessions:
        return JSONResponse(status_code=400, content={"error": "Invalid or missing session_id."})

    session_data = chat_sessions[session_id]
    chat_history = session_data.get("chat_history", [])
    customer_data = session_data.get("customer_data")

    if not chat_history or all(msg.content == "__start__" for msg in chat_history if isinstance(msg, HumanMessage)):
        logger.info("🛑 Conversation not completed. Email not sent.")
        return JSONResponse(status_code=200, content={"message": "No actual conversation. Email not sent."})

    if not customer_data:
        return JSONResponse(status_code=400, content={"error": "No customer data found."})

    try:
        thread_details = []
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                continue

            if msg.content.strip() != "__start__":
                thread_details.append({"role": role, "content": msg.content.strip()})

        customer = customer_data[0] if isinstance(customer_data, list) else customer_data
        model_input = {
            "account_name": customer.get("COMPANY_NAME", "Not specified"),
            "customer_name": customer.get("CLIENT_NAME", "Not specified"),
            "feedback_entries": thread_details
        }

        response = sentiment_model.invoke([
            {"role": "system", "content": analysis_instructions},
            {"role": "user", "content": json.dumps(model_input, indent=2)}
        ])
        structured_output = json.loads(response.content)

        sentiment_section = format_sentiment_output(structured_output)
        chat_transcript = format_chat_thread(thread_details, model_input["customer_name"])

        email_body = f"""
Hi Team,

Here's the feedback summary for {model_input['customer_name']} from {model_input['account_name']}.


============================
📊 Sentiment Analysis
============================
{sentiment_section}


============================
📌 Chat Transcript
============================
{chat_transcript}



Regards,  
Ava Virtual AI Assistant.
"""

        send_email(f"Sentiment Analysis: Feedback from {model_input['customer_name']}", email_body)
        logger.info("✅ Email sent after user closed the chat.")

        # Optional cleanup
        del chat_sessions[session_id]
        session_last_active.pop(session_id, None)

        return JSONResponse(status_code=200, content={"message": "Chat ended and email sent."})

    except Exception as e:
        logger.error(f"❌ Failed to process end chat: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to send feedback email."})
