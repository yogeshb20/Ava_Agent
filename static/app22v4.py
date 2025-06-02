from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
# ✅ Import sentiment and email functions (no DB)
from mailsent import (
    send_email,
    format_sentiment_output,
    analysis_instructions,
    model as sentiment_model
)

app = FastAPI()

# Static files (e.g. frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model init
model, tools = init_model_and_tools()
workflow = build_workflow().compile()

# Global state
chat_initialized = False
chat_history = []
stored_customer_data = None
# Global session tracker
session_last_active = {}


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_message: Optional[str] = None
    message: Optional[str] = None  # ✅ Fix: Make it optional
    customer_data: Optional[dict] = None



# async def monitor_inactive_sessions():
#     while True:
#         await asyncio.sleep(60)  # Run check every 1 minute
#         now = datetime.utcnow()
#         inactive_sessions = []

#         for session_id, last_active in session_last_active.items():
#             if (now - last_active).total_seconds() > 3600:  # 30 minutes
#                 inactive_sessions.append(session_id)

#         for session_id in inactive_sessions:
#             try:
#                 logger.info(f"⏰ Auto-ending inactive session {session_id}")
#                 await end_chat()  # You could pass session-specific data here if needed
#                 del session_last_active[session_id]
#             except Exception as e:
#                 logger.error(f"Failed to auto-end session {session_id}: {e}")

# @app.on_event("startup")
# async def on_startup():
#     asyncio.create_task(monitor_inactive_sessions())


# @app.get("/start_chat")
# def start_chat(client_id: int = Query(...), client_name: str = Query(...)):
#     global chat_initialized, stored_customer_data
#     try:
#         # Replace this stub with actual customer data function
#         stored_customer_data = customer_information(client_id=client_id, client_name=client_name)
#         print("stored customer data from start chat api:\n",stored_customer_data)
#         # stored_customer_data = {"client_id": client_id, "client_name": client_name, "company_name": "Unknown Company"}
#         chat_initialized = True
#         return JSONResponse(content={"data": jsonable_encoder(stored_customer_data)})
#     except Exception as e:
#         logger.exception("Unexpected error in start_chat")
#         return JSONResponse(status_code=500, content={"error": f"Unexpected error: {str(e)}"})

@app.get("/start_chat")
def start_chat(client_id: int = Query(...), client_name: str = Query(...)):
    global chat_initialized, stored_customer_data
    try:
        stored_customer_data = customer_information(client_id=client_id, client_name=client_name)
        # print("customer data from start chat api:\n",stored_customer_data)
        chat_initialized = True
        return JSONResponse(content={"data": jsonable_encoder(stored_customer_data)})

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
    global chat_initialized, chat_history, stored_customer_data, session_last_active
    session_id = request.session_id or str(uuid.uuid4())
    

    if not chat_initialized:
        return JSONResponse(status_code=400, content={"error": "Chat not initialized. Call /start_chat first."})

    customer_data = stored_customer_data or request.customer_data
    # print("customer data obtained from chat api:\n",customer_data)
    if not customer_data:
        return JSONResponse(status_code=400, content={"error": "Customer data missing. Please start chat again."})

    if request.message.strip() == "":
        chat_history.clear()
        chat_history.append(HumanMessage(content="__start__"))
    else:
        chat_history.append(HumanMessage(content=request.message))

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
        chat_history[:] = state["messages"]
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
async def end_chat():
    global chat_history, stored_customer_data

    if not chat_history or all(msg.content == "__start__" for msg in chat_history if isinstance(msg, HumanMessage)):
        logger.info("🛑 Conversation not completed. Email not sent.")
        return JSONResponse(status_code=200, content={"message": "No actual conversation. Email not sent."})
    print("stored customer data in end chat api:\n",stored_customer_data)
    if not stored_customer_data:
        return JSONResponse(status_code=400, content={"error": "No customer data found."})

    try:
        feedback_entries = [
            {
                "feedback_text": msg.content,
                "aspect": "Chat",
                "developer": "AI",
                "sentiment": "Positive|Negative|Mixed"
            }
            for msg in chat_history if isinstance(msg, HumanMessage) and msg.content != "__start__"
        ]

        customer = stored_customer_data[0] if isinstance(stored_customer_data, list) else stored_customer_data


        model_input = {
            "account_name": customer.get("COMPANY_NAME", "Not specified"),
            "customer_name": customer.get("CLIENT_NAME", "Not specified"),
            "feedback_entries": feedback_entries
        }
        print("Model input from end chat api:\n",model_input)

        response = sentiment_model.invoke([
            {"role": "system", "content": analysis_instructions},
            {"role": "user", "content": json.dumps(model_input, indent=2)}
        ])

        structured_output = json.loads(response.content)
        email_body = format_sentiment_output(structured_output)

        send_email(f"Sentiment Analysis: Feedback from {model_input['customer_name']}", email_body)
        logger.info("✅ Email sent after user closed the chat.")

        return JSONResponse(status_code=200, content={"message": "Chat ended and email sent."})

    except Exception as e:
        logger.error(f"❌ Failed to process end chat: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to send feedback email."})



