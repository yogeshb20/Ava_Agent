# ========================== Imports ==========================
import os
import re
import json
import time
import smtplib
import logging
import logging.config

from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import quote_plus
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# from apscheduler.schedulers.background import BackgroundScheduler

from openai import OpenAI
from langchain_openai import ChatOpenAI

from secret import secret_dict, LocalDB_dict, email_config, email_conf
from tools import get_db_connection, name, username, password, host, port



# ====================== Logging Setup ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ================ OpenAI Environment Setup ===================
os.environ["OPENAI_API_KEY"] = secret_dict["Api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini",temperature=0.00001)


# ===================== Email Functions =======================

def send_error_notification(subject: str, error_details: str):
    body = f"An error occurred in the chatbot application:\n\n{error_details}"
    send_email_error(subject, body)


def send_email_error(subject: str, body: str):
    try:
        msg = MIMEMultipart()
        msg['From'] = email_conf["EMAIL"]
        msg['To'] = ', '.join(email_conf["TO"])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        logger.info("Attempting to send error notification email.")
        with smtplib.SMTP(email_conf["SMTP_SERVER"], email_conf["SMTP_PORT"]) as server:
            server.starttls()
            server.login(email_conf["EMAIL"], email_conf["PASSWORD"])
            server.send_message(msg)

        logger.info("Error notification email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send error email: {e}")


def send_email(subject: str, body: str):
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config["EMAIL"]
        msg['To'] = ', '.join(email_config["TO"])
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        logger.info("Sending email...")
        with smtplib.SMTP(email_config["SMTP_SERVER"], email_config["SMTP_PORT"]) as server:
            server.starttls()
            server.login(email_config["EMAIL"], email_config["PASSWORD"])
            server.send_message(msg)

        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        send_error_notification("Chatbot Error: Failed to send Chat email", str(e))


# ============== Sentiment Analysis Instructions ==============
analysis_instructions = """
You are an AI assistant responsible for analyzing structured customer feedback.
Return only a raw JSON object with the following structure:

{
  "account_name": "<Use from input or 'Not specified'>",
  "customer_name": "<Use from input or 'Not specified'>",
  "overall_sentiment": "Positive | Neutral | Negative | Mixed",
  "account_health": "Healthy | Stable but Watch | At Risk",
  "risk_flags": ["..."],
  "key_quotes": ["..."],
  "action_recommendation": "...",
  "opportunities": ["..."],
  "priority_level": "High | Medium | Low"
}

Do not generate anything beyond this structure.
"""


analysis_new_instructions = """
AI Agent Instructions: Customer Feedback Analysis & Reporting

Goal:
Analyze a structured customer feedback conversation and generate a JSON summary capturing key insights, satisfaction levels, pain points, and actionable recommendations.

Input:
A conversation transcript between Ava (Clarion’s virtual assistant) and a customer. The conversation follows a consistent structure with 5 key areas:
1. Business Health & Direction
2. Expectations from Clarion
3. Team Member Feedback
4. Technology & AI Exploration
5. Future Needs, Risks, or Opportunities

Output:
Respond with only a valid JSON object, using the following structure. Do not include any extra commentary, formatting, markdown, or separators. Ensure all values are properly escaped and the JSON can be parsed using json.loads() in Python.

Example Format:

{
  "customer_name": "<Customer Name>",
  "conversation_date": "<YYYY-MM-DD>",
  "overall_sentiment": "<Positive | Neutral | Challenging>",
  "business_insight": "<Short summary of business status>",
  "business_posture": "<Growth | Holding Steady | Risk-Averse>",
  "expectations": [
    {
      "priority": 1,
      "expectation": "<text>",
      "rating": <1-10>,
      "gap": <true | false>,
      "notes": "<optional comment>"
    }
    // up to 4
  ],
  "team_feedback": [
    {
      "name": "<Team Member Name>",
      "role": "<Role>",
      "rating": <1-5>,
      "strengths": "<text>",
      "improvement": "<text>"
    }
    // one per team member
  ],
  "manager_name": "<Manager Name>",
  "manager_summary": "<Summary of performance and support>",
  "sdm_name": "<SDM Name>",
  "sdm_summary": "<Summary or 'Limited interaction'>",
  "technology_summary": "<Summary or 'No current initiatives shared'>",
  "future_summary": "<Summary or 'Customer currently sees no new needs or risks'>",
  "action_recommendations": [
    "<recommendation 1>",
    "<recommendation 2>"
    // optional list
  ]
}

Instructions:
- Use double quotes for all strings.
- Return only valid JSON. Do not add headings, bullet points, markdown, or horizontal lines.
- Ensure the JSON is syntactically correct.
"""



# ==================== Main Processing ========================


# def format_sentiment_output(data: dict) -> str:
#     def section(title: str, items: list) -> str:
#         if not items:
#             return f"{title}:\nNone identified."
#         return f"{title}:\n" + "\n".join(f"• {item}" for item in items)

#     parts = [
#         "",
#         f"🔹 Account Name: {data.get('account_name', 'Not specified')}",
#         f"🔹 Customer Name: {data.get('customer_name', 'Not specified')}",
#         f"🔹 Overall Sentiment: {data.get('overall_sentiment', 'Unknown')}",
#         f"🔹 Account Health: {data.get('account_health', 'Unknown')}",
#         f"🔹 Priority Level: {data.get('priority_level', 'Unknown')}",
#         "",
#         section("⚠️ Risk Flags", data.get("risk_flags", [])),
#         "",
#         section("💬 Key Quotes", data.get("key_quotes", [])),
#         "",
#         "🧭 Action Recommendation:",
#         data.get("action_recommendation", "No recommendation provided."),
#         "",
#         section("🌱 Opportunities", data.get("opportunities", []))
#     ]
#     return "\n".join(parts)



from datetime import datetime

def format_sentiment_output(data: dict) -> str:
    def format_expectations(expectations: list) -> str:
        if not expectations:
            return "ℹ️ No expectations captured."
        header = "📌 Priority | Expectation | Rating (1–10) | Notes / Gap Identified"
        rows = []
        for item in expectations:
            gap_note = item.get("notes", "No notes")
            rows.append(f"{item['priority']} | {item['expectation']} | {item['rating']} | {gap_note}")
        return f"{header}\n" + "\n".join(rows)

    def format_team_feedback(feedback: list) -> str:
        if not feedback:
            return "ℹ️ No team feedback provided."
        header = "👤 Name | Role | Rating | Strengths | Areas for Improvement"
        rows = []
        for member in feedback:
            rows.append(f"{member['name']} | {member['role']} | {member['rating']} | {member['strengths']} | {member['improvement']}")
        return f"{header}\n" + "\n".join(rows)

    def format_recommendations(recommendations: list) -> str:
        if not recommendations:
            return "📭 No actionable recommendations provided."
        return "\n".join(f"✅ {rec}" for rec in recommendations)

    today_str = datetime.today().strftime("%B %d, %Y")

    parts = [
        "📝 Customer Feedback Summary Report",
        
        f"👤 Customer Name: {data.get('customer_name', 'Not specified')}",
        f"🗓️ Date of Conversation: {today_str}",
        "",
        "📊 1. Business Health & Direction",
        f"🔸 Sentiment: {data.get('overall_sentiment', 'Unknown')}",
        f"🔸 Key Insight: {data.get('business_insight', 'No insight provided.')}",
        f"🔸 Business Posture: {data.get('business_posture', 'Unknown')}",
        "",
        "🎯 2. Expectations from Clarion",
        format_expectations(data.get("expectations", [])),
        "",
        "👥 3. Team Feedback",
        format_team_feedback(data.get("team_feedback", [])),
        "",
        f"👨‍💼 Manager ({data.get('manager_name', 'Not specified')}): {data.get('manager_summary', 'No feedback provided.')}",
        f"👩‍💼 SDM ({data.get('sdm_name', 'Not specified')}): {data.get('sdm_summary', 'No feedback provided.')}",
        "",
        "🧠 4. Technology & AI Landscape",
        data.get("technology_summary", "No current initiatives shared."),
        "",
        "🔮 5. Future Needs, Risks, Opportunities",
        data.get("future_summary", "Customer currently sees no new needs or risks."),
        "",
        "💡 Recommendations",
        format_recommendations(data.get("action_recommendations", []))
    ]

    return "\n".join(parts)


def format_chat_thread(thread: list, customer_name: str) -> str:
    formatted_lines = []
    for entry in thread:
        content = entry.get("content", "").strip()

        # Skip irrelevant routing/system messages
        if content.lower().startswith("triage routed to:"):
            continue

        role = entry.get("role")
        if role == "assistant":
            speaker = "Ava"
        elif role == "user":
            speaker = customer_name
        else:
            speaker = "Unknown"

        formatted_lines.append(f"{speaker}: {content}")

    return "\n\n".join(formatted_lines)





# def send_email_after_chat_end(session_id: str):
#     data = active_conversations.pop(session_id, None)
#     if not data:
#         logger.warning(f"No data found for session: {session_id}")
#         return

#     logger.info(f"Session {session_id} has ended. Sending email.")

#     thread_details = data["thread"]
#     customer_name = data["customer_name"]
#     company_name = data["company_name"]

#     model_input = {
#         "account_name": company_name,
#         "customer_name": customer_name,
#         "feedback_entries": thread_details
#     }

#     try:
#         response = model.invoke([
#             {"role": "system", "content": analysis_instructions},
#             {"role": "user", "content": json.dumps(model_input, indent=2)}
#         ])

#         structured_output = json.loads(response.content)
        
#         sentiment_section = format_sentiment_output(structured_output)
#         chat_transcript = format_chat_thread(thread_details)

#         email_body = f"""
# Hi Team,

# Here's the feedback summary for {customer_name} from {company_name}.


# ============================
# 📊 Sentiment Analysis Summary
# ============================
# {sentiment_section}

# ============================
# 📌 Chat Transcript
# ============================
# {chat_transcript}



# Regards,  
# Your Chatbot System
# """

#         send_email(f"Sentiment Analysis: Feedback from {customer_name}", email_body)
#         logger.info(f"✅ Sentiment email sent for session {session_id}")
#     except Exception as e:
#         logger.error(f"❌ Failed to send email for session {session_id}: {e}")
#         send_error_notification("Chatbot Error: Failed to send sentiment email", str(e))



# import threading
# from datetime import datetime, timedelta

# active_conversations = {}  # session_id -> conversation data

# def handle_user_message(session_id: str, customer_name: str, company_name: str, message: dict):
#     now = datetime.now()
#     data = active_conversations.get(session_id)

#     if not data:
#         # First message in conversation
#         data = {
#             "thread": [],
#             "customer_name": customer_name,
#             "company_name": company_name,
#             "last_message_time": now,
#             "timer": None
#         }
#         active_conversations[session_id] = data

#     # Add message to thread
#     data["thread"].append(message)
#     data["last_message_time"] = now

#     # Cancel any previous timer
#     if data["timer"]:
#         data["timer"].cancel()

#     # Set new timer to trigger email after 1 minute of inactivity
#     timer = threading.Timer(60.0, send_email_after_chat_end, args=[session_id])
#     data["timer"] = timer
#     timer.start()
