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

from email.utils import formataddr

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



# def send_email(subject: str, body: str):
#     try:
#         msg = MIMEMultipart()
#         msg['From'] = formataddr(("Ava – Clarion Virtual Assistant", email_config["EMAIL"]))
#         msg['To'] = ', '.join(email_config["TO"])
#         msg['Subject'] = subject

#         msg.attach(MIMEText(body, 'html'))  # HTML formatting

#         logger.info("Sending email...")
#         with smtplib.SMTP(email_config["SMTP_SERVER"], email_config["SMTP_PORT"]) as server:
#             server.starttls()
#             server.login(email_config["EMAIL"], email_config["PASSWORD"])
#             server.send_message(msg)

#         logger.info("Email sent successfully.")
#     except Exception as e:
#         logger.error(f"Failed to send email: {e}")
#         send_error_notification("Chatbot Error: Failed to send Chat email", str(e))



def send_email(subject: str, body: str, to_emails: list = None, cc_emails: list = None):
    try:
        msg = MIMEMultipart()
        msg['From'] = formataddr(("Ava – Clarion Virtual Assistant", email_config["EMAIL"]))
        msg['To'] = ', '.join(to_emails or email_config["TO"])
        msg['Cc'] = ', '.join(cc_emails or email_config["CC"])
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        # Combine TO and CC recipients for delivery
        recipients = (to_emails or email_config["TO"]) + (cc_emails or email_config["CC"])

        logger.info(f"Sending email to {recipients}")
        with smtplib.SMTP(email_config["SMTP_SERVER"], email_config["SMTP_PORT"]) as server:
            server.starttls()
            server.login(email_config["EMAIL"], email_config["PASSWORD"])
            server.send_message(msg, from_addr=email_config["EMAIL"], to_addrs=recipients)

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
  "business_insight": "<Summary of business status>",
  "business_posture": "<Growth | Holding Steady | Risk-Averse>",
  "expectations": [
    // Leave this array empty if the customer has not explicitly shared any expectations
    
    {
      "priority": 1,
      "expectation": "<text>",
      "rating": <1-10>,
      "gap": <true | false>,
      "notes": "<optional comment>"
    },
    {
      "priority": 2,
      "expectation": "<text>",
      "rating": <1-10>,
      "gap": <true | false>,
      "notes": "<optional comment>"
    },
    {
      "priority": 3,
      "expectation": "<text>",
      "rating": <1-10>,
      "gap": <true | false>,
      "notes": "<optional comment>"
    },
    {
      "priority": 4,
      "expectation": "<text>",
      "rating": <1-10>,
      "gap": <true | false>,
      "notes": "<optional comment>"
    }
  ],
  "team_feedback": [
  // Exclude Managers and SDM inside team_feedback only include team members
    {
      "name": "<Team Member Name>",
      "role": "<Role>",
      "rating": <1-5>,
      "strengths": "<text>",
      "improvement": "<text>"
    }
  ],
  "manager_name": "<Manager Name>",
  "manager_summary": "<Summary of performance and support>",
  "sdm_name": "<SDM Name>",
  "sdm_summary": "<Summary or 'Limited interaction'>",
  "technology_summary": "<Summary or 'No current initiatives shared'>",
  "future_summary": "<Summary or 'Customer currently sees no new needs or risks'>",
   "analysis_and_recommendations": "<A well-structured paragraph explaining the insights and actionable recommendations based on the conversation.>"

}


Instructions:
- Use double quotes for all strings.
- Return only valid JSON. Do not add headings, bullet points, markdown, or horizontal lines.
- Ensure the JSON is syntactically correct.
"""



def format_sentiment_output_html(data: dict, chat_transcript: str, customer_name: str, conversation_date: str) -> str:
    def render_expectations(expectations, manager_name, sdm_name):
        rows = ""
        manager_name = (manager_name or "").lower()
        sdm_name = (sdm_name or "").lower()

        for item in expectations:
            exp_text = item.get("expectation", "").lower()

            # Skip if it refers to the manager or SDM by name or title
            if any(kw in exp_text for kw in [manager_name, sdm_name, "manager", "sdm"]):
                continue

            rows += f"""
            <tr>
                <td style="text-align: center;">{item.get("priority", "")}</td>
                <td>{item.get("expectation", "")}</td>
                <td style="text-align: center;">{item.get("rating", "")}</td>
                <td>{item.get("notes", "")}</td>
            </tr>
            """
        return rows or "<tr><td colspan='4'>No valid customer expectations shared.</td></tr>"

    def render_team_feedback(team_feedback):
        rows = ""
        for member in team_feedback:
            rows += f"""
            <tr>
                <td>{member.get("name", "")}</td>
                <td style="text-align: center;">{member.get("rating", "")}</td>
                <td>{member.get("strengths", "")}</td>
                <td>{member.get("improvement_areas", "")}</td>
            </tr>
            """
        return rows

    def render_analysis_and_recommendations(paragraph: str):
        return f"<p>{paragraph}</p>" if paragraph else "<p>No analysis or recommendations shared.</p>"

    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6;">
        <p>Hi Team,</p>
        <p>I am Ava, the virtual assistant from Clarion Technologies.</p>
        <p>I have just completed a feedback conversation with our customer, <strong>{customer_name}</strong>, and I would like to share my <strong>analysis and recommendations</strong> based on that discussion.</p>
        <p>Please note that this analysis is generated purely from the conversation and the signals shared by the customer during the chat.</p>
        <p>Since many of you have a deeper understanding of the account context and relationship history, you may have better interpretation and judgment on certain points.
        To support that, I’ve included the <strong>entire chat transcript</strong> below the analysis for your reference.</p>

        <h2>Customer Feedback Summary Report</h2>
        <p><strong>Customer Name:</strong> {customer_name}<br>
        <strong>Date of Conversation:</strong> {conversation_date}</p>

        <h3>1. Business Health & Direction</h3>
        <ul>
            <li><strong>Sentiment:</strong> {data.get("overall_sentiment", "Unknown")}</li>
            <li><strong>Key Insight:</strong> {data.get("business_insight", "N/A")}</li>
            <li><strong>Business Posture:</strong> {data.get("business_posture", "N/A")}</li>
        </ul>

        <h3>2. Expectations from Clarion</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th style="text-align: center;">Priority</th>
                <th>Expectation</th>
                <th style="text-align: center;">Rating (1–10)</th>
                <th>Notes / Gap Identified</th>
            </tr>
            { render_expectations(data.get("expectations", []), data.get("manager_name", ""), data.get("sdm_name", "")) }
        </table>

        <h3>3. Team Feedback</h3>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>Name</th>
                <th>Rating (1–5)</th>
                <th>Strengths</th>
                <th>Areas for Improvement</th>
            </tr>
            { render_team_feedback(data.get("team_feedback", [])) }
        </table>

        <p>
            <strong>{data.get("manager_name", "N/A")} (Manager):</strong> {data.get("manager_summary", "N/A")}<br>
            <strong>{data.get("sdm_name", "N/A")} (SDM):</strong> {data.get("sdm_summary", "N/A")}
        </p>

        <h3>4. Technology & AI Landscape</h3>
        <ul>
            <p>{data.get("technology_summary", "No current initiatives shared.")}</p>
        </ul>
        <h3>5. Future Needs, Risks, Opportunities</h3>
        <ul>
            <p>{data.get("future_summary", "None shared.")}</p>
        </ul>

        <h3>6. Analysis & Recommendations</h3>
        <div>
            {render_analysis_and_recommendations(data.get("analysis_and_recommendations", ""))}
        </div>

        <h2>Full Chat Transcript with {customer_name}:</h2>
    <pre style="background-color: #f4f4f4; padding: 10px; border: 1px solid #ccc; white-space: pre-wrap;">
{chat_transcript}
</pre>

        <p>Regards,<br>Ava<br>Virtual Assistant – Clarion Technologies</p>
    </body>
    </html>
    """

def format_chat_thread(thread: list, customer_name: str) -> str:
    formatted_lines = []
    customer_first_name = customer_name.split()[0]

    for entry in thread:
        content = entry.get("content", "").strip()

        # Skip irrelevant routing/system messages or empty content
        if not content or content.lower().startswith("triage routed to:"):
            continue

        # Clean up excessive line breaks and spaces inside the message
        cleaned_lines = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

        # Remove leading or trailing whitespace from cleaned_lines to avoid unnecessary gap
        cleaned_lines = cleaned_lines.strip()

        role = entry.get("role")
        if role == "assistant":
            speaker = f"<strong><span style='font-size: 1.05em;'>Ava</span></strong>"
        elif role == "user":
            speaker = f"<strong><span style='font-size: 1.05em;'>{customer_first_name}</span></strong>"
        else:
            speaker = "Unknown"

        # Only add to formatted_lines if content is not empty
        if cleaned_lines:
            formatted_lines.append(f"{speaker}: {cleaned_lines}")

    return "\n\n".join(formatted_lines)



# def format_chat_thread(thread: list, customer_name: str) -> str:
#     formatted_lines = []
#     customer_first_name = customer_name.split()[0]

#     for entry in thread:
#         content = entry.get("content", "").strip()

#         # Skip irrelevant routing/system messages or empty content
#         if not content or content.lower().startswith("triage routed to:"):
#             continue

#         # Clean up excessive line breaks and spaces inside the message
#         cleaned_lines = "\n".join([line.strip() for line in content.splitlines() if line.strip()])

#         role = entry.get("role")
#         if role == "assistant":
#             speaker = f"<strong><span style='font-size: 1.05em;'>Ava</span></strong>"
#         elif role == "user":
#             speaker = f"<strong><span style='font-size: 1.05em;'>{customer_first_name}</span></strong>"
#         else:
#             speaker = "Unknown"

#         formatted_lines.append(f"{speaker}: {cleaned_lines}")

#     return "\n\n".join(formatted_lines)



# def format_chat_thread(thread: list, customer_name: str) -> str:
#     formatted_lines = []
#     for entry in thread:
#         content = entry.get("content", "").strip()

#         # Skip irrelevant routing/system messages
#         if content.lower().startswith("triage routed to:"):
#             continue

#         role = entry.get("role")
#         if role == "assistant":
#             speaker = "Ava"
#         elif role == "user":
#             speaker = customer_name.split()[0]
#         else:
#             speaker = "Unknown"

#         formatted_lines.append(f"{speaker}: {content}")

#     return "\n\n".join(formatted_lines)



