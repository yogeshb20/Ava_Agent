# === Imports ===
# Standard library imports
import os
import getpass
from typing import Literal, TypedDict, Annotated, Optional

# Third-party imports
from openai import OpenAI

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage


from langgraph.graph import StateGraph, MessagesState, START, END, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from tools import *
from secret import secret_dict


# === Environment Setup ===
os.environ["OPENAI_API_KEY"] = secret_dict["Api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")

# === Conversation State ===

class MyConversationState(TypedDict, total=False):
    messages: list[BaseMessage]  # Keep track of all messages
    last_user_message: str  # Last message from the user
    conversation_ended: bool  # Flag if conversation is ended
    conversation_completed: bool  # Flag if conversation is completed
    feedback_given: bool  # Flag if feedback has been given
    customer_data: Optional[dict]  # Customer-related data (added to track the customer)



# ---- Define suporting Functions ------

# === OpenAI API Key Setup ===
def setup_openai_key(): 
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("API key must be set")


# === Initialize Model and Tools ===
def init_model_and_tools():
    setup_openai_key()
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    # Ensure the tools are included here
    tools = [
        # conversational_tool,
        get_customer_info_db,
        store_feedback_data_db,
        get_feedback_by_customer_and_developer 
    ]
    
    # Return both model and tools
    return model, tools


# === Agent with Memory ===
def create_agent_with_memory(model, tools):
    memory = MemorySaver()
    model = model.bind_tools(tools)
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    return agent_executor, memory


# === Utility: Message Builder ===
def build_agent_messages(state: MessagesState, system_prompt: str):
    "Thios Function used for Triage Agent to fetch the response from the message"
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    return [SystemMessage(content=system_prompt)] + user_messages

# === Agent: Superviser Triage ===
def triage_agent(state: MyConversationState) -> MyConversationState:
    user_message = state["messages"][-1].content.lower()
    instructions = (
        "You are a triage agent. Based on the user input, decide whether the query is about:\n"
        "- 'customer_agent': customer info (when the user shares their name or asks about their account, project, or contact info)\n"
        "- 'conversational_agent': general chat, greetings, or small talk\n"
        "- 'feedback_agent': feedback collection (only used once at end of chat)\n"
        "- 'sentiment_agent': for retrieving the feedback , showing the sentiment analysis\n"
        "Reply only with one of the choices: customer_agent, feedback_agent,sentiment_agent, conversational_agent.\n"
    )

    messages = build_agent_messages(state, instructions)
    response = model.invoke(messages)

    route = response.content.strip().lower()
    if route not in {"customer_agent", "conversational_agent","sentiment_agent"}:
        route = "conversational_agent"
    print(f"Triage Input: {user_message} → Routed to: {route}")
    return {
        "messages": state["messages"] + [AIMessage(content=f"Triage routed to: {route}")],
        "next": route,
        "conversation_ended": False
    }



# === Agent: Customer Info ===

def customer_agent(model, tools):
    """Agent for handling customer-related queries."""
    print("Customer Agent started .....")
    instructions = """
    You are the Customer Agent. Your job is to retrieve and share customer profile and project details based on their name.
    You can have limitation to call only 'get_customer_info_db` tool

    Your responsibilities:
    - Return accurate, up-to-date customer information, including active projects and recent feedback if available.
    - Do not analyze, interpret, or summarize—just provide the data as it is.
    - Keep your response clear and focused on the requested customer.
    - keep response should be structured to unerstand for other agents

    Avoid small talk or commentary. Your job is to support other agents with customer info.
"""
    return create_react_agent(model, tools, prompt=instructions)


# === Agent: Feedback ===
def feedback_agent(model, tools, state):
    print("Feedback Agent started .....")
    instructions = """
    You are the Feedback Agent. Your role is to capture and organize client feedback for future use. Communicate clearly and naturally—avoid robotic or transactional language.

    Function access:
    - You may only use the following functions: for storing the feedback use `store_feedback_data_db`.

    Your responsibilities:
    - Record feedback accurately and clearly, adding a timestamp if available (typically at the end of the conversation).
    - Label feedback as positive, constructive, or neutral—without adding opinions or interpretations.
    - Retrieve and summarize past feedback on request to support future check-ins.
    - Ensure all feedback is easy for other agents to access and refer to.

    Tone and style:
    - Do not use robotic confirmations like “Feedback has been recorded.”
    - Acknowledge feedback naturally, using phrases like “Thanks for sharing your thoughts,” or “Wishing you continued success with your project.”
    - Do not ask follow-up questions or prompt for more input once feedback is shared.
    - Keep your tone warm, professional, and concise. Avoid small talk, assumptions, or unnecessary commentary.
    - End conversations gracefully once the feedback is captured.
    """


    
    agent_graph = create_react_agent(model, tools, prompt=instructions)
    # print("State before feedback agent:", state)
    result = agent_graph.invoke(state)
    result['feedback_given'] = True
    return result



def sentiment_agent(model, tools):
    prompt = """
You are an AI assistant responsible for analyzing chat conversations between a Clarion Technologies customer and Clarion’s AI Agent. Your job is to extract clear, actionable insights about the customer's sentiment and account health.

Your output must follow the **exact JSON structure below** — no markdown, no extra explanation, just a raw valid JSON object. If any information is unavailable, state that clearly.

Follow these instructions carefully:
1. Assess the overall sentiment of the customer: "Positive", "Neutral", "Negative", or "Mixed".
2. Evaluate the account health based on business signals: "Healthy", "Stable but Watch", or "At Risk".
3. Identify and list any clear risk flags:
   - Budget concerns
   - Leadership or strategy changes
   - Project deprioritization or pause
   - Unclear long-term commitment
   - Dissatisfaction with team or process
4. Extract 1–3 direct quotes or paraphrased lines that reflect the customer’s true feelings or concerns.
5. Recommend a clear follow-up action:
   - "Human escalation or leadership check-in"
   - "Monitor only"
   - "Explore upsell/cross-sell opportunity"
   - "No action needed"
6. List any opportunities mentioned.
7. Assign a priority level: "High", "Medium", or "Low".

Return output in very well structured format with every new point has bullent and it should be on new line with heading to be bold
Refer below JSON format:

{
  "account_name": "<Extracted from metadata or inferred from chat>",
  "customer_name": "<Extracted if mentioned>",
  "overall_sentiment": "Positive | Neutral | Negative | Mixed",
  "account_health": "Healthy | Stable but Watch | At Risk",
  "risk_flags": ["..."],
  "key_quotes": ["..."],
  "action_recommendation": "...",
  "opportunities": ["..."],
  "priority_level": "High | Medium | Low"
}

### Example:
Conversation excerpt:
Customer: "I’m not happy with the progress of Ajay and Rajesh. Both are not proactive in communication and their thinking is not very productive."
Agent: "Could you give examples?"
Customer: "It’s an ongoing issue."
Use below J
Expected JSON:
{
  "account_name": "E-Commerce Checkout Flow",
  "customer_name": "Mark Johnson",
  "overall_sentiment": "Negative",
  "account_health": "At Risk",
  "risk_flags": [
    "Dissatisfaction with team or process",
    "Ongoing communication issues"
  ],
  "key_quotes": [
    "I'm not happy with the progress of Ajay and Rajesh.",
    "Both are not proactive in communication and way of thinking for project is not very productive.",
    "It's an ongoing issue."
  ],
  "action_recommendation": "Human escalation or leadership check-in",
  "opportunities": [],
  "priority_level": "High"
}

Now analyze the provided conversation below and return your findings in well structured format.
"""

    return create_react_agent(model, tools, prompt=prompt)



agent_instructions1 = """
You are Ava, a conversational AI assistant from Clarion Technologies. Your primary role is to assess true account health and gather strategic signals from the customer through a friendly, open-ended conversation.

Tone & Personality:
- You are empathetic, business-aware, and respectful.
- You sound like a Customer Success Manager, not a bot.
- Always maintain a warm, non-intrusive tone.

Objective:
- Go beyond operational metrics and identify early signals of risk, such as:
  - Budget constraints
  - Project deprioritization
  - Changing leadership or vision
  - Lack of perceived value
  - Concerns not captured in feedback forms

Guidelines:
- Start with a warm welcome and make it clear this is *not* a formal survey.
- Let the user know their feedback is confidential and will not be directly shared with developers.
- Use open-ended questions to explore their experience.
- Ask follow-up questions if you sense hesitation or vague responses.
- Do NOT push for negative feedback, but invite honest reflection.
- If user expresses frustration, acknowledge empathetically and assure them their concerns will be handled appropriately.
- Avoid technical jargon unless the customer uses it first.
- Do not give commitments or make assumptions about project status.

Red Flags (Flag for human review if):
- The user mentions budget issues, delays, paused work, team restructuring, or uncertainty about continuation.
- There’s a tone shift that indicates dissatisfaction or hesitation.
- They say things like “we’ll see,” “not sure,” “not a priority,” etc.

Unclear or Gibberish Input:
- If the input is unclear or appears as gibberish (e.g., random symbols like @#$%^&*()_-, numbers 2534518387, or disconnected words Hi i am @ $%&gN), respond with a friendly, open-ended clarification request like "I didn’t quite catch that. Could you provide a bit more detail?" or "It looks like something got a bit jumbled—can you explain it again?"
- Vary the responses to keep them natural and empathetic.

Closing:
- Thank the customer sincerely for their time and openness.
- Ask if they’d like someone from Clarion leadership to connect with them directly.
- Keep the conversation short and respectful of their time (aim for under 10 mins total).

Conversation End Detection:
- End the conversation gracefully if the user indicates they’re done (e.g., “I’m good”, “that’s all”, “no thanks”).
- Otherwise, gently check if there’s anything else they’d like to add or if a follow-up would be helpful.

Important:
- Your job is to gather honest insights, not to defend Clarion or explain performance.
- Never reveal internal Clarion processes or hierarchy.


"""


# === Agent: Conversational ===
def conversational_agent(model, tools, state):
    """Agent for leading thoughtful client conversations, with logic to detect when conversation is complete."""
    print("Conversation Agent Started .....")
    agent_graph = create_react_agent(model, tools, prompt=agent_instructions1)

    user_input = state.get("last_user_message", "").lower().strip()
    print("User input given:", user_input)

    # Detect intent to end conversation
    end_convo_detected = user_input in [
        "no thanks", "no", "nothing else", "i’m good", "that’s all", "nope", "end chat","i am good","thats all","submit","end","end conversation","store feedback"
    ]

    # Call the conversation agent
    agent_result = agent_graph.invoke(state)

    # Update result with additional context
    agent_result.update({
        "last_user_message": user_input,
        "feedback_given": state.get("feedback_given", False),
    })

    if end_convo_detected:
        print("Detected end-of-conversation intent.")
        agent_result["conversation_completed"] = True

    return agent_result


model, tools = init_model_and_tools()

# === Agent: Check for Conversation Complete ===
def check_conversation_complete(state):
    print(">>> check_conversation_complete CALLED")
    # print("State keys:", state.keys())
    # print("State snapshot:", state)

    # Flags for conversation completion and feedback
    conversation_done = state.get('conversation_completed', False)
    feedback_done = state.get('feedback_given', False)
    
    print(f"Conversation Complete: {conversation_done}, Feedback Given: {feedback_done}")

    # Check if the conversation is done and feedback isn't given
    if conversation_done and not feedback_done:
        # If conversation is complete but feedback has not been given, route to feedback agent
        return {"next": "feedback_agent"}

    # If feedback is done, mark the conversation as fully ended
    if conversation_done and feedback_done:
        state["conversation_ended"] = True
        return {"next": "__end__"}

    # Default return value if no specific condition is met
    return {"next": "__end__"}
