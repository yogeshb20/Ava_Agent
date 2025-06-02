
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



# ---- Define ALL Agents -------

# === Agent: Triage ===

def triage_agent(state: MyConversationState) -> MyConversationState:
    instructions = (
        "You are a triage agent. Based on the user input, decide whether the query is about:\n"
        "- 'customer_agent': customer info (name, contact, project)\n"
        "- 'feedback_agent': feedback, complaints, suggestions\n"
        "- 'conversational_agent': general chat or greetings\n"
        "Reply only with one of the choices: customer_agent, feedback_agent, conversational_agent."
    )

    messages = build_agent_messages(state, instructions)
    response = model.invoke(messages)
    # print("response printed:\n", response)

    route = response.content.strip().lower()
    if route not in {"customer_agent", "feedback_agent", "conversational_agent"}:
        route = "conversational_agent"

    return {
        "next": route,
        "messages": state["messages"] + [AIMessage(content=f"Triage routed to: {route}")]
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

    """
    
    agent_graph = create_react_agent(model, tools, prompt=instructions)
    
    # Process feedback as needed (state modification happens here, but we focus on the last user message)
    result = agent_graph.invoke(state)
    
    # Only pass the last_user_message to the conversational agent (filtering out other data)
    last_user_message = state.get('last_user_message', '')
    if last_user_message:
        # Create a new state for the conversational agent with just the last user message
        result = {
            'messages': [HumanMessage(content=last_user_message)],
            'last_user_message': last_user_message,
            'feedback_given': True  # Ensure feedback flag is set
        }
    
    print("Result passing to conversational:\n", result)
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
- Do not give commitments or make assumptions about project status..

Red Flags (Flag for human review if):
- The user mentions budget issues, delays, paused work, team restructuring, or uncertainty about continuation.
- There’s a tone shift that indicates dissatisfaction or hesitation.
- They say things like “we’ll see,” “not sure,” “not a priority,” etc.


Closing:
- Thank the customer sincerely for their time and openness.
- Ask if they’d like someone from Clarion leadership to connect with them directly.
- Keep the conversation short and respectful of their time (aim for under 10 mins total).



Important:
- Always give preference to user input.
- Your job is to gather honest insights, not to defend Clarion or explain performance.
- Never reveal internal Clarion processes or hierarchy.


"""


# === Agent: Conversational ===
def conversational_agent(model, tools, state):
    """Agent for leading thoughtful client conversations, with logic to detect when conversation is complete."""
    print("Conversation Agent Started .....")

    # print("state before conversational agent:\n",state)
    customer_list = state.get("customer_data", [])
    customer = customer_list[0] if isinstance(customer_list, list) and customer_list else {}
    print("Customer details fetched:\n",customer)

    client_name = customer['CLIENT_NAME']
    company = customer['project_name']
    # project = customer.get("project_name", "a project")

    personalized_prompt = f"""
You are Ava, a conversational AI assistant from Clarion Technologies speaking with **{client_name}** from **{company}** for quick , casual conversation to hear honest thoughts about how's things going for your business and would it be okay if i ask few short questions? ask this in initial greetings message only not with every response or questions.
Remember below Instructions:-
- Handles unclear or unexpected responses without restarting.
- Maintains conversational flow by prompting clarification or offering options.
- Avoids repetitive messages (especially greetings)
You can refer this instructions rest of the things {agent_instructions1}
    """
    agent_graph = create_react_agent(model, tools, prompt=personalized_prompt)
    # print("State before feedback agent:", state)

    return agent_graph




# === Intialize Model === 
model, tools = init_model_and_tools()



# === Workflow to store feedback after every response when required


def build_workflow():
    builder = StateGraph(MyConversationState)

    builder.add_node("triage_agent", triage_agent)
    builder.add_node("customer_agent", lambda state: customer_agent(model, tools).invoke(state))
    builder.add_node("conversational_agent", lambda state: conversational_agent(model, tools, state).invoke(state))
    builder.add_node("feedback_agent", lambda state: feedback_agent(model, tools, state))

    builder.set_entry_point("triage_agent")

    # Step 1 Routing
    builder.add_conditional_edges(
        "triage_agent", lambda x: x["next"], {
            "customer_agent": "customer_agent",
            "feedback_agent": "feedback_agent",
            "conversational_agent": "conversational_agent"
        }
    )

    # Step 2 Routing (chain customer_agent to conversational_agent)
    builder.add_edge("customer_agent", "conversational_agent")
    builder.add_edge("feedback_agent", "conversational_agent")  # Ensure the feedback agent routes to conversational agent
    # Final Endpoints
    builder.add_edge("conversational_agent", END)

    return builder


# if __name__ == "__main__":
#     # Initialize model, tools, and workflow
#     model, tools = init_model_and_tools()
#     workflow = build_workflow().compile()

#     # Initial state with an initial greeting trigger
#     state = {
#         "messages": [HumanMessage(content="__start__")],  # System-injected dummy message
#         "last_user_message": "__start__",
#         "conversation_ended": False,
#         "conversation_completed": False,
#         "feedback_given": False,
#         "customer_data": None
#     }

#     print("💬 Clarion Agent Chat Started. Type 'exit' to quit.\n")

#     # Invoke the workflow immediately to trigger the greeting
#     state = workflow.invoke(state)

#     # Show the initial response
#     ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
#     if ai_messages:
#         print("Clarion Agent:", ai_messages[-1].content)

#     # Start interactive loop
#     while True:
#         user_input = input("You: ")

#         if user_input.lower() in ["exit", "quit"]:
#             print("👋 Ending chat. Have a great day!")
#             break

#         # Append user message
#         state["messages"].append(HumanMessage(content=user_input))
#         state["last_user_message"] = user_input

#         # Run workflow step
#         state = workflow.invoke(state)

#         # Find last AI response
#         ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
#         if ai_messages:
#             print("Clarion Agent:", ai_messages[-1].content)

