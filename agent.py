
# === Imports ===
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from typing import Literal
import os
import getpass
from openai import OpenAI
from secret import secret_dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from tools import *
from typing import TypedDict
from langchain.schema import BaseMessage
from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

# === Environment Setup ===
os.environ["OPENAI_API_KEY"] = secret_dict["Api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")

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
    model = init_chat_model("gpt-4o", model_provider="openai")
    
    # Ensure the tools are included here
    tools = [
        # get_customer_info_db,
        store_feedback_data_db,   
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
        "- 'conversational_agent': general chat or greetings\n"
        "Reply only with one of the choices:  conversational_agent."
    )

    messages = build_agent_messages(state, instructions)
    response = model.invoke(messages)
    # print("response printed:\n", response)

    route = response.content.strip().lower()
    if route not in {"conversational_agent"}:
        route = "conversational_agent"

    return {
        "next": route,
        "messages": state["messages"] + [AIMessage(content=f"Triage routed to: {route}")]
    }



# === Agent: Conversational ===
def conversational_agent(model, tools, state):
    """Rewritten agent to fetch customer data and use it in a personalized prompt."""
    print("Conversation Agent Started .....")
    # print("Conversation state:\n", state)
    # print("Available keys in state:", list(state.keys()))
    
    # Try to get client_id and client_name from state first
    client_id = state.get("CLIENT_ID")
    client_name = state.get("CLIENT_NAME")
    
    # If not found, try to extract from customer_data
    customer_info = state.get("customer_data")
    # print("Customer information from conversational agent:\n", customer_info)
    
    if not client_id or not client_name:
        if customer_info and isinstance(customer_info, list) and len(customer_info) > 0:
            first_customer = customer_info[0]
            client_id = client_id or first_customer.get("CLIENT_ID")
            client_name = client_name or first_customer.get("CLIENT_NAME")
            first_name = client_name.split()[0]
    
    print("Obtained client id and client name:\n", client_id, client_name)

    # Fetch customer data using your provided function
    customer_list = customer_information(client_id=client_id, client_name=client_name)
    # print("Customer data from conversational agent:\n", customer_list)
    
    # Store updated customer list back in state
    state["customer_data"] = customer_list

    # Select first customer (if available)
    customer = customer_list[0] if customer_list else {}
    client_name = customer.get("CLIENT_NAME", "there")
    employees = customer.get("project_assigned_to", "")
    employee_list = [e.strip() for e in employees.split(",") if e.strip()]
    SDM_name = ['Prakash']
    Manager_name = ['Darshan']
    Team_size = 2
    # Get user input and history
    user_input = state.get("last_user_message", "")
    chat_history = state.get("messages", [])


    personalized_prompt = f"""You are Ava, a conversational AI assistant from Clarion Technologies. Your goal is to engage the customer in a brief, friendly, and professional conversation to gather insights in five key areas.

    Please follow this sequence exactly and maintain a natural, conversational tone — not robotic.

    ---

    ### Welcome Script:
    Begin every conversation exactly as follows:
    "Hi {first_name}, I’m Ava — a virtual assistant from Clarion Technologies.
    I’m here to check in and understand how things are going for you and your business. This is a quick, casual conversation to hear your honest thoughts so we can support you better.
    
    Would it be okay if I ask you a few short questions?"

    ---

    ### Topics and Questions:

    **1. Business Health & Direction**  
    - **Do ask explicitly about their business, NOT about Clarion specifically yet.**
    - **Do NOT mention Clarion or projects directly at this stage.**
    - Use exactly:
    > “How are things going with your business currently? Are there any shifts in direction, priorities, or challenges we should be aware of?”
    - **You may ask follow up questions to get better understanding.**

    ---

    **2. Feedback and Satisfaction for Clarion**  
    - After the business health question, explicitly move to Clarion feedback:
    > "To serve you better, may I ask—what are your top 3-4 expectations from a technology partner like Clarion? If possible, please list these expectations in order of priority, starting with the most important."
    -- **Once the customer shares their expectations, ask the following question**
    > "Thank you for sharing that. On a scale of 1 to 10, how would you rate us on each of those expectations based on your experience so far?"

    ---

    **3. Feedback and Satisfaction for Assigned Team Members** 
     Team members name for referance {employee_list} and team size {Team_size}
    - Begin by asking the general question:
    > “How are our team members performing from your perspective—in terms of ownership, communication, and technical capabilities?”

    - explicitly ask the following, mentioning each team member by name:
    > “Could you please rate [Team Member Name] individually on a scale of 1 to 5?
    Additionally, could you briefly share [Team Member Name]’s strengths, any areas for improvement, and any suggestions you might have?”

    - Repeat this individually for each team member whose name you've been provided.

    - After gathering feedback for team members, explicitly ask for feedback about the Manager and SDM, mentioning them by name:
    > “Could you also share your feedback on Manager {Manager_name}  and SDM {SDM_name}—particularly in terms of leadership, communication, and overall support?”

    ---

    **4. Technology Landscape & AI Plans**  
    - Ask exactly:
    > “Are there any upcoming technology changes, innovations, or AI use cases you’re exploring where we could support you?”

    ---

    **5. Future Demand, Opportunities & Risks**  
    - Ask exactly:
    > “Do you foresee any upcoming needs, risks, or opportunities where you’d like us to be more involved?”

    ---

    ### Tone & Personality:
    - Empathetic, respectful, and professional.
    - Sound like a Customer Success Manager, not robotic.
    - Warm, natural, and conversational.

    ---

    ### General Guidelines:
    - Assure the customer explicitly if confidentiality is questioned:  
    "Your feedback is confidential and will only be shared with Clarion’s leadership team—not directly with developers."
    - Invite honest reflection, but do NOT press hard for negative feedback.
    - Empathize with frustration if expressed and commit only to escalating appropriately, without promising specifics.
    - Begin by acknowledging or summarizing the customer's last response in a thoughtful, empathetic tone wherever required. Then, transition smoothly into the next question, keeping the conversation natural and connected.
    - Avoid abrupt topic shifts — new question should feel like a continuation.
    - If the customer shares something meaningful, thank them or reflect it back before moving on.
    - If the customer's answer lacks clarity or they provide a low rating or negative remark, politely ask a follow-up question to better understand their perspective and gather insights for an improvement plan.
    - Always maintain empathy and professionalism to reassure the customer that their feedback is valuable and will help Clarion serve them better.

    ---

    ### Closing Script:
    Always end with this exact phrasing:
    "Thank you very much for your time and openness today. Would you like someone from Clarion leadership to connect with you directly? If so, please let me know."
    """

    # Create and return agent graph
    agent_graph = create_react_agent(model, tools, prompt=personalized_prompt)
    return agent_graph




# === Intialize Model === 
model, tools = init_model_and_tools()



# === Workflow to store feedback after every response when required


def build_workflow():
    builder = StateGraph(MyConversationState)

    builder.add_node("triage_agent", triage_agent)
    builder.add_node("conversational_agent", lambda state: conversational_agent(model, tools, state))

    builder.set_entry_point("triage_agent")

    # Step 1 Routing
    builder.add_conditional_edges(
        "triage_agent", lambda x: x["next"], {
            "conversational_agent": "conversational_agent"
        }
    )


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

