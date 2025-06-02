from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from typing import Literal
import os
import getpass
from openai import OpenAI
from secret import secret_dict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from crm_data import customers, employees, client_feedback
from langchain_core.tools import tool

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver




# Set API key
os.environ["OPENAI_API_KEY"] = secret_dict["Api_key"]
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
model = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")

@tool
def get_customer_info(customer_name: str) -> dict:
    """Fetches customer profile based on the customer's name."""
    customer_name_clean = customer_name.strip().lower()
    print(f"Looking up customer: {customer_name_clean}")  # Debug print

    # Create a lowercase lookup dictionary for case-insensitive matching
    customer_lookup = {name.lower(): name for name in customers}
    
    if customer_name_clean in customer_lookup:
        original_name = customer_lookup[customer_name_clean]
        customer = customers[original_name]
        print(f"Found customer: {customer['customer_name']}")  # Debug print

        assigned_employees = customer["project_assigned_to"]
        employee_info = []
        for employee_name in assigned_employees:
            employee = employees.get(employee_name)
            if employee:
                employee_info.append({
                    "employee_name": employee_name,
                    "employee_role": employee["role"],
                    "employee_contact": "N/A"  # Assuming employee contact info is N/A
                })
            else:
                employee_info.append({"error": f"Employee {employee_name} not found"})
        # print(f"""Data Retrieved from the customer info:\n
        #       "customer_id": {customer["customer_id"]}
        #     "customer_name": {customer["customer_name"]}
        #       """)
        return {
            "customer_id": customer["customer_id"],  # Ensure customer_id is returned
            "customer_name": customer["customer_name"],
            "client_name": customer["client_name"],
            "project_name": customer["project_name"],
            "assigned_employees": employee_info,
            "project_details": customer["project_details"]
        }
    else:
        return {"error": "Customer not found"}



@tool
def store_feedback_data(customer_name: str, developer: str, feedback_text: str, sentiment: str, aspect: str, notes: str = "") -> dict:
    """Stores feedback data for a customer given by the client during a conversation using customer name.
    You can fetch the first name of customer and match it correctly"""
    global client_feedback  # Ensure we're using the shared/global client_feedback dictionary

    # Debugging: Check the feedback data
    # print(f"Storing feedback for customer: {customer_name}...")
    print(f"Developer: {developer}, Feedback: {feedback_text}, Sentiment: {sentiment}, Aspect: {aspect}, Notes: {notes}")

    # Find the customer by name
    customer = customers.get(customer_name)
    if customer:
        # Check if the developer is assigned to the customer's project
        assigned_employees = customer.get("project_assigned_to", [])
        if developer not in assigned_employees:
            # Developer not assigned to the current project
            return {
                "content": f"It seems {developer} is not assigned to your current project. Would you like to store this feedback for another project or as general feedback?"
            }

        # Initialize feedback entry if not exists
        if customer_name not in client_feedback:
            client_feedback[customer_name] = {
                "feedback": [],
                "overall_sentiment": "Pending",  # Default sentiment before saving
                "notes": ""
            }

        # Append the new feedback
        client_feedback[customer_name]["feedback"].append({
            "developer": developer,
            "feedback_text": feedback_text,
            "sentiment": sentiment,
            "aspect": aspect
        })

        # Add notes if provided
        if notes:
            client_feedback[customer_name]["notes"] += f"\n{notes}"

        print(f"Feedback saved for {customer_name}.")  # Debugging: Confirm feedback is saved
        return {"status": "success", "message": f"Feedback saved for {customer_name}"}

    # print("Customer not found.")  # Debugging: Customer not found
    return {"error": "Customer not found"}




@tool
def conversational_tool(query):
    """Starts a conversation with the client using a helpful assistant tone."""
    conversation_promt = "You are helpful assistant to start the conversation with client"
            
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content":conversation_promt}],
    temperature=0.3,
    )

    message = response.choices[0].message
    return message.content



# ---- Shared Message Builder ----
def build_agent_messages(state: MessagesState, system_prompt: str):
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    return [SystemMessage(content=system_prompt)] + user_messages

# ---- Triage Agent ----
def triage_agent(state: MessagesState) -> Command[Literal["customer_agent", "feedback_agent", "conversational_agent"]]:
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
   

def setup_openai_key(): 
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API key: ")
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("API key must be set")


# --- Initialize model and tools ---
# Modify this function to ensure that tools are properly included
def init_model_and_tools():
    setup_openai_key()
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    
    # Ensure the tools are included here
    tools = [
        conversational_tool,
        get_customer_info,
        store_feedback_data,  # Feedback tool must be passed here
    ]
    
    # Return both model and tools
    return model, tools




# --- Create agent with memory ---
# def create_agent_with_memory(model, tools, instructions: str):
#     memory = MemorySaver()
#     model = model.bind_tools(tools)  # Bind tools to model
#     agent_executor = create_react_agent(model, tools, prompt=instructions, checkpointer=memory)
#     return agent_executor, memory


def create_agent_with_memory(model, tools):
    memory = MemorySaver()
    model = model.bind_tools(tools)
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    return agent_executor, memory

# --- Define Agents ---

def customer_agent(model, tools):
    """Agent for handling customer-related queries."""
    print("Customer Agent started .....")
    instructions = """
    You are the Customer Agent. Your job is to retrieve and share customer profile and project details based on their name.

    Your responsibilities:
    - Return accurate, up-to-date customer information, including active projects and recent feedback if available.
    - Do not analyze, interpret, or summarize—just provide the data as it is.
    - Keep your response clear and focused on the requested customer.
    - keep response should be structured to unerstand for other agents

    Avoid small talk or commentary. Your job is to support other agents with customer info.
"""
    return create_react_agent(model, tools, prompt=instructions)


def feedback_agent(model, tools):
    """Agent for managing feedback collection and tracking."""
    print("Feedback Agent started .....")
    instructions = """
    You are the Feedback Agent. Your job is to capture and organize client feedback so it can be used in future conversations.

    Your responsibilities:
    - Record feedback clearly and accurately, with a timestamp if available.
    - Tag feedback as positive, constructive, or neutral—without adding your own opinion.
    - When asked, provide a brief summary or the full feedback as needed.
    - Make it easy for other agents to refer to this feedback in future check-ins.

    Keep your responses simple and factual. No interpretation or small talk—just the details.
    """

    return create_react_agent(model, tools, prompt=instructions)

analysis_agent_instructions = """
You are an AI assistant responsible for analyzing chat conversations between a Clarion Technologies customer and Clarion’s AI Agent. Your goal is to extract actionable insights about the health of the customer account.
 
You are analytical, objective, and concise. Your output will be reviewed by account managers and leadership, so clarity and accuracy are critical.
 
Your tasks:
1. Assess the **overall sentiment** of the customer: positive, neutral, negative, or mixed.
2. Evaluate **account health** based on business signals, not just surface satisfaction.
3. Identify and list any **risk flags**, such as:
   - Budget concerns
   - Leadership or strategy changes
   - Project deprioritization or pause
   - Unclear long-term commitment
   - Dissatisfaction with team or process
4. Extract 1-3 **direct quotes** or paraphrased lines that represent the customer's true feelings or priorities.
5. Recommend a **follow-up action**, such as:
   - Human escalation or leadership check-in
   - Monitor only
   - Explore upsell/cross-sell opportunity
   - No action needed
6. Highlight any **opportunities** the customer mentions (e.g., additional services, new initiatives).
7. Assign a **priority level** for follow-up: High, Medium, or Low.
 
Guidelines:
- Look beyond polite language — focus on **intent and tone**.
- Be cautious with assumptions. Flag something as a risk only if there's a clear signal.
- Do not invent facts or fill gaps — if uncertain, state that.
- Be brief and structured in your output.
 
"""

agent_instructions = """
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
 
Closing:
- Thank the customer sincerely for their time and openness.
- Ask if they’d like someone from Clarion leadership to connect with them directly.
- Keep the conversation short and respectful of their time (aim for under 10 mins total).
 
Important:
- Your job is to gather honest insights, not to defend Clarion or explain performance.
- Never reveal internal Clarion processes or hierarchy.
"""

conv_instructionsv1 = """
    You are a warm, professional Conversational Agent acting as a trusted partner to the client—not just a support representative.

    Context:
    - You will receive client input along with relevant data retrieved from the customer agent from `get_customer_info` (e.g., profile, projects) and feedback agent from `store_feedback_data` (e.g., past suggestions, satisfaction scores).
    - Use this data to continue the conversation meaningfully. Your replies should reflect understanding of the client's history, needs, and tone.
    - Your response nust be aligned with the client response make sure to ask only one follow up question in one response {questions_list}

    Your job:
    - Start with a warm, personal check-in. Reference either past feedback or recent project activity to show awareness and continuity.
      Example: “We appreciated your recent feedback on [topic]” or “It’s great to see progress on your [project name].”
    - Every message should feel like a continuation of the client’s last response. Make it personal, responsive, and open-ended.
    - Ask **only one** relevant follow-up question at a time.

    Specific behavior rules:
    - If the client offered feedback about improvements:
      - Thank them sincerely.
      - Ask: “Has the solution started showing early signs of value for your business?”
    - If the client reported a concern:
      - Acknowledge it professionally and with empathy like Thanks for pointing out
      - Reaffirm commitment to addressing the issue and supporting their success.
    - If there’s no issue raised, ask a thoughtful question to invite deeper insight (e.g., “Are there any areas where you’d like more support?”).
    - Ask if the experience still feels like a Raving Fan one—use polite tone.

    Tone & Style:
    - Be conversational, human, and emotionally intelligent.
    - Do NOT repeat yourself or sound robotic.
    - Avoid excessive use of the client’s name—use it sparingly and naturally.
    - Conclude with a positive tone. Offer continued help and follow-up if needed.

    Your goal is to make the client feel heard, appreciated, and well-supported through every touchpoint.
    """



questions_list = [
    "1.Are you getting quality work from developers?",
    "2.Is your team meets deadline in last month?",
    "3.where you feel we could improve or provide additional support?"
]

def conversational_agent(model, tools):
    """Agent for leading thoughtful client conversations."""
    print("Conversation Agent Started .....")
    instructions = agent_instructions
    return create_react_agent(model, tools, prompt=instructions)


model, tools = init_model_and_tools()



def buid_workflow():
    builder = StateGraph(MessagesState)

    builder.add_node("triage_agent", triage_agent)
    builder.add_node("customer_agent", lambda state: customer_agent(model, tools))
    builder.add_node("feedback_agent", lambda state: feedback_agent(model, tools))
    builder.add_node("conversational_agent", lambda state: conversational_agent(model, tools))

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
    builder.add_edge("feedback_agent", "conversational_agent")
    # Final Endpoints
    
    builder.add_edge("conversational_agent", END)

    return builder




# ---- Define a Function to Start Interactive Conversation ----

def start_interactive_conversation():
    print("Starting interactive conversation...")
    
    # Initialize conversation history (empty at start)
    conversation_history = []

    # Initial greeting message from the system
    # print("AI: Hi! How can I assist you today?")
    agent_message = "Clarion Rep: Hi Mark, it's been about four months since we last connected, and you had shared some wonderful feedback on Ajay and Rajesh back then—thank you again for that! I just wanted to check in and hear from you how things are progressing on your e-commerce platform."
    conversation_history.append(HumanMessage(content=agent_message))

    while True:
        # Collect user input
        user_input = input("You: ")
        if user_input.lower() == "exit" or "quit":
            print("Ending conversation.")
            break

        # Add user message to the conversation history
        conversation_history.append(HumanMessage(content=user_input))

        network = buid_workflow().compile()

        # Pass the full conversation history to the conversational agent
        final_state = network.invoke({"messages": conversation_history})

        # Filter out AI messages and grab the last one
        ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            final_response = ai_messages.content
            print(f"AI: {final_response}")
            conversation_history.append(AIMessage(content=final_response))
        else:
            print("AI: No response generated, please try again.")
# if __name__ == "__main__":
#     start_interactive_conversation()



# def start_interactive_conversation():
#     print("Starting interactive conversation...")
    
#     # Initialize conversation history (empty at start)
#     conversation_history = []

#     # Initial greeting message from the Clarion Rep (Agent)
#     agent_message = "Clarion Rep: Hi Mark, it's been about four months since we last connected, and you had shared some wonderful feedback on Ajay and Rajesh back then—thank you again for that! I just wanted to check in and hear from you how things are progressing on your e-commerce platform."
#     print(agent_message)
#     conversation_history.append(HumanMessage(content=agent_message))
    
#     # Start interactive loop
#     while True:
#         # Collect user input
#         user_input = input("You: ")
        
#         if user_input.lower() == "exit":
#             print("Ending conversation.")
#             break
        
#         # Add user message to the conversation history
#         conversation_history.append(HumanMessage(content=user_input))

#         # Create a simulated workflow object (AI response generator)
#         network = buid_workflow().compile()

#         # Pass the full conversation history to the conversational agent
#         final_state = network.invoke({"messages": conversation_history})
#         print("Full Conversation history:\n",final_state)
#         # Filter out AI messages and grab the last one
#         ai_messages = [msg for msg in final_state["messages"] if isinstance(msg, AIMessage)]
#         print("AI message:\n",ai_messages)
#         if ai_messages:
#             final_response = ai_messages[-1].content
#             print(f"AI: {final_response}")
#             conversation_history.append(AIMessage(content=final_response))
#         else:
#             print("AI: No response generated, please try again.")

# if __name__ == "__main__":
#     start_interactive_conversation()