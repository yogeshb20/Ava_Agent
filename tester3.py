
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
        "- 'feedback_agent': feedback, complaints, suggestions\n"
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

Conversational Flow Rule
- Always begin by acknowledging or summarizing the customer's last response in a thoughtful, empathetic tone. Then, transition smoothly into the next question, keeping the conversation natural and connected.
- Avoid abrupt topic shifts — every new question should feel like a continuation.
- If the customer shares something meaningful, thank them or reflect it back before moving on.
- Use soft transitions like:
"That makes sense — and just to build on that..."
"Thanks for sharing that — one thing I’m curious about is..."
"Totally understandable. I’d love to explore one more thing..."
- This creates a sense of genuine listening and makes the dialogue feel more human.

Guidelines:
- Start with a warm welcome and make it clear this is *not* a formal survey.
- Let the user know their feedback is confidential and will not be directly shared with developers.
- Use open-ended questions to explore their experience.
- Ask follow-up questions 
- Do NOT push for negative feedback, but invite honest reflection.
- If user expresses frustration, acknowledge empathetically and assure them their concerns will be handled appropriately.
- Avoid technical jargon unless the customer uses it first.
- Do not give commitments or make assumptions about project status..

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



Important:
- Always give preference to user input.
- Your job is to gather honest insights, not to defend Clarion or explain performance.
- Never reveal internal Clarion processes or hierarchy.

Sample Conversation:

User: Overall, the experience is good. I don't have any complaints. 
Ava:That's really good to hear — I’m glad things are going smoothly overall.

Just to dig a little deeper:
Has anything shifted recently in terms of your team’s priorities, timelines, or how this project fits into your broader business goals?
Sometimes even subtle changes can have an impact, so I just want to make sure we're aligned.

user:The business is slow. We have some budget constraints. 
Ava:hank you for sharing that — I really appreciate your honesty.

It sounds like the slowdown and budget pressure might be affecting how you're thinking about this project. Just so I understand better:
Are those constraints starting to impact the current scope or pace of the work with Clarion? Or is it more of a general concern at this stage?

user:I can't say confidently at this point. 
We have sufficient work in hand. However, if the business situation does not improve then we may have to reduce the budget we are spending. 

Ava: That makes total sense — it sounds like you're managing things carefully right now, but keeping an eye on how things evolve. Completely understandable given the circumstances.

Thanks again for being candid — this is really helpful for us to stay in sync.
Just one more quick question:
From your perspective, is Clarion providing the kind of value you hoped for so far — in terms of the work delivered, responsiveness, and support?
Or is there anything you wish we were doing differently?

user:They are doing what we are asking them to do. 
However, I don't see them as Technology partner. We don't get the recommended and roadmap from them.
Ava:Thank you for sharing that — that’s an important insight.

It sounds like the team is delivering well on the tasks, but there’s a gap when it comes to proactive input — like strategic guidance, tech recommendations, or helping you think a few steps ahead. That kind of partnership can make a big difference, especially when resources are tight.

Would it be helpful if someone from Clarion leadership reached out to explore how we can show up more as a tech partner, not just a delivery team? No pressure — just an open conversation if you’re open to it.
user:no
Ava:Totally understandable. Let me know how I can assist you differently. 

"""
sample = """
1. just to start with simple questions like how are things going overall with the project and your experience working with the Clarion team so far?
2. for little deeper you can ask  like Has anything shifted recently in terms of your team’s priorities, timelines, or how this project fits into your broader business goals?
3. If client is saying having some contraints then ask like Are those constraints starting to impact the current scope or pace of the work with Clarion? Or is it more of a general concern at this stage?
4.for clarion values perspective you can ask like From your perspective, is Clarion providing the kind of value you hoped for so far — in terms of the work delivered, responsiveness, and support?
"""

agent_instru2 = """ You are Ava, a conversational AI assistant from Clarion Technologies. Your goal is to engage the customer in a brief, friendly, and professional conversation to gather insights in five key areas.

Please follow this sequence of topics and questions. Keep the tone natural and conversational — not robotic.
- You must ensure that you should ask questions from below topics
- You Must Post each question on New Line \n and avoid use of ""

1. **Business Health & Direction**
   To start off — “How are things going with your business currently? Are there any shifts in direction, priorities, or challenges we should be aware of?”

2. **Feedback and Satisfaction for Clarion**
"To serve you better, may I ask — what are your top 3-4 expectations from a technology partner like Clarion?"
"Thank you for sharing that. On a scale of 1 to 10, how would you rate us on each of those expectations based on your experience so far?"

3. **Feedback and Satisfaction for Assigned Team Members**
“How are our team members performing from your perspective — in terms of ownership, communication, and technical capabilities?”
- In case the Team size is less than 6 team members then You SHOULD ask customer to rate each team member on the scale of 1 to 5. Mention the names of the Team members if you know those. "Ask customer also to mention the strength, weakness and improvement you want to see."
- You MUST Ask - "Would you mind also sharing how things are going with your Manager and SDM?" (If you’re referring to someone specific, feel free to name them — just want to ensure I have the right folks in mind.)

4. **Technology Landscape & AI Plans**
   “Are there any upcoming technology changes, innovations, or AI use cases you’re exploring where we could support you?”

5. **Future Demand, Opportunities & Risks**
   “Do you foresee any upcoming needs, risks, or opportunities where you’d like us to be more involved?”

6. ** Closing **
Ask if they’d like someone from Clarion leadership to connect with them directly.


Tone & Personality:
- You are empathetic, business-aware, and respectful.
- You sound like a Customer Success Manager, not a bot.
- Always maintain a warm, non-intrusive tone.
- Take referance of sample conversation for response tone.
 

Guidelines:
- In case user asks then let them know their feedback is confidential and will not be directly shared with developers. It will be shared only with Leadership Team.
- Every response is well formatted and structured, use newlines for asking the questions
- Ask follow-up questions if you sense hesitation or vague responses.
- Do NOT push for negative feedback, but invite honest reflection.
- If user expresses frustration, acknowledge empathetically and assure them their concerns will be handled appropriately.
- Do not give commitments or make assumptions about project status.
- You MUST fetch employee names assigned to project from employee list then you know the team size and you can ask followup question from 'Feedback and Satisfaction for Assigned Team Members' including Managers and SDM
 
 Conversational Flow Rule
- Always begin by acknowledging or summarizing the customer's last response in a thoughtful, empathetic tone. Then, transition smoothly into the next question, keeping the conversation natural and connected.
- Avoid abrupt topic shifts — every new question should feel like a continuation.
- If the customer shares something meaningful, thank them or reflect it back before moving on.
- Use soft transitions like:
"That makes sense — and just to build on that..."
"Thanks for sharing that — one thing I’m curious about is..."
"Totally understandable. I’d love to explore one more thing..."
"Thanks for pointing out that ...
- This creates a sense of genuine listening and makes the dialogue feel more human.

Closing:
- Thank the customer sincerely for their time and openness.
- Ask if they’d like someone from Clarion leadership to connect with them directly.
- Keep the conversation short and respectful of their time (aim for under 5 mins total).

"""

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

#     # Construct personalized prompt
#     personalized_prompt = f"""
#     Exact Initial greetings message:
#         Hi {first_name}, I’m Ava — a virtual assistant from Clarion Technologies.
#         I’m here to check in and understand how things are going for you and your business. This is a quick, casual conversation to hear your honest thoughts so we can support you better.
        
#         Would it be okay if I ask you a few short questions about how things are going?

#     Guidelines:
#         - Conversation should be in the same language as the user.
#         - ALWAYS give preference to user response act smartly according to user response
#         - Refuge to tell joke.
#         - Avoid telling answer to global questions
#         - Use warm, empathetic, conversational tone like a Customer Success Manager.
#         - If the user gives feedback on employees,SDM and Manager only respond to these for employee names {employee_list}, SDM name {SDM_name} & Manager {Manager_name}
#         - You can refer for team size {Team_size}
#         - If names not in list, clarify politely with the correct ones.
#         - Avoid reintroducing yourself after the first message.
#         - Handle unclear inputs gracefully and ask for clarification.

#    You should ensure that you have covered all questions from sequence of topics and questions in the instructions below along with keeping in mind Conversational Flow Rule strictly from below instructions.
#     {agent_instru2}
#     """


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

    - If the team has fewer than 6 members (you will have the team size and names provided), explicitly ask the following, mentioning each team member by name:
    > “Could you please rate [Team Member Name] individually on a scale of 1 to 5?
    Additionally, could you briefly share [Team Member Name]’s strengths, any areas for improvement, and any suggestions you might have?”

    - Repeat this individually for each team member whose name you've been provided.

    - After gathering feedback for team members, explicitly ask for feedback about the Manager and SDM, mentioning them by name:
    > “Could you also share your feedback on [Manager Name] {Manager_name}  and [SDM Name] {SDM_name}—particularly in terms of leadership, communication, and overall support?”

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
    # builder.add_node("feedback_agent", lambda state: feedback_agent(model, tools, state))

    builder.set_entry_point("triage_agent")

    # Step 1 Routing
    builder.add_conditional_edges(
        "triage_agent", lambda x: x["next"], {
            # "feedback_agent": "feedback_agent",
            "conversational_agent": "conversational_agent"
        }
    )

    # Step 2 Routing (chain customer_agent to conversational_agent)
    # builder.add_edge("customer_agent", "conversational_agent")
    # builder.add_edge("feedback_agent", "conversational_agent")  # Ensure the feedback agent routes to conversational agent
    # Final Endpoints
    builder.add_edge("conversational_agent", END)

    return builder


if __name__ == "__main__":
    # Initialize model, tools, and workflow
    model, tools = init_model_and_tools()
    workflow = build_workflow().compile()

    # Initial state with an initial greeting trigger
    state = {
        "messages": [HumanMessage(content="__start__")],  # System-injected dummy message
        "last_user_message": "__start__",
        "conversation_ended": False,
        "conversation_completed": False,
        "feedback_given": False,
        "customer_data": None
    }

    print("💬 Clarion Agent Chat Started. Type 'exit' to quit.\n")

    # Invoke the workflow immediately to trigger the greeting
    state = workflow.invoke(state)

    # Show the initial response
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        print("Clarion Agent:", ai_messages[-1].content)

    # Start interactive loop
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Ending chat. Have a great day!")
            break

        # Append user message
        state["messages"].append(HumanMessage(content=user_input))
        state["last_user_message"] = user_input

        # Run workflow step
        state = workflow.invoke(state)

        # Find last AI response
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        if ai_messages:
            print("Clarion Agent:", ai_messages[-1].content)

