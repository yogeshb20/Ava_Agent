from agent import *





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




# def build_workflow():
#     builder = StateGraph(MyConversationState)

#     builder.add_node("triage_agent", triage_agent)

#     builder.add_node("conversational_agent", lambda state: conversational_agent(model, tools, state))

#     builder.add_node("customer_agent", lambda state: customer_agent(model, tools).invoke(state))

#     builder.add_node("feedback_agent", lambda state: feedback_agent(model, tools, state))

#     builder.add_node("sentiment_agent", lambda state: sentiment_agent(model, tools).invoke(state))

#     builder.add_node("conversation_complete", lambda state: check_conversation_complete(state))

#     builder.set_entry_point("triage_agent")

#     # Step 1 Routing — triage only chooses between customer_agent or conversational_agent
#     builder.add_conditional_edges(
#         "triage_agent", lambda x: x["next"], {
#             "customer_agent": "customer_agent",
#             "conversational_agent": "conversational_agent",
#             "sentiment_agent":"sentiment_agent"
#         }
#     )

#     # Step 2 — customer info first, then conversation
#     builder.add_edge("customer_agent", "conversational_agent")

#     builder.add_edge("sentiment_agent", "conversational_agent")
#     # Step 3 — After the conversation finishes, check if it’s complete
#     builder.add_edge("conversational_agent", "conversation_complete")

#     # Step 4 — Conditionally go to feedback agent or remain in conversation_complete
#     builder.add_conditional_edges(
#         "conversation_complete", 
#         lambda x: x["next"], 
#         {
#             "feedback_agent": "feedback_agent",
#             "__end__": END
#         }
#     )

#     # Step 5 — end after feedback
#     builder.add_edge("feedback_agent", "triage_agent")

#     return builder