import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import Agent  
from langchain_community.tools.tavily_search import TavilySearchResults

# tools = None
tools = [TavilySearchResults(max_results=1)]


# Inicializa tu agente
agent = Agent(model_type="openai", tools=tools)


st.title("Agent Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# React to user input
if prompt := st.chat_input("User input"):
    # Create a HumanMessage and add it to chat history
    human_message = HumanMessage(content=prompt)
    st.session_state.messages.append(human_message)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Invoke the agent to get a list of AI messages
    response_messages = agent.invoke(st.session_state.messages)
    
    # Replace the entire chat history with the new response
    st.session_state.messages = response_messages

    # Display the last AI message in the chat
    if response_messages and isinstance(response_messages[-1], AIMessage):
        last_ai_message = response_messages[-1]
        st.chat_message("assistant").markdown(last_ai_message.content)