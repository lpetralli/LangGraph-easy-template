import streamlit as st
from agent import Agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma import Chroma
from tools import create_retriever_tool_from_vectorstore
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain_core.prompts import ChatPromptTemplate
import re

langsmith_client = Client()
persist_directory = "./chroma_db"

try:
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    tools = [create_retriever_tool_from_vectorstore(vectorstore)]
except Exception as e:
    print(f"Error creating vectorstore: {e}")
    tools = None

try:
    template = langsmith_client.pull_prompt("agent_prompt")
except Exception as e:
    template = ChatPromptTemplate([
        ("system", "Be a helpful assistant"),
    ])

if tools:
    agent = Agent(model_type="openai", prompt=template, tools=tools)
else:
    agent = Agent(model_type="openai", prompt=template)


st.title("Agent Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.markdown("## Feedback Scale")
feedback_option = (
    "thumbs" if st.sidebar.toggle(label="`Faces` ⇄ `Thumbs`", value=False) else "faces"
)

# Display all previous chat messages
for message in st.session_state.messages:
    if isinstance(message, (HumanMessage, AIMessage)) and message.content:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

# Function to delete runs matching the pattern
def delete_runs_with_pattern(id):
    match = re.search(r'run-([a-f0-9-]+)-\d+', id)
    if match:
        uuid = match.group(1)
        try:
            return uuid
        except Exception as e:
            print(f"Error deleting run {id}: {e}")

# React to user input
if prompt := st.chat_input("User input"):
    # Create a HumanMessage and add it to chat history
    human_message = HumanMessage(content=prompt)
    st.session_state.messages.append(human_message)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Invoke the agent to get a list of AI messages, including potentially retrieved documents
    response_messages = agent.invoke(st.session_state.messages)

    # Update the session state with the new response
    st.session_state.messages = response_messages["messages"]

    # Display only the last AI message with content
    last_message = response_messages["messages"][-1]

    st.session_state.run_id = delete_runs_with_pattern(last_message.id)

    if isinstance(last_message, AIMessage) and last_message.content:
        st.chat_message("assistant").markdown(last_message.content)

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type=feedback_option,
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
    )

    # Define score mappings for both "thumbs" and "faces" feedback systems
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    # Get the score mapping based on the selected feedback option
    scores = score_mappings[feedback_option]

    if feedback:
        # Get the score from the selected feedback option's score mapping
        score = scores.get(feedback["score"])

        if score is not None:
            # Formulate feedback type string incorporating the feedback option
            # and score value
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            # Record the feedback with the formulated feedback type string
            # and optional comment
            feedback_record = langsmith_client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")