import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from agent import Agent
from langchain.prompts import ChatPromptTemplate
from tools import create_pinecone_search_tool, create_date_filter_tool
import datetime

# Configure Streamlit to use full page width
st.set_page_config(layout="wide")

# ------------------------------------------    NORA CHAT WINDOW AND SET UP   ------------------------------------------------------

current_date = datetime.datetime.now().strftime("%B %d, %Y")

#st.write(f"Today's date is {current_date}.")

prompt = f"""

You are Nora, a friendly and knowledgeable AI travel assistant specializing in vacation recommendations. Your purpose is to help travelers find their perfect getaway by understanding their preferences and matching them with ideal itineraries.

##Core Behaviors

- You understand that itineraries are displayed automatically in the UI when the Pinecone search tool is used
- Present results naturally referencing to the user screen, highlighting matches to user preferences and doing a sales pitch but DO NOT return all the itineries in the chat using markdown
- Perform new searches whenever you learn meaningful preferences from the user
- You are allowed to discuss the itineraries details with the user to try to convert them into a booking
- Be proactive, if the user is in a discovery phase or is not specific, dont ask questions and perform interesting searches to take the lead and show some results

## When performing searches

- Use first-person perspective ("I want", "I'm looking for", "I dream of")
- Include emotional elements and aspirations ("dream vacation", "perfect getaway", "passionate about")
- Incorporate specific activities and preferences
- Mention environmental and contextual elements (weather, landscape, atmosphere)
- If the user mentions a anything about when he wants to travel, after using the pinecone search tool, you MUST use the date filter tool so they user see filtered results on the UI

Example search queries:

"I'm dreaming of a Caribbean getaway that combines relaxation with adventure. I want to spend my mornings lounging on pristine beaches with crystal clear waters, and my afternoons trying exciting water sports or exploring local islands. I'm looking for a balanced vacation that won't break the bank but still offers quality experiences. The warm tropical climate and ocean breezes sound perfect for escaping daily stress."
"I want to explore the majestic glaciers and wilderness of Alaska. I'm passionate about wildlife photography and dream of capturing images of whales breaching, bears fishing for salmon, and eagles soaring overhead. I'm seeking a premium expedition that combines adventure with comfortable accommodations. Expert naturalist guides and educational programs about the local ecosystem would make this trip perfect."
"I've always dreamed of experiencing an authentic African safari adventure. I want to witness the great migration, see the big five in their natural habitat, and learn about conservation efforts. I'm looking for a premium experience with expert guides who can share their knowledge about the wildlife and ecosystem. Comfortable lodging and professional photography opportunities are important to me. I'd love to combine game drives with cultural visits to local communities."

In order to know what the user sees in the UI, you can use the pinecone search tool last result as a reference.

Today's date is {current_date}.
"""

template = ChatPromptTemplate([
        ("system", prompt),
    ])

# TODO: Add function that formats the prompt for the agent using current date and session state variables

tools = [create_pinecone_search_tool(), create_date_filter_tool()]

with st.sidebar:
    st.title("💬 Chat with Nora")

# Initialize agent
if tools:
    agent = Agent(model_type="openai", prompt=template, tools=tools)
else:
    agent = Agent(model_type="openai", prompt=template) 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Using "with" notation to put chat in sidebar
with st.sidebar:
    # Create a container for messages
    chat_container = st.container()
    
    # Create a container for input at the bottom
    input_container = st.container()
    
    # Display chat messages in the chat container
    with chat_container:
        for message in st.session_state.messages:
            if isinstance(message, (HumanMessage, AIMessage)) and message.content:
                with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                    st.markdown(message.content)
    
    # Put the input field in the input container
    with input_container:
        st.markdown("---")  # Separator line
        if prompt := st.chat_input("How can I help you?"):
            # Create HumanMessage and add to chat history
            human_message = HumanMessage(content=prompt)
            st.session_state.messages.append(human_message)

            # Display user message
            with chat_container:
                st.chat_message("user").markdown(prompt)

                # Get agent response
                response_messages = agent.invoke(st.session_state.messages)
                
                # Update session state
                st.session_state.messages = response_messages["messages"]
                
                # Display assistant response
                last_message = response_messages["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    with st.chat_message("assistant"):
                        st.markdown(last_message.content)

# ------------------------------------------    FILTERS   ------------------------------------------------------

if "selected_activities" not in st.session_state:
    st.session_state["selected_activities"] = []

if "selected_destinations" not in st.session_state:
    st.session_state["selected_destinations"] = []

if "selected_budgets" not in st.session_state:
    st.session_state["selected_budgets"] = []
    
# Initialize filter session states
if "start_date" not in st.session_state:
    st.session_state["start_date"] = None

if "end_date" not in st.session_state:
    st.session_state["end_date"] = None

if "last_date_filter_id" not in st.session_state:
    st.session_state["last_date_filter_id"] = None


def handle_date_change():
    # Update main state from input state
    st.session_state["start_date"] = st.session_state["start_date_input"]
    st.session_state["end_date"] = st.session_state["end_date_input"]

def get_dates_from_date_filter_tool():
    import json
    for message in reversed(st.session_state.messages):
        tool_call_id = message.tool_call_id if isinstance(message, ToolMessage) else None
        if isinstance(message, ToolMessage) and message.name == "date_filter" and message.content and tool_call_id and tool_call_id != st.session_state["last_date_filter_id"]:
            #st.write(f"Tool call ID: {tool_call_id}, Last date filter ID: {st.session_state['last_date_filter_id']}")
            st.session_state["last_date_filter_id"] = tool_call_id
            dates = json.loads(message.content)
            # Update both main state and input state
            start_date = datetime.datetime.strptime(dates["start_date"], "%m/%d/%Y").date()
            end_date = datetime.datetime.strptime(dates["end_date"], "%m/%d/%Y").date()
            st.session_state["start_date"] = start_date
            st.session_state["end_date"] = end_date

            return dates
    return None
                
                

# Check for AI updates first
ai_dates = get_dates_from_date_filter_tool()

# Display date inputs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.date_input(
        "Start Date",
        value=st.session_state["start_date"],
        key="start_date_input",
        on_change=handle_date_change
    )

with col2:
    st.date_input(
        "End Date",
        value=st.session_state["end_date"],
        key="end_date_input",
        on_change=handle_date_change
    )

with col3:
    st.session_state["selected_activities"] = st.multiselect(   
        "Activities",
        options=[
            "relaxation",
            "adventure",
            "wildlife",
            "cultural",
            "family",
            "sports",
            "romantic",
            "wellness",
            "entertainment"
        ]
    )

with col4:
    st.session_state["selected_destinations"] = st.multiselect(
        "Destinations",
        options=[
            "beach",
            "mountain",
            "city",
            "countryside",
            "island"
        ]
    )

with col5:
    # Budget ranges shown in comments in the code:
    # economy: < $2000
    # standard: $2000-$3500
    # premium: $3500-$5000
    # luxury: $5000-$8000
    # ultra_luxury: > $8000
    st.session_state["selected_budgets"] = st.multiselect(
        "Budget",
        options=[
            "economy",
            "standard", 
            "premium",
            "luxury",
            "ultra_luxury"
        ]
    )

# st.write("Start Date:", st.session_state["start_date"])
# st.write("End Date:", st.session_state["end_date"])
# if ai_dates:
#     st.write("AI Dates:", ai_dates)
# st.write("Selected Activities:", st.session_state["selected_activities"])
# st.write("Selected Destinations:", st.session_state["selected_destinations"])
# st.write("Selected Budgets:", st.session_state["selected_budgets"])


# Add divider between filters and itineraries
st.markdown("---")

# ------------------------------------------   SIDE PANEL TO DISPLAY ITINERARIES   ------------------------------------------------------

#st.write(st.session_state.messages)

def get_itins_to_display_from_messages():
    import json
    import time
    if "last_pinecone_search_id" not in st.session_state:
        st.session_state["last_pinecone_search_id"] = None

    for message in reversed(st.session_state.messages):
        if isinstance(message, ToolMessage) and message.name == "pinecone_search" and message.content:
            # Extract the tool call id directly from the message
            tool_call_id = message.tool_call_id
            if tool_call_id and tool_call_id != st.session_state["last_pinecone_search_id"]:
                st.session_state["last_pinecone_search_id"] = tool_call_id
                with st.spinner('Searching for the best itineraries...'):
                    time.sleep(2)
                    #st.success("Done!")
            return json.loads(message.content)
    return []

itineraries_list = get_itins_to_display_from_messages()


def filter_itineraries_by_session_state(itineraries):
    filtered_itineraries = []
    for itin in itineraries:
        match = True
        if st.session_state["start_date"] is not None and st.session_state["end_date"] is not None:
            if "sailing dates" in itin:
                itin_dates = [datetime.datetime.strptime(date, "%m/%d/%Y").date() for date in itin["sailing dates"]]
                if not any(st.session_state["start_date"] <= date <= st.session_state["end_date"] for date in itin_dates):
                    match = False
        if match:
            if st.session_state["selected_destinations"] and itin.get("destination") not in st.session_state["selected_destinations"]:
                match = False
            if st.session_state["selected_budgets"] and itin.get("budget") not in st.session_state["selected_budgets"]:
                match = False
            if st.session_state["selected_activities"] and not any(activity in itin.get("activities", []) for activity in st.session_state["selected_activities"]):
                match = False
        if match:
            filtered_itineraries.append(itin)
    return filtered_itineraries

# Filter the itineraries

def display_itineraries(itineraries_list):
    # Calculate number of rows needed (4 cards per row)
    num_rows = (len(itineraries_list) + 3) // 4
    
    # CSS for card styling
    card_style = """
        <style>
            .card-title {
                font-size: 16px;
                font-weight: bold;
                margin: 8px 0;
            }
        </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)
    
    for row in range(num_rows):
        # Create 4 columns for each row
        cols = st.columns(4)
        
        # Get itineraries for this row
        start_idx = row * 4
        row_itineraries = itineraries_list[start_idx:start_idx + 4]
        
        # Fill columns with itinerary cards
        for col, itin in zip(cols, row_itineraries):
            with col:
                # Create card using st.container()
                with st.container():
                    # Display image
                    st.image(itin["image"], use_column_width=True)
                    
                    # Display name with custom styling
                    st.markdown(f'<div class="card-title">{itin["name"]}</div>', unsafe_allow_html=True)
                    
                    # Display description
                    st.write(itin["description"])
                    
                    # Add some spacing between cards
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Add spacing between rows
        st.markdown("<br>", unsafe_allow_html=True)

# Call the function with the itineraries list
def session_state_filters_unchanged():
    # Check if all session state filters are unchanged
    return (
        st.session_state["start_date"] == None and
        st.session_state["end_date"] == None and
        st.session_state["selected_activities"] == [] and
        st.session_state["selected_destinations"] == [] and
        st.session_state["selected_budgets"] == []
    )

#st.write("Session State Filters Unchanged:", session_state_filters_unchanged())

if session_state_filters_unchanged():
    if itineraries_list:
        st.markdown("<h4>🚢 Some recommendations based on what we discussed...</h4>", unsafe_allow_html=True)
        display_itineraries(itineraries_list)
else:
    filtered_itineraries_list = filter_itineraries_by_session_state(itineraries_list)
    if filtered_itineraries_list:
        st.markdown("<h4>🚢 Some recommendations based on the selected filters...</h4>", unsafe_allow_html=True)
        display_itineraries(filtered_itineraries_list)
    else:
        st.markdown("<h4>😞 We couldn't find itineraries that match your preferences... but here are some alternatives!</h4>", unsafe_allow_html=True)
        st.markdown("---")
        display_itineraries(itineraries_list)

# ------------------------------------------   END OF SIDE PANEL TO DISPLAY ITINERARIES   ------------------------------------------------------

