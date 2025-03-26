import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
from chatbot import ChatManager
from langchain.schema import AIMessage, HumanMessage


# Initialize chat session and memory
def init_chat():
    """Initialize chat session and memory in Streamlit."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    if "chat_versions" not in st.session_state:
        st.session_state.chat_versions = []

    if "change_log" not in st.session_state:
        st.session_state.change_log = []

# Set up the Streamlit page configuration
def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("💬 OpenAI Chatbot with Memory Management")

# # Ensure we can handle AIMessage, HumanMessage, BotMessage correctly
# def handle_user_input():
#     """Process the user input and update the chat history."""
#     user_input = st.chat_input("Type your message...")

#     if user_input:
#         # If no versions exist yet (initial state), initialize the first version
#         if not st.session_state.chat_versions:
#             st.session_state.chat_manager.memory.chat_memory.messages = []  # Empty chat memory
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             initial_version = (timestamp, st.session_state.chat_manager.memory.chat_memory.messages)
#             st.session_state.chat_versions.append(initial_version)

#         # Get the latest version (the current one)
#         current_version = st.session_state.chat_versions[-1]  # The last item is the current version

#         # Copy the current version (don't modify the original one)
#         updated_messages = copy.deepcopy(current_version[1])  # Copy the list of messages from the current version

#         # Process the user input and get the bot's response
#         response = st.session_state.chat_manager.process_user_input(user_input)

#         # Append the user input and bot response to the copied version
#         updated_messages.append({"type": "human", "content": user_input})

#         # Handle the AIMessage response properly
#         if isinstance(response, AIMessage):
#             updated_messages.append({"type": "ai", "content": response.content})
#         else:
#             updated_messages.append({"type": "bot", "content": response['text']})

#         # Create a new version with the updated messages
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         new_version = (timestamp, updated_messages)

#         # Append the new version to the chat_versions list
#         st.session_state.chat_versions.append(new_version)

#         # Optionally rerun the app to display the updated state
#         st.rerun()

def handle_user_input():
    """Process the user input and update the chat history."""
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Process the user input and get the bot's response
        response = st.session_state.chat_manager.process_user_input(user_input)

        # Ensure we start fresh after restore
        latest_version = copy.deepcopy(st.session_state.chat_versions[-1][1]) if st.session_state.chat_versions else []

        # Append new messages
        latest_version.append({"type": "human", "content": user_input})
        latest_version.append({"type": "ai", "content": response.content if isinstance(response, AIMessage) else response['text']})

        # Create a fresh version after every new message
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.chat_versions.append((timestamp, latest_version))

        st.rerun()


# Display chat history
def display_chat_history():
    """Display the chat history with the chat messages."""
    st.subheader("Chat History")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_manager.memory.chat_memory.messages:
            if msg.type == "human":
                st.markdown(f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin-bottom: 5px;">👤 {msg.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #ffe0b2; padding: 10px; border-radius: 5px; margin-bottom: 5px;">🤖 {msg.content}</div>', unsafe_allow_html=True)

# Update version management to display messages correctly
def handle_version_management():
    """Handle version selection and display messages properly."""
    st.subheader("Manage Versions")

    # Get version timestamps for dropdown
    version_timestamps = [f"{i + 1}. {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)]

    # Dropdown to select a version
    selected_version = st.selectbox("Select Version", version_timestamps)

    if selected_version:
        version_index = version_timestamps.index(selected_version)
        selected_version_messages = st.session_state.chat_versions[version_index][1]

        st.write(f"Messages in {selected_version}:")
        for msg in selected_version_messages:
            # Check if msg has 'content' attribute, then display accordingly
            if isinstance(msg, dict) and 'content' in msg:
                st.write(f"{msg['type'].capitalize()}: {msg['content']}")
            else:
                st.write("Unknown message format. Please check your data.")






# Function to plot token usage from actual data (coming from ChatManager)
def show_usage_plots():
    """Displays both total token usage as a line plot and stacked token usage as a bar chart on the same plot."""
    fig, ax = plt.subplots(figsize=(10, 6))  # Single plot for both line and bar chart

    # Fetch the actual token usage history from the ChatManager object
    token_usage_history = st.session_state.chat_manager.token_usage_history

    # Ensure token usage data exists
    if not token_usage_history:
        st.warning("No token usage history available.")
        return

    # X positions for each set of bars (same as number of steps)
    index = np.arange(len(token_usage_history["steps"]))  
    bar_width = 0.5  # Width of each bar

    # Stacked bar chart for Prompt & Completion Tokens
    ax.bar(index, token_usage_history["individual_prompt_tokens"], bar_width, label="Prompt Tokens", color="green")
    ax.bar(index, token_usage_history["individual_completion_tokens"], bar_width, 
            bottom=token_usage_history["individual_prompt_tokens"], label="Completion Tokens", color="red")

    # Line plot for Total Token Usage (on top of the bar chart)
    ax.plot(index, token_usage_history["cumulative_tokens"], label="Total Tokens", color="blue", marker="o", linewidth=2)

    # Labels and titles
    ax.set_title("Token Usage (Total, Prompt, and Completion)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Tokens")
    ax.set_xticks(index)
    ax.set_xticklabels(token_usage_history["steps"])

    # Ensure the x-axis labels are aligned to the right
    plt.xticks(rotation=45, ha='right')

    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()  # Adjust layout to avoid overlaps

    # Display the plot in Streamlit
    st.pyplot(fig)

# # Function to display all versions
# def display_versions():
#     """Display all versions in the dropdown and show messages."""
#     st.subheader("Manage Versions")

#     # Ensure that the first version (Version 0) is in the list
#     if len(st.session_state.chat_versions) == 0:
#         timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         st.session_state.chat_versions.append((timestamp, []))  # Adding the empty version as Version 0

#     # Get version timestamps for the dropdown, including Version 0
#     version_timestamps = [f"Version {i + 1}: {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)]

#     # Dropdown to select a version
#     selected_version = st.selectbox("Select Version", version_timestamps)

#     if selected_version:
#         version_index = version_timestamps.index(selected_version)
#         selected_version_messages = st.session_state.chat_versions[version_index][1]

#         st.write(f"Messages in {selected_version}:")

#         # Fixing issue: Handling non-dict-like message objects correctly
#         for msg in selected_version_messages:
#             if isinstance(msg, dict) and 'content' in msg:
#                 st.write(f"{msg['type'].capitalize()}: {msg['content']}")
#             else:
#                 st.write("Unknown message format. Please check your data.")


def display_versions():
    """Display all versions in the dropdown and show messages."""
    st.subheader("Manage Versions")

    # Ensure that the first version (Version 0) is in the list
    if len(st.session_state.chat_versions) == 0:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.chat_versions.append((timestamp, []))  # Adding the empty version as Version 0

    # Get version timestamps for the dropdown, including Version 0
    version_timestamps = [f"Version {i + 1}: {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)]

    # Dropdown to select a version
    selected_version = st.selectbox("Select Version", version_timestamps)

    if selected_version:
        version_index = version_timestamps.index(selected_version)
        selected_version_messages = st.session_state.chat_versions[version_index][1]

        st.write(f"Messages in {selected_version}:")
        for msg in selected_version_messages:
            if isinstance(msg, dict) and 'content' in msg:
                st.write(f"{msg['type'].capitalize()}: {msg['content']}")
            else:
                st.write("Unknown message format. Please check your data.")




        # Move the button here for better accessibility
        if st.button("Restore Selected Version"):
            st.session_state.current_chat = selected_version_messages  # Update the current chat state
            st.session_state.chat_manager.restore_chat_version(version_index)
            st.success(f"Successfully restored to {selected_version}!")
            st.rerun()  # Use experimental_rerun for better state handling

# Main function to handle chat, memory, and token stats
def main():
    """Main function to display the app with chat, memory, and token usage plot."""
    setup_page()
    init_chat()

    # Layout setup: Two columns, left column for chat and right column for memory & stats
    col1, col2 = st.columns([60, 40])  # Left column 60% for the chat area, right 40% for memory & stats

    with col1:
        # Chat Section
        display_chat_history()

        # Custom styling for the input box (fixed at the bottom)
        st.markdown("""
        <style>
        .fixed-input { 
            position: fixed; 
            bottom: 20px; 
            width: 80%; 
            left: 10%; 
            background-color: white; 
            padding: 10px; 
            border-radius: 10px; 
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); 
            z-index: 9999;
        }
        </style>
        """, unsafe_allow_html=True)

        # Handle user input here, making sure the input is always at the bottom of the screen
        handle_user_input()

    with col2:
        # Tabs Section for Memory, Token Stats, and Version Management
        tab1, tab2, tab3 = st.tabs(["🗂 Version Management", "📈 Token Stats", "📝 Change Log"])

        with tab1:
            display_versions()

        with tab2:
            show_usage_plots()  # Plot the token usage in the Token Stats tab

        with tab3:
            # Placeholder for Change Log (not yet implemented)
            st.subheader("Change Log")
            st.write("### Recorded Changes:")
            st.write("This section will display the change logs once implemented.")

if __name__ == "__main__":
    main()