import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
from chatbot import ChatManager
from langchain.schema import AIMessage, HumanMessage


def get_message_content(msg):
    """Extracts the content from a message, handling both dict and object formats."""
    if isinstance(msg, dict):
        return msg.get("content", "No content")
    elif hasattr(msg, "content"):
        return msg.content
    return "Unknown format"

# Initialize chat session and memory
def init_chat():
    """Initialize chat session and memory in Streamlit."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    if "chat_versions" not in st.session_state:
        st.session_state.chat_versions = []

    if "change_log" not in st.session_state:
        st.session_state.change_log = []
    
    if "restored" not in st.session_state:
        st.session_state.restored = False  # Default state

    if "reset_tokens" not in st.session_state:
        st.session_state.reset_tokens = False  # Default state

# Set up the Streamlit page configuration
def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("💬 OpenAI Chatbot with Memory Management")

def handle_user_input():
    """Process the user input and update the chat history."""
    user_input = st.chat_input("Type your message...")

    if user_input:

        if st.session_state.reset_tokens:  # ✅ Runs only once after restore
            st.session_state.chat_manager._track_token_usage(is_reset=True)  
            st.session_state.reset_tokens = False  # 🚨 Reset flag after handling

        # 🔹 Capture the state before processing
        latest_version = copy.deepcopy(st.session_state.chat_manager.memory.chat_memory.messages)

        # 🔹 Process user input (this modifies memory directly)
        response = st.session_state.chat_manager.process_user_input(user_input)

        # 🔹 Capture the state *after* processing
        updated_version = copy.deepcopy(st.session_state.chat_manager.memory.chat_memory.messages)

        # Convert to dictionary format for versioning using the helper function
        latest_version = [{"type": "human", "content": get_message_content(msg)} if isinstance(msg, HumanMessage) 
                        else {"type": "ai", "content": get_message_content(msg)} for msg in latest_version]

        updated_version = [{"type": "human", "content": get_message_content(msg)} if isinstance(msg, HumanMessage) 
                        else {"type": "ai", "content": get_message_content(msg)} for msg in updated_version]


        # If restored, save a "checkpoint" version before appending new messages
        if st.session_state.restored:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.chat_versions.append((timestamp, latest_version))  # Save the restore state
            st.session_state.restored = False  # Reset flag

        # Save the final updated version
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.chat_versions.append((timestamp, updated_version))

        st.rerun()

def display_chat_history():
    """Display the chat history with the chat messages."""
    st.subheader("Chat History")
    chat_container = st.container()

    with chat_container:

        for msg in st.session_state.chat_manager.memory.chat_memory.messages:
            msg_content = get_message_content(msg)
            msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

            # Debug output
            print(f'Message: {msg}, Content: {msg_content}, Type: {msg_type}')

            if msg_type == "human":
                st.markdown(f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin-bottom: 5px;">👤 {msg_content}</div>', unsafe_allow_html=True)
            elif msg_type == "ai":
                st.markdown(f'<div style="background-color: #ffe0b2; padding: 10px; border-radius: 5px; margin-bottom: 5px;">🤖 {msg_content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 5px;">⚠️ Unknown Message Type: {msg_content}</div>', unsafe_allow_html=True)

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
            # Use the get_message_content helper function to retrieve content
            msg_content = get_message_content(msg)  # Retrieve content uniformly
            msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

            # Display the message appropriately
            if msg_type == "human":
                st.write(f"👤 {msg_content}")
            elif msg_type == "ai":
                st.write(f"🤖 {msg_content}")
            else:
                st.write(f"⚠️ Unknown Message Type: {msg_content}")




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

def display_versions():
    """Display all versions in the dropdown and allow restoring chat states."""
    st.subheader("Manage Versions")

    if not st.session_state.chat_versions:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        st.session_state.chat_versions.append((timestamp, []))  # Ensure at least one empty version

    version_timestamps = [f"Version {i + 1}: {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)]
    selected_version = st.selectbox("Select Version", version_timestamps)

    if selected_version:
        version_index = version_timestamps.index(selected_version)
        selected_version_messages = st.session_state.chat_versions[version_index][1]

        st.write(f"Messages in {selected_version}:")
        for msg in selected_version_messages:
            # Use the get_message_content helper function to retrieve content
            msg_content = get_message_content(msg)  # Retrieve content uniformly
            msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

            # Display the message appropriately
            if msg_type == "human":
                st.write(f"👤 {msg_content}")
            elif msg_type == "ai":
                st.write(f"🤖 {msg_content}")
            else:
                st.write(f"⚠️ Unknown Message Type: {msg_content}")

        # Restore Button Logic (Updated)
        if st.button("Restore Selected Version"):
            st.session_state.restored = True  # 🔹 Mark as restored
            st.session_state.reset_tokens = True  # ✅ Indicate tokens should reset

            # **Ensure old messages are cleared before restoring**
            st.session_state.chat_manager.memory.chat_memory.messages.clear()

            # **Deep copy to prevent reference issues**
            restored_messages = copy.deepcopy(selected_version_messages)
            st.session_state.chat_manager.memory.chat_memory.messages = restored_messages

            # **Create a new version with restored messages**
            new_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.chat_versions.append((new_timestamp, restored_messages))


            st.success(f"Successfully restored to {selected_version}! A new version has been created.")
            st.rerun()

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