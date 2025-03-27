import time
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import copy
from chatbot_v2 import ChatManager
from langchain.schema import AIMessage, HumanMessage
import logging

# 🏷️ Set Streamlit Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Chatbot", layout="wide")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ChatApp:
    """
    💬 A Streamlit-based chatbot with:
        - ✅ Memory management
        - ✅ Version control
        - ✅ Token usage tracking
    """

    def __init__(self):
        """
        🔧 Initializes session state variables and sets up the chat environment.
        
        ✅ Ensures all required session state variables are initialized.
        ✅ Prevents reinitialization on every rerun.
        """
        
        # 🗄️ Initialize Chat Manager
        if "chat_manager" not in st.session_state:
            st.session_state.chat_manager = ChatManager()  # Ensure safe initialization
        self.chat_manager = st.session_state.chat_manager  # Reference from session state
        
        # 🗂️ Stores different versions of the chat for version control
        if "chat_versions" not in st.session_state:
            st.session_state.chat_versions = []

        # 📝 Keeps track of changes (future feature)
        if "change_log" not in st.session_state:
            st.session_state.change_log = []

        # 🔄 Tracks if a chat version was restored
        if "restored" not in st.session_state:
            st.session_state.restored = False

        # 🔄 Tracks if token usage needs resetting
        if "reset_tokens" not in st.session_state:
            st.session_state.reset_tokens = False


    def setup_page(self):
        """
        📌 Configures the Streamlit page settings (title, layout, and design).

        ✅ Sets the page title and layout.
        ✅ Displays the main header.
        ✅ (Optional) Additional styling for UI improvements.
        """
        
        # 🏷️ Set Streamlit Page Configuration
        # st.set_page_config(page_title="Chatbot", layout="wide")

        # 🏆 Display App Title
        st.title("💬 OpenAI Chatbot with Memory Management")
        
        # 🎨 Custom Styling (Optional)
        st.markdown("""
                    <style>
                        .block-container {
                            padding-top: 3rem !important; /* Adjust if needed */
                        }
                    </style>
                    """, unsafe_allow_html=True)

    def get_message_content(self, msg):
        """
        📩 Extracts the content from a message object, supporting both dictionary and object formats.
        
        - ✅ Works with both `dict` and message objects (AIMessage/HumanMessage).
        - 🛑 Returns `"Unknown format"` if the structure is unrecognized.
        
        :param msg: The message object (dict or AIMessage/HumanMessage instance)
        :return: Extracted message content as a string
        """
        if isinstance(msg, dict):
            return msg.get("content", "No content")
        elif hasattr(msg, "content"):
            return msg.content
        return "Unknown format"



    def handle_user_input(self):
        """
        ⌨️ Processes user input, updates chat history, tracks token usage, logs key steps, and manages chat versions.
        """
        user_input = st.chat_input("Type your message...")

        if user_input:
            logging.info(f"User input received: {user_input}")

            # ✅ Step 2: Track token reset state
            if st.session_state.reset_tokens:
                logging.info("Token reset detected. Updating token usage tracking.")
                st.session_state.chat_manager._track_token_usage(is_reset=True)
                st.session_state.reset_tokens = False  # 🚨 Reset flag after handling

            # 🔹 Step 3: Capture the chat state before processing
            latest_version = copy.deepcopy(st.session_state.chat_manager.memory.chat_memory.messages)
            logging.info("Captured chat state before processing user input.")

            # 🔹 Step 4: Process user input (this modifies memory directly)
            response = st.session_state.chat_manager.process_user_input(user_input)
            logging.info("User input processed. AI response generated.")

            # 🔹 Step 5: Capture the chat state after processing
            updated_version = copy.deepcopy(st.session_state.chat_manager.memory.chat_memory.messages)
            logging.info("Captured chat state after processing user input.")

            # 🔄 Step 6: Convert messages into structured format
            latest_version = [
                {"type": "human", "content": self.get_message_content(msg)} if isinstance(msg, HumanMessage) 
                else {"type": "ai", "content": self.get_message_content(msg)} 
                for msg in latest_version
            ]
            
            updated_version = [
                {"type": "human", "content": self.get_message_content(msg)} if isinstance(msg, HumanMessage) 
                else {"type": "ai", "content": self.get_message_content(msg)} 
                for msg in updated_version
            ]

            logging.info("Converted chat messages to structured format.")

            # 🕒 Step 7: Save a checkpoint if a version was restored
            if st.session_state.restored:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                st.session_state.chat_versions.append((timestamp, latest_version))  # Save the restore state
                st.session_state.restored = False  # Reset flag
                logging.info(f"Chat version restored at {timestamp}. Checkpoint saved.")

            # 📌 Step 8: Save the final updated version
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.chat_versions.append((timestamp, updated_version))
            logging.info(f"New chat version saved at {timestamp}.")

            # 🔄 Refresh the Streamlit app state
            logging.info("Triggering Streamlit app rerun.")
            st.rerun()

    def display_chat_history(self):
        """
        🗂️ Displays formatted chat history, distinguishing between human and AI messages.

        Workflow:
        1️⃣ Create a Streamlit container to display the chat history.
        2️⃣ Loop through stored chat messages from session state.
        3️⃣ Extract content and message type from each message.
        4️⃣ Apply appropriate styling based on whether the message is from the human or AI.
        5️⃣ Display the formatted messages using Streamlit's markdown.
        """

        st.subheader("📜 Chat History")
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_manager.memory.chat_memory.messages:
                msg_content = self.get_message_content(msg)
                msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

                # 🛠️ Debug Output (Optional: Remove in production)
                print(f'Message: {msg}, Content: {msg_content}, Type: {msg_type}')

                # 🎨 Apply styling based on message type
                if msg_type == "human":
                    st.markdown(
                        '<div style="background-color: #e0f7fa; padding: 10px; border-radius: 8px; margin-bottom: 5px;">'
                        f'👤 {msg_content}</div>',
                        unsafe_allow_html=True,
                    )
                elif msg_type == "ai":
                    st.markdown(
                        '<div style="background-color: #ffe0b2; padding: 10px; border-radius: 8px; margin-bottom: 5px;">'
                        f'🤖 {msg_content}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div style="background-color: #f8d7da; padding: 10px; border-radius: 8px; margin-bottom: 5px;">'
                        f'⚠️ Unknown Message Type: {msg_content}</div>',
                        unsafe_allow_html=True,
                    )


    def handle_version_management(self):
        """
        📜 Handles version selection and allows users to browse past conversations.

        Workflow:
        1️⃣ Retrieves all stored chat versions.
        2️⃣ Displays a dropdown menu to select a previous version.
        3️⃣ Extracts and displays the corresponding chat messages.
        4️⃣ Uses helper functions to format and categorize messages.
        """

        st.subheader("📂 Manage Chat Versions")

        # 📌 Get version timestamps for dropdown menu
        version_timestamps = [
            f"{i + 1}. {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)
        ]

        # 📌 Dropdown to select a specific chat version
        selected_version = st.selectbox("🔽 Select Version", version_timestamps, index=len(version_timestamps) - 1)

        if selected_version:
            # Extract selected version index
            version_index = version_timestamps.index(selected_version)
            selected_version_messages = st.session_state.chat_versions[version_index][1]

            st.write(f"📝 Messages in {selected_version}:")

            for msg in selected_version_messages:
                # ✅ Extract content and type
                msg_content = self.get_message_content(msg)  
                msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

                # 🎨 Display messages with formatting
                if msg_type == "human":
                    st.markdown(f'👤 **User:** {msg_content}')
                elif msg_type == "ai":
                    st.markdown(f'🤖 **AI:** {msg_content}')
                else:
                    st.markdown(f'⚠️ **Unknown:** {msg_content}')


    def display_versions(self):
        """
        🗂️ Displays stored chat versions and allows restoration of previous states, with logging.
        """

        st.subheader("📂 Manage Chat Versions")

        # ✅ Ensure at least one chat version exists
        if not st.session_state.chat_versions:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.chat_versions.append((timestamp, []))
            logging.info("No chat versions found. Initialized with an empty version.")

        # 📌 Populate dropdown with version timestamps
        version_timestamps = [
            f"Version {i + 1}: {timestamp}" for i, (timestamp, _) in enumerate(st.session_state.chat_versions)
        ]

        # 🔽 Dropdown for selecting a chat version
        selected_version = st.selectbox("🔄 Select a Version to View or Restore", version_timestamps)

        if selected_version:
            version_index = version_timestamps.index(selected_version)
            selected_version_messages = st.session_state.chat_versions[version_index][1]

            logging.info(f"User selected {selected_version} for viewing.")

            st.write(f"📝 Messages in {selected_version}:")

            for msg in selected_version_messages:
                # ✅ Extract content and type
                msg_content = self.get_message_content(msg)
                msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else getattr(msg, "type", "unknown")

                # 🎨 Display messages with formatting
                if msg_type == "human":
                    st.markdown(f'👤 **User:** {msg_content}')
                elif msg_type == "ai":
                    st.markdown(f'🤖 **AI:** {msg_content}')
                else:
                    st.markdown(f'⚠️ **Unknown:** {msg_content}')

            # 🔄 Restore button logic
            if st.button("⏪ Restore Selected Version"):
                st.session_state.restored = True  # 🔹 Mark as restored
                st.session_state.reset_tokens = True  # ✅ Indicate tokens should reset

                # 🗑️ Clear current chat messages before restoring
                st.session_state.chat_manager.memory.chat_memory.messages.clear()

                # 🔄 Deep copy to prevent reference issues
                restored_messages = copy.deepcopy(selected_version_messages)
                st.session_state.chat_manager.memory.chat_memory.messages = restored_messages

                # 🕒 Create a new version for restored messages
                new_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                st.session_state.chat_versions.append((new_timestamp, restored_messages))

                # 📝 Log the restoration event
                if hasattr(st.session_state.chat_manager, "change_log"):
                    st.session_state.chat_manager.change_log.append(
                        f"[{new_timestamp}] Restored chat from {selected_version}"
                    )

                st.success(f"✅ Successfully restored to {selected_version}! A new version has been created.")
                st.rerun()


    def show_usage_plots(self):
        """
        📈 Visualizes token usage trends using both bar and line charts.

        Workflow:
        1️⃣ Retrieves token usage data from the session state.
        2️⃣ Validates if there is data available.
        3️⃣ Creates a combined visualization:
        - 📊 Stacked bar chart for Prompt & Completion tokens.
        - 📉 Line chart for Total Token usage.
        4️⃣ Enhances readability with labels, legends, and formatting.
        """

        st.subheader("📊 Token Usage Analysis")

        # ✅ Retrieve token usage history from ChatManager
        token_usage_history = st.session_state.chat_manager.token_usage_history

        # 🚨 Ensure token usage data exists before plotting
        if not token_usage_history:
            st.warning("⚠️ No token usage history available.")
            return

        # 🎨 Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))  # Single plot for both bar and line charts

        # 📌 Extract token usage data
        steps = token_usage_history["steps"]
        prompt_tokens = token_usage_history["individual_prompt_tokens"]
        completion_tokens = token_usage_history["individual_completion_tokens"]
        total_tokens = token_usage_history["cumulative_tokens"]

        index = np.arange(len(steps))  # X-axis positions
        bar_width = 0.5  # Width of each bar

        # 📊 Stacked bar chart (Prompt & Completion Tokens)
        ax.bar(index, prompt_tokens, bar_width, label="🟢 Prompt Tokens", color="green")
        ax.bar(index, completion_tokens, bar_width, bottom=prompt_tokens, label="🔴 Completion Tokens", color="red")

        # 📉 Line plot for Total Token Usage
        ax.plot(index, total_tokens, label="🔵 Total Tokens", color="blue", marker="o", linewidth=2)

        # 🏷️ Formatting and labels
        ax.set_title("📈 Token Usage (Total, Prompt, and Completion)")
        ax.set_xlabel("⏳ Steps")
        ax.set_ylabel("🔢 Token Count")
        ax.set_xticks(index)
        ax.set_xticklabels(steps)

        # 🔄 Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # 🎯 Add legend and grid
        ax.legend()
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()  # Adjust layout to avoid overlaps

        # 📌 Display the plot in Streamlit
        st.pyplot(fig)

    def display_change_log(self):
        """📜 Displays the recorded change log."""
        st.subheader("📜 Change Log")
        
        if not hasattr(st.session_state.chat_manager, "change_log"):
            st.session_state.chat_manager.change_log = []  # Ensure change log exists

        if st.session_state.chat_manager.change_log:
            for entry in reversed(st.session_state.chat_manager.change_log):  # Show latest changes first
                if "Restored chat" in entry:
                    st.markdown(f"🔄 **{entry}**")  # Highlight restorations
                else:
                    st.markdown(f"🆕 {entry}")  # Highlight new versions
        else:
            st.write("⚠️ No changes logged yet.")


    def main(self):
        """
        🚀 Controls the app flow: initializes components, manages layout, and runs the chat interface.
        """

        # 🏗️ Setup page
        self.setup_page()
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            overflow: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # 🎨 Layout: Left (Chat) | Right (Memory & Stats)
        col1, col2 = st.columns([60, 40])  # 60% chat, 40% utilities

        with col1:
            # 🗂️ Display chat history
            self.display_chat_history()

            # ⌨️ Handle user input in the chat interface
            self.handle_user_input()

        with col2:
            # 🏷️ Sidebar Tabs for Utility Functions
            tab1, tab2, tab3 = st.tabs(["🗂 Version Management", "📈 Token Stats", "📝 Change Log"])

            with tab1:
                st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
                self.display_versions()  # Handle version selection & restoration

            with tab2:
                self.show_usage_plots()  # Display token usage analytics

            with tab3:
                self.display_change_log()  # 📜 Show Change Log



# Run the application
if __name__ == "__main__":
    app = ChatApp()
    app.main()
