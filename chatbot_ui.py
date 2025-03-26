import streamlit as st
import matplotlib.pyplot as plt
from chatbot import ChatManager
import time
import copy

def init_chat():
    """Initialize chat session and memory in Streamlit."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()  # Use existing main code

    # ✅ Ensure chat versions are stored
    if "chat_versions" not in st.session_state:
        st.session_state.chat_versions = []

st.set_page_config(page_title="Chatbot", layout="wide")

st.title("💬 OpenAI Chatbot with Memory Management")

init_chat()

# --- 📌 TABS ---
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📜 Memory", "📈 Token Stats"])

# --- 🟢 TAB 1: CHAT ---
with tab1:
    st.subheader("Chat History")

    # 🔹 Scrollable Chat Container
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_manager.memory.chat_memory.messages:
            if msg.type == "human":
                st.markdown(
                    f'<div style="background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin-bottom: 5px;">👤 {msg.content}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color: #ffe0b2; padding: 10px; border-radius: 5px; margin-bottom: 5px;">🤖 {msg.content}</div>',
                    unsafe_allow_html=True
                )

# 🔹 FIXED Input Bar at Bottom
st.markdown(
    """<style>
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
    </style>""",
    unsafe_allow_html=True
)

user_input = st.chat_input("Type your message...")

if user_input:
    response = st.session_state.chat_manager.process_user_input(user_input)

    # ✅ Store a deep copy of messages in session state
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    chat_snapshot = (timestamp, copy.deepcopy(st.session_state.chat_manager.memory.chat_memory.messages))

    # Append the chat snapshot, but don't include the first "empty version"
    if len(st.session_state.chat_versions) == 0:
        st.session_state.chat_versions.append(chat_snapshot)
    else:
        st.session_state.chat_versions.append(chat_snapshot)  # Save new chat version

    st.success("Chat history saved!")  # Notify user
    st.rerun()  # Refresh UI

# --- 📜 TAB 2: MEMORY MANAGEMENT ---
with tab2:
    st.subheader("Chat Memory Management")

    # View saved chat versions
    chat_manager = st.session_state.chat_manager
    versions = chat_manager.chat_versions

    if not versions:
        st.info("No saved chat versions yet.")
    else:
        # # Exclude the first "empty version"
        # version_options = [f"Version {i} - {v[0]}" for i, v in enumerate(versions[1:])]
        
        if version_options:
            selected_version = st.selectbox("Select a version to restore:", version_options)

            # Extract version index and data from the selected version
            version_index = version_options.index(selected_version)
            version_data = versions[version_index + 1]  # Offset by 1 to exclude the "empty version"

            # Show messages in the selected version
            st.write(f"Messages in selected version {version_index + 1}:")
            for msg in version_data[1]:
                st.write(f"{msg.type}: {msg.content}")

            # Deleting messages from the selected version
            delete_selected_messages = st.multiselect("Select messages to delete:", 
                                                    [f"{i}: {msg.type} - {msg.content}" for i, msg in enumerate(version_data[1])])

            if delete_selected_messages:
                delete_indexes = [int(msg.split(":")[0]) for msg in delete_selected_messages]
                if st.button("Delete Selected Messages"):
                    chat_manager.delete_chat_messages(delete_indexes)
                    chat_manager.save_version(f"Deleted Messages Version {version_index}")  # New version after deletion
                    st.success("Selected messages deleted successfully. The original chat history is preserved.")
                    st.rerun()  # Immediately reflect updates

            # Restoring a version
            if st.button(f"Restore Version {version_index + 1}"):
                chat_manager.restore_chat_version(version_index + 1)  # Offset by 1 to skip the empty version
                chat_manager.save_version(f"Restored Version {version_index + 1}")  # Save after restore
                st.success(f"Restored to version {version_index + 1}. The original chat history is intact.")
                st.rerun()  # Immediately reflect updates
        else:
            st.warning("No versions available to select.")

# --- 🔵 TAB 3: TOKEN USAGE PLOTS ---
with tab3:
    st.subheader("📊 Token Usage Statistics")

    # 🔹 Get screen width dynamically (Estimation)
    screen_width = st.get_option("server.maxUploadSize")  # Approximate screen width in px

    # 🔹 Adjust figure size based on screen width (Adaptive Scaling)
    if screen_width > 1200:  # Desktop
        fig_width, fig_height = 15, 10
    elif screen_width > 800:  # Tablet
        fig_width, fig_height = 15, 10
    else:  # Mobile
        fig_width, fig_height = 10, 10

    # 🔹 Create responsive container to prevent stretching
    with st.container():
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])  # Center plot on wider screens
        with col2:
            fig, axs = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=200)

            # 📊 Cumulative Token Usage Plot
            axs[0].plot(st.session_state.chat_manager.token_usage_history["steps"], 
                        st.session_state.chat_manager.token_usage_history["cumulative_tokens"], 
                        label="Total Tokens", color="blue", marker="o")

            axs[0].plot(st.session_state.chat_manager.token_usage_history["steps"], 
                        st.session_state.chat_manager.token_usage_history["cumulative_prompt_tokens"], 
                        label="Prompt Tokens", color="green", marker="o")

            axs[0].plot(st.session_state.chat_manager.token_usage_history["steps"], 
                        st.session_state.chat_manager.token_usage_history["cumulative_completion_tokens"], 
                        label="Completion Tokens", color="red", marker="o")

            axs[0].set_title("Cumulative Token Usage")
            axs[0].legend()
            axs[0].grid(True)

            # 📊 Per-Request Token Usage Bar Chart
            axs[1].bar(st.session_state.chat_manager.token_usage_history["steps"], 
                    st.session_state.chat_manager.token_usage_history["individual_total_tokens"], 
                    label="Total Tokens", color="blue", alpha=0.7)

            axs[1].bar(st.session_state.chat_manager.token_usage_history["steps"], 
                    st.session_state.chat_manager.token_usage_history["individual_prompt_tokens"], 
                    label="Prompt Tokens", color="green", alpha=0.7)

            axs[1].bar(st.session_state.chat_manager.token_usage_history["steps"], 
                    st.session_state.chat_manager.token_usage_history["individual_completion_tokens"], 
                    label="Completion Tokens", color="red", alpha=0.7)

            axs[1].set_title("Token Usage Per Request")
            axs[1].legend()
            axs[1].grid(axis="y", linestyle="--", alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)
