import time
import copy
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.openai_info import OpenAICallbackHandler  # <-- Add this import
import numpy as np

# Your class implementation here...




class ChatManager:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initializes the Chat Manager with memory, LLM model, and token tracking."""
        self.model_name = model_name
        self.chat_versions = []
        self.current_version = None

        # ✅ Global storage for ALL token usage across chat versions
        self.token_usage_history = {
            "steps": [],
            "cumulative_tokens": [],
            "cumulative_prompt_tokens": [],
            "cumulative_completion_tokens": [],
            "individual_total_tokens": [],
            "individual_prompt_tokens": [],
            "individual_completion_tokens": []
        }

        self._initialize_llm_and_tracking()

        # Create initial empty version (Version 0)
        self._create_initial_version()

        self.change_log = []  # Log changes like restores and deletions

    def _initialize_llm_and_tracking(self):
        """Sets up LLM, memory, and ensures token tracking is updated correctly."""
        self.callback_handler = OpenAICallbackHandler()
        self.llm = ChatOpenAI(model_name=self.model_name, callbacks=[self.callback_handler])
        self.memory = ConversationBufferMemory(return_messages=True)
        self.prompt = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=False)

    def _create_initial_version(self):
        """Creates the initial empty version (Version 0)."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Create Version 0 with no messages to mark the starting point of the conversation
        self.chat_versions.append((timestamp, []))  # Initial empty version
        self.current_version = 0  # The current version is Version 0

    def _create_prompt_template(self):
        """Defines the chat prompt format."""
        template = """
        The following is a friendly conversation between an intelligent chatbot and a human.
        Chatbot: {history}
        Human: {input}
        Chatbot:
        """
        return PromptTemplate(template=template, input_variables=["history", "input"])

    def process_user_input(self, user_input):
        """Processes user input, updates chat memory, and tracks token usage."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Generate a response from the model
        result = self.chain.invoke(user_input)

        # Add current memory (chat history) to versions list after processing
        self.chat_versions.append((timestamp, copy.deepcopy(self.memory.chat_memory.messages)))
        self.current_version = len(self.chat_versions) - 1

        # Track token usage after processing the input
        self._track_token_usage()
        
        return result

    def _track_token_usage(self, is_reset=False):
        """Tracks token usage, handling restores properly while keeping history clean."""

        total_used = self.callback_handler.total_tokens
        prompt_used = self.callback_handler.prompt_tokens
        completion_used = self.callback_handler.completion_tokens

        # 🔹 Handle Reset (Restore)
        if is_reset:
            # ✅ Reset cumulative tracking to the new base values
            self.token_usage_history["steps"].append("RESET")  # Mark the reset point
            self.token_usage_history["cumulative_tokens"].append(0)
            self.token_usage_history["cumulative_prompt_tokens"].append(0)
            self.token_usage_history["cumulative_completion_tokens"].append(0)
            self.token_usage_history["individual_total_tokens"].append(0)
            self.token_usage_history["individual_prompt_tokens"].append(0)
            self.token_usage_history["individual_completion_tokens"].append(0)

            # ✅ Mark new baseline for tracking
            self.last_known_cumulative = {
                "total": total_used,
                "prompt": prompt_used,
                "completion": completion_used
            }
            return  # Exit to avoid logging an extra step

        # 🔹 Adjust Tracking After Restore
        prev_total = self.last_known_cumulative["total"] if hasattr(self, "last_known_cumulative") else 0
        prev_prompt = self.last_known_cumulative["prompt"] if hasattr(self, "last_known_cumulative") else 0
        prev_completion = self.last_known_cumulative["completion"] if hasattr(self, "last_known_cumulative") else 0

        # ✅ Ensure the first step after reset starts from zero correctly
        individual_total = max(0, total_used - prev_total)
        individual_prompt = max(0, prompt_used - prev_prompt)
        individual_completion = max(0, completion_used - prev_completion)

        # Append new token usage values
        step = len(self.token_usage_history["steps"]) + 1
        self.token_usage_history["steps"].append(step)
        self.token_usage_history["cumulative_tokens"].append(individual_total)
        self.token_usage_history["cumulative_prompt_tokens"].append(individual_prompt)
        self.token_usage_history["cumulative_completion_tokens"].append(individual_completion)
        self.token_usage_history["individual_total_tokens"].append(individual_total)
        self.token_usage_history["individual_prompt_tokens"].append(individual_prompt)
        self.token_usage_history["individual_completion_tokens"].append(individual_completion)


    def show_chat_versions(self):
        """Displays all chat versions saved so far."""
        print("\n📜 Chat Versions:")
        for i, (timestamp, _) in enumerate(self.chat_versions):
            mark = " (In Use) 🔵" if i == self.current_version else ""
            print(f"Version {i}: {timestamp}{mark}")

    def show_version_messages(self, version_number):
        """Displays all messages from a specific chat version."""
        if 0 <= version_number < len(self.chat_versions):
            print(f"\n🗂️ Messages in Version {version_number}:")
            for i, msg in enumerate(self.chat_versions[version_number][1]):
                print(f"{i}. {msg.type}: {msg.content}")
        else:
            print("⚠️ Invalid version number!")

    def restore_chat_version(self, version_number):
        """Restores a previous chat version while maintaining full token tracking history."""
        if 0 <= version_number < len(self.chat_versions):
            self.memory.clear()  # Fully clear memory before restoring

            for msg in self.chat_versions[version_number][1]:
                if msg.type == "human":
                    self.memory.chat_memory.add_user_message(msg.content)
                else:
                    self.memory.chat_memory.add_ai_message(msg.content)

            self.current_version = version_number
            self._initialize_llm_and_tracking()

            # 🛠️ Reset token usage tracking to avoid negative values
            self._track_token_usage(is_reset=True)

            # Log the restoration with timestamp and version number
            change_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Capture the time of the restoration
            self.change_log.append(f"[{change_timestamp}] Restored to Version {version_number}.")

            print(f"✅ Successfully restored to Version {version_number}. Token tracking corrected.")
        else:
            print("⚠️ Invalid version number!")


    def delete_chat_messages(self, indexes_to_delete):
        """Deletes selected messages but keeps full token tracking history."""
        print("Deleting messages with the following indexes:", indexes_to_delete)  # Debugging line
        latest_messages = copy.deepcopy(self.memory.chat_memory.messages)

        if not latest_messages:
            print("⚠️ No messages to delete!")
            return

        # Ensure we only delete valid indexes
        # Check that all elements of indexes_to_delete are integers
        if not all(isinstance(i, int) for i in indexes_to_delete):
            print("⚠️ Invalid input: indexes_to_delete should be a list of integers.")
            return

        valid_indexes = [i for i in indexes_to_delete if 0 <= i < len(latest_messages)]
        if not valid_indexes:
            print("⚠️ No valid messages to delete! Operation ignored.")
            return

        # Log before deletion
        print(f"Deleting indexes: {valid_indexes}. Messages before deletion: {len(latest_messages)}")

        # Keep only messages that are NOT being deleted
        new_version_messages = [msg for i, msg in enumerate(latest_messages) if i not in valid_indexes]

        # Log after deletion
        print(f"Messages after deletion: {len(new_version_messages)}")

        # Fully clear memory
        self.memory.clear()

        # Restore only the remaining messages
        for msg in new_version_messages:
            if msg.type == "human":
                self.memory.chat_memory.add_user_message(msg.content)
            else:
                self.memory.chat_memory.add_ai_message(msg.content)

        # Save this new version
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.chat_versions.append((timestamp, copy.deepcopy(new_version_messages)))
        self.current_version = len(self.chat_versions) - 1

        # Log the change, including the version number and timestamp
        change_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Capture the time of the change
        self.change_log.append(f"[{change_timestamp}] Deleted messages at indexes: {valid_indexes} from Version {self.current_version}.")
        print(f"✅ Created new version {self.current_version} after deletion! Remaining messages: {len(self.memory.chat_memory.messages)}")


    def show_change_log(self):
        """Displays the log of changes (restores, deletions, etc.)."""
        print("\n📝 Change Log:")
        if self.change_log:
            for entry in self.change_log:
                print(f"- {entry}")
        else:
            print("⚠️ No changes recorded.")
