import time
import copy
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.openai_info import OpenAICallbackHandler

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

    def _initialize_llm_and_tracking(self):
        """Sets up LLM, memory, and ensures token tracking is updated correctly."""
        self.callback_handler = OpenAICallbackHandler()
        self.llm = ChatOpenAI(model_name=self.model_name, callbacks=[self.callback_handler])
        self.memory = ConversationBufferMemory(return_messages=True)
        self.prompt = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=False)

    def _create_prompt_template(self):
        """Defines the chat prompt format."""
        template = """
        The following is a friendly conversation between an intelligent chatbot and a human.
        Chatbot: {history}
        Human: {input}
        Chatbot:
        """
        return PromptTemplate(template=template, input_variables=["history", "input"])

    # def process_user_input(self, user_input):
    #     """Processes user input, updates chat memory, and tracks token usage."""
    #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     self.chat_versions.append((timestamp, copy.deepcopy(self.memory.chat_memory.messages)))
    #     self.current_version = len(self.chat_versions) - 1

    #     result = self.chain.invoke(user_input)
    #     self._track_token_usage()
    #     return result

    def process_user_input(self, user_input):
        """Processes user input, updates chat memory, and tracks token usage."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Only append to versions if there are messages
        if self.memory.chat_memory.messages:
            self.chat_versions.append((timestamp, copy.deepcopy(self.memory.chat_memory.messages)))
            self.current_version = len(self.chat_versions) - 1

        result = self.chain.invoke(user_input)
        self._track_token_usage()
        return result


    def _track_token_usage(self, is_reset=False):
        """Tracks token usage while preserving chat history. Handles resets properly."""
        total_used = self.callback_handler.total_tokens
        prompt_used = self.callback_handler.prompt_tokens
        completion_used = self.callback_handler.completion_tokens

        # 🛠️ Get last known cumulative values (before reset)
        prev_total = self.token_usage_history["cumulative_tokens"][-1] if self.token_usage_history["cumulative_tokens"] else 0
        prev_prompt = self.token_usage_history["cumulative_prompt_tokens"][-1] if self.token_usage_history["cumulative_prompt_tokens"] else 0
        prev_completion = self.token_usage_history["cumulative_completion_tokens"][-1] if self.token_usage_history["cumulative_completion_tokens"] else 0

        # 🛠️ Handle Reset Properly
        if is_reset:
            # Instead of resetting to zero, continue tracking smoothly
            self.last_known_cumulative = {
                "total": total_used,
                "prompt": prompt_used,
                "completion": completion_used
            }
            return  # Don't log a new step yet; let the next request handle it correctly

        # Adjust first step after reset to avoid zero dip
        if hasattr(self, "last_known_cumulative"):
            prev_total = self.last_known_cumulative["total"]
            prev_prompt = self.last_known_cumulative["prompt"]
            prev_completion = self.last_known_cumulative["completion"]
            del self.last_known_cumulative  # Ensure it's only used once

        # Ensure no negative values appear
        individual_total = max(0, total_used - prev_total)
        individual_prompt = max(0, prompt_used - prev_prompt)
        individual_completion = max(0, completion_used - prev_completion)

        # Store new token usage values
        steps = len(self.token_usage_history["steps"]) + 1
        self.token_usage_history["steps"].append(steps)
        self.token_usage_history["cumulative_tokens"].append(total_used)
        self.token_usage_history["cumulative_prompt_tokens"].append(prompt_used)
        self.token_usage_history["cumulative_completion_tokens"].append(completion_used)
        self.token_usage_history["individual_total_tokens"].append(individual_total)
        self.token_usage_history["individual_prompt_tokens"].append(individual_prompt)
        self.token_usage_history["individual_completion_tokens"].append(individual_completion)

    # def show_chat_versions(self):
    #     """Displays all chat versions saved so far."""
    #     print("\n📜 Chat Versions:")
    #     for i, (timestamp, _) in enumerate(self.chat_versions):
    #         mark = " (In Use) 🔵" if i == self.current_version else ""
    #         print(f"Version {i}: {timestamp}{mark}")

    # def show_chat_versions(self):
    #     """Displays all chat versions saved so far."""
    #     print("\n📜 Chat Versions:")
    #     for i, (timestamp, _) in enumerate(self.chat_versions):
    #         mark = " (In Use) 🔵" if i == self.current_version else ""
    #         # Append (Deleted) or (Restored) based on the action taken
    #         if hasattr(self, 'deleted_versions') and i in self.deleted_versions:
    #             mark += " (Deleted)"
    #         if hasattr(self, 'restored_versions') and i in self.restored_versions:
    #             mark += " (Restored)"
    #         print(f"Version {i}: {timestamp}{mark}")

    def show_chat_versions(self):
        """Displays all chat versions saved so far."""
        print("\n📜 Chat Versions:")
        
        # Skip the first version if it's empty
        for i, (timestamp, messages) in enumerate(self.chat_versions):
            if not messages:  # Skip empty versions
                continue
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

    # def restore_chat_version(self, version_number):
    #     """Restores a previous chat version while maintaining full token tracking history."""
    #     if 0 <= version_number < len(self.chat_versions):
    #         self.memory.clear()  # Fully clear memory before restoring

    #         for msg in self.chat_versions[version_number][1]:
    #             if msg.type == "human":
    #                 self.memory.chat_memory.add_user_message(msg.content)
    #             else:
    #                 self.memory.chat_memory.add_ai_message(msg.content)

    #         self.current_version = version_number
    #         self._initialize_llm_and_tracking()

    #         # 🛠️ Reset token usage tracking to avoid negative values
    #         self._track_token_usage(is_reset=True)

    #         print(f"✅ Successfully restored to Version {version_number}. Token tracking corrected.")
    #     else:
    #         print("⚠️ Invalid version number!")


    # def delete_chat_messages(self, indexes_to_delete):
    #     """Deletes selected messages but keeps full token tracking history."""
    #     latest_messages = copy.deepcopy(self.memory.chat_memory.messages)

    #     if not latest_messages:
    #         print("⚠️ No messages to delete!")
    #         return

    #     # 🔹 Ensure we only delete valid indexes
    #     valid_indexes = [i for i in indexes_to_delete if 0 <= i < len(latest_messages)]
    #     if not valid_indexes:
    #         print("⚠️ No valid messages to delete! Operation ignored.")
    #         return

    #     # ✅ Log before deletion
    #     print(f"Deleting indexes: {valid_indexes}. Messages before deletion: {len(latest_messages)}")

    #     # 🔹 Keep only messages that are NOT being deleted
    #     new_version_messages = [msg for i, msg in enumerate(latest_messages) if i not in valid_indexes]

    #     # ✅ Log after deletion
    #     print(f"Messages after deletion: {len(new_version_messages)}")

    #     # 🔹 Fully clear memory
    #     self.memory.clear()
        
    #     # 🔹 Restore only the remaining messages
    #     for msg in new_version_messages:
    #         if msg.type == "human":
    #             self.memory.chat_memory.add_user_message(msg.content)
    #         else:
    #             self.memory.chat_memory.add_ai_message(msg.content)

    #     # 🔹 Save this new version
    #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #     self.chat_versions.append((timestamp, copy.deepcopy(new_version_messages)))
    #     self.current_version = len(self.chat_versions) - 1

    #     print(f"✅ Created new version {self.current_version} after deletion! Remaining messages: {len(self.memory.chat_memory.messages)}")
    
    def delete_chat_messages(self, indexes_to_delete):
        """Deletes selected messages but keeps full token tracking history."""
        latest_messages = copy.deepcopy(self.memory.chat_memory.messages)

        if not latest_messages:
            print("⚠️ No messages to delete!")
            return

        # Ensure we only delete valid indexes
        valid_indexes = [i for i in indexes_to_delete if 0 <= i < len(latest_messages)]
        if not valid_indexes:
            print("⚠️ No valid messages to delete! Operation ignored.")
            return

        # Keep only messages that are NOT being deleted
        new_version_messages = [msg for i, msg in enumerate(latest_messages) if i not in valid_indexes]

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

        # Mark as deleted
        if not hasattr(self, 'deleted_versions'):
            self.deleted_versions = []
        self.deleted_versions.append(self.current_version)

        print(f"✅ Created new version {self.current_version} after deletion! Remaining messages: {len(self.memory.chat_memory.messages)}")
        # st.rerun()  # Trigger rerun to reflect changes

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

            # Reset token usage tracking to avoid negative values
            self._track_token_usage(is_reset=True)

            # Mark as restored
            if not hasattr(self, 'restored_versions'):
                self.restored_versions = []
            self.restored_versions.append(self.current_version)

            print(f"✅ Successfully restored to Version {version_number}. Token tracking corrected.")
            # st.rerun()  # Trigger rerun to reflect changes
        else:
            print("⚠️ Invalid version number!")


    def save_version(self, version_name):
        """Saves the current state of the chat memory as a new version."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Here you can customize how you want to save the version, e.g., adding a version name or description
        self.chat_versions.append((version_name, copy.deepcopy(self.memory.chat_memory.messages)))
        self.current_version = len(self.chat_versions) - 1
        print(f"Saved version: {version_name} at {timestamp}")


    def show_usage_plots(self):
        """Displays cumulative & per-request token usage plots."""
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(self.token_usage_history["steps"], self.token_usage_history["cumulative_tokens"], label="Total Tokens", color="blue", marker="o")
        axs[0].plot(self.token_usage_history["steps"], self.token_usage_history["cumulative_prompt_tokens"], label="Prompt Tokens", color="green", marker="o")
        axs[0].plot(self.token_usage_history["steps"], self.token_usage_history["cumulative_completion_tokens"], label="Completion Tokens", color="red", marker="o")
        axs[0].set_title("Cumulative Token Usage")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].bar(self.token_usage_history["steps"], self.token_usage_history["individual_total_tokens"], label="Total Tokens", color="blue", alpha=0.7)
        axs[1].bar(self.token_usage_history["steps"], self.token_usage_history["individual_prompt_tokens"], label="Prompt Tokens", color="green", alpha=0.7)
        axs[1].bar(self.token_usage_history["steps"], self.token_usage_history["individual_completion_tokens"], label="Completion Tokens", color="red", alpha=0.7)
        axs[1].set_title("Token Usage Per Request")
        axs[1].legend()
        axs[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

