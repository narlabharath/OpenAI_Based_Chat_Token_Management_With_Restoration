import time
import copy
import matplotlib.pyplot as plt
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.openai_info import OpenAICallbackHandler  # <-- Add this import
import numpy as np


class ChatManager:
    def __init__(self, model_name="gpt-4o-mini"):
        """🚀 Initialize ChatManager with LLM, memory & token tracking"""

        self.model_name = model_name  # 🤖 Set the LLM model name
        self.chat_versions = []  # 📂 Stores different versions of chat history
        self.current_version = None  # 🔄 Tracks the active chat version
        self.change_log = []  # ✅ Add this to track changes

        # ✅ Global storage for ALL token usage across chat versions
        self.token_usage_history = {
            "steps": [],  # 📊 Step-wise tracking
            "cumulative_tokens": [],  # 🔢 Cumulative token count
            "cumulative_prompt_tokens": [],  # ✍️ Prompt token count
            "cumulative_completion_tokens": [],  # 📝 Completion token count
            "individual_total_tokens": [],  # 🔍 Per-interaction total tokens
            "individual_prompt_tokens": [],  # 📩 Per-interaction prompt tokens
            "individual_completion_tokens": []  # ✅ Per-interaction completion tokens
        }

        self._initialize_llm_and_tracking()  # 🔧 Load LLM & tracking system

        self._create_initial_version()  # 🆕 Create an initial empty chat version

        self.change_log = []  # 📜 Log changes like restores & deletions

    def _create_initial_version(self):
        """🆕 Initializes an empty chat version at the start."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.chat_versions.append((timestamp, []))  # Empty chat history
        self.current_version = 0
        self.change_log.append(f"[{timestamp}] Initialized Chat")  # ✅ Log initialization



    def _initialize_llm_and_tracking(self):
        """🔧 Initialize LLM, memory & token tracking"""

        # 📡 Callback handler for tracking token usage
        self.callback_handler = OpenAICallbackHandler()

        # 🤖 Load the LLM model with tracking enabled
        self.llm = ChatOpenAI(model_name=self.model_name, callbacks=[self.callback_handler])

        # 🧠 Memory management for conversation history
        self.memory = ConversationBufferMemory(return_messages=True)

        # 📜 Create & store the prompt template
        self.prompt = self._create_prompt_template()

        # 🔗 Set up LLM processing chain with memory & prompt
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory, verbose=False)


    def _create_prompt_template(self):
        """📜 Define structured prompt templates for LLM interactions"""

        # 🎯 System-level instructions for the LLM
        system_prompt = "You are an AI assistant. Provide clear, concise, and helpful responses."

        # 🔄 User prompt format for structured input
        user_prompt = "{user_input}"

        # 📝 Final formatted prompt
        return PromptTemplate(input_variables=["user_input"], template=system_prompt + "\nUser: " + user_prompt)


    def process_user_input(self, user_input):
        """💬 Process user input, generate response, track history & tokens"""

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 🚀 Send input to LLM and receive response
        result = self.chain.invoke(user_input)

        # 📜 Save chat history as a new version
        self.chat_versions.append((timestamp, copy.deepcopy(self.memory.chat_memory.messages)))
        self.current_version = len(self.chat_versions) - 1

        # ✅ Log the change
        self.change_log.append(f"[{timestamp}] New chat version created (Version {self.current_version})")

        # 🔢 Track token usage after processing input
        self._track_token_usage()

        return result


    def restore_version(self, version_index):
        """⏪ Restore a previous chat version"""

        if version_index < 0 or version_index >= len(self.chat_versions):
            return "❌ Invalid version index!"

        # 📜 Retrieve the selected chat history
        timestamp, restored_messages = self.chat_versions[version_index]

        # 🧠 Restore memory to the selected version
        self.memory.chat_memory.messages = copy.deepcopy(restored_messages)
        self.current_version = version_index

        # ✅ Log restoration
        restore_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.change_log.append(f"[{restore_time}] Restored to Version {version_index} ({timestamp})")

        # 🔄 Reset token tracking to avoid inconsistencies
        self._track_token_usage(is_reset=True)

        return f"✅ Restored to version {version_index} from {timestamp}."


    def _track_token_usage(self, is_reset=False):
        """📊 Track token usage, handle resets & maintain history"""
        
        total_used = self.callback_handler.total_tokens
        prompt_used = self.callback_handler.prompt_tokens
        completion_used = self.callback_handler.completion_tokens

        # 🔄 Handle Reset Case (Restoring to a Previous State)
        if is_reset:
            # 🆕 Mark reset event & reinitialize tracking
            self.token_usage_history["steps"].append("RESET")
            self.token_usage_history["cumulative_tokens"].append(0)
            self.token_usage_history["cumulative_prompt_tokens"].append(0)
            self.token_usage_history["cumulative_completion_tokens"].append(0)
            self.token_usage_history["individual_total_tokens"].append(0)
            self.token_usage_history["individual_prompt_tokens"].append(0)
            self.token_usage_history["individual_completion_tokens"].append(0)

            # 🔄 Store baseline values after reset
            self.last_known_cumulative = {
                "total": total_used,
                "prompt": prompt_used,
                "completion": completion_used
            }
            return  # 🚀 Exit early to prevent additional logging

        # 🔍 Adjust Tracking Based on Previous Usage
        prev_total = getattr(self, "last_known_cumulative", {}).get("total", 0)
        prev_prompt = getattr(self, "last_known_cumulative", {}).get("prompt", 0)
        prev_completion = getattr(self, "last_known_cumulative", {}).get("completion", 0)

        # 🛠 Compute Individual Token Usage Since Last Update
        individual_total = max(0, total_used - prev_total)
        individual_prompt = max(0, prompt_used - prev_prompt)
        individual_completion = max(0, completion_used - prev_completion)

        # 📌 Append New Token Usage Data
        step = len(self.token_usage_history["steps"]) + 1
        self.token_usage_history["steps"].append(step)
        self.token_usage_history["cumulative_tokens"].append(individual_total)
        self.token_usage_history["cumulative_prompt_tokens"].append(individual_prompt)
        self.token_usage_history["cumulative_completion_tokens"].append(individual_completion)
        self.token_usage_history["individual_total_tokens"].append(individual_total)
        self.token_usage_history["individual_prompt_tokens"].append(individual_prompt)
        self.token_usage_history["individual_completion_tokens"].append(individual_completion)