from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Literal

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Conversation:
    messages: list[Message]

class LLM(ABC):
    @abstractmethod
    def infer(self,conversations: Conversation|list[Conversation]) -> str|list[str]:
        pass

class Qwen(LLM):
    def __init__(self, model_path: str, torch_dtype = "auto", device_map = "auto"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    @staticmethod
    def create_conversation(roles: list[Literal["system","user"]], contents: list[str]) -> Conversation:
        return Conversation([Message(role, content) for role, content in zip(roles, contents)])
    def infer(self,conversations: Conversation|list[Conversation]) -> str|list[str]:
        batch_op = True
        if isinstance(conversations, Conversation):
            conversations = [conversations]
            batch_op = False
        responses = []
        for conversation in conversations:
            text = self.tokenizer.apply_chat_template(
                conversation.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)
        if batch_op:
            return responses
        return responses[0]