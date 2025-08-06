from pathlib import Path
from typing import Dict, OrderedDict, Tuple, Union
from .custom_types import MessageId, Message, Conversation, MessageText
from typing import List
import re
import json


class TelegramDataExtractor:
    @classmethod
    def load_messages_from_json(cls, path: str, max_message_count: int = None) -> OrderedDict[MessageId, Message]:
        messages: OrderedDict[MessageId, Message] = OrderedDict()
        with open(path, "r", encoding="utf-8") as file:
            chat_json = json.load(file)
            for i, message in enumerate(chat_json["messages"]):
                if max_message_count and i == max_message_count:
                    break
                if message["type"] != "message":
                    continue
                new_message = {
                    "from": cls.normalize_username(message["from"]),
                    "id": message["id"],
                    "text": cls.normalize(message["text"])
                }
                if not new_message["text"]: # Check for empty message
                    continue
                if "reply_to_message_id" in message.keys():
                    new_message["reply_to_id"] = message["reply_to_message_id"]

                messages[new_message["id"]] = new_message
        return messages

    @staticmethod
    def conversations_from_messages(save_to: Path, tokenizer, messages: OrderedDict[MessageId, Message]) -> List[Conversation]:
        _MAX_MESSAGE_LEN = 150
        _MAX_QA_LEN_DIFF = 20
        def remove_duplicates_keep_order(lst: List[Conversation]) -> List[Conversation]:
            lst = list(dict.fromkeys(lst)) # Remove duplicates and keep order
            return [(list(x[0]), list(x[1]), x[2]) for x in lst] # Tuples are only needed for hashability
        def remove_answers_with_only_special_symbols(lst: List[Conversation]) -> List[Conversation]:
            return [i for i in lst if re.findall(r"[а-я]", " ".join(i[1]))]
        def remove_long_qa(lst: List[Conversation]) -> List[Conversation]:
            return [i for i in lst if len(i[0]) <= _MAX_MESSAGE_LEN and len(i[1]) <= _MAX_MESSAGE_LEN]
        def remove_unbalanced_qa(lst: List[Conversation]) -> List[Conversation]:
            return [i for i in lst if abs(len(i[0]) - len(i[1])) <= _MAX_QA_LEN_DIFF]
        def normalize_conversations(lst: List[Conversation]) -> List[Conversation]:
            lst = remove_duplicates_keep_order(lst)
            lst = remove_answers_with_only_special_symbols(lst)
            lst = remove_long_qa(lst)
            lst = remove_unbalanced_qa(lst)
            return lst

        conversations: List[Conversation] = []
        questions: Dict[MessageText, int] = dict()

        messages_values = list(messages.values()) # TODO: try changing this cast to something more applicable
        for i in range(len(messages) - 1): # There's no answer for last message so add -1
            try: # Message is answer for message with `id` of `reply_to_id`
                prev_message = messages[messages_values[i+1]["reply_to_id"]]
            except KeyError:
                prev_message = messages_values[i]
            qa = (prev_message["text"], messages_values[i+1]["text"], prev_message["from"])
            if qa[0] in questions.keys(): # If there are multiple answers for same message, choose the longest one
                if len(conversations[questions[qa[0]]][1]) < len(qa[1]) and abs(len(conversations[questions[qa[0]]][1]) - len(qa[1])) <= _MAX_QA_LEN_DIFF:
                    conversations[questions[qa[0]]] = (qa[0], qa[1], qa[2])
                continue
            else:
                questions[qa[0]] = len(conversations)
            conversations.append(qa)

        conversations = normalize_conversations(conversations)
        output_path = save_to / "train_dataset.txt"
        with open(output_path, "w", encoding="utf-8") as file:
            for conversation in conversations:
                line = "<user> " + conversation[2] + " <says> " + " ".join(conversation[0]) + f" {tokenizer.eos_token} <response> " + " ".join(conversation[1]) + f" {tokenizer.eos_token}" + "\n"
                file.write(line)
        return output_path

    @staticmethod
    def normalize(text: Union[str, List]) -> Tuple[str]:
        if isinstance(text, List):
            text = " ".join([word for word in text if isinstance(word, str)])
        text = text.lower().strip()
        text = re.sub(r"[^а-яё.!?:\d]+", r" ", text) # Leave only russian and special characters
        text = re.sub(r'\.(\s*\.)+', '... ', text) # Replace any sequence of 2+ dots with '...'
        text = re.sub(r'([?!])(\s*\1)+', r'\1 ', text) # Collapse repeating ? or !
        text = re.sub(r"([!?]|\.+)", r"\1 ", text) # Separate special symbols by whitespaces
        text = re.sub(r"ё", r"е", text)
        text = re.sub(r"(.*[ауспэиычвекьхъз]{6,}.*|\b[апх][аеписх]{2,3}\b|\b[ах]{2,}\b)", r" <laugh> ", text) # Laugh token for strings such as `ахах` etc.
        text = re.sub(r"(<laugh>)(\s*\1)+", r" <laugh> ", text) # Collapse repeating <laugh> tokens
        text = re.sub(r"\s+", r" ", text).strip() # Leave only one space between each word
        return tuple(text.split())

    @staticmethod
    def normalize_username(text: str) -> Tuple[str]:
        text = text.lower()
        text = re.sub(r"[^а-яa-z\s]+", "", text).strip()
        return text