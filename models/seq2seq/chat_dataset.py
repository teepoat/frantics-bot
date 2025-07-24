import torch
import torch.utils.data as data
from typing import List, Union, Tuple
from collections import OrderedDict
from .vocab import Vocab
from .custom_types import Message, MessageId, Conversation
from torch.nn.utils.rnn import pad_sequence
from .custom_types import Token
import re
import json

class ChatDataset(data.Dataset):
    def __init__(self, path: str, max_message_count: int = None, batch_size=5):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.messages: OrderedDict[MessageId, Message] = self.__load_messages_from_json(path, max_message_count)
        self.conversations: List[Conversation] = ChatDataset.__conversations_from_messages(self.messages)
        self.vocab = Vocab(list(self.messages.values())) # TODO: try changing this cast to something more applicable

        self.batches_X, self.batches_y, self.lengths, self.mask = self.__batches_from_conversations()

        self.length = len(self.batches_X)

    def __batches_from_conversations(self) -> Tuple[List[torch.LongTensor], List[torch.LongTensor], List[torch.LongTensor], List[torch.BoolTensor]]: # Shape of tensor in batch: (batch_size, max_len_in_batch)
        conversations = sorted(self.conversations, key=lambda x: len(x[0])) # Sort by input sequence length
        batches_X: List[torch.LongTensor] = list()
        batches_y: List[torch.LongTensor] = list()
        lengths: List[torch.LongTensor] = list()
        mask: List[torch.BoolTensor] = list()
        for i in range(0, len(conversations), self.batch_size):
            batches_X.append(pad_sequence([self.vocab.sentence_indices(conversations[i+j][0] + ["<eos>"]) for j in range(self.batch_size) if i+j < len(conversations)], batch_first=True, padding_value=0))
            batches_y.append(pad_sequence([self.vocab.sentence_indices(conversations[i+j][1] + ["<eos>"]) for j in range(self.batch_size) if i+j < len(conversations)], batch_first=True, padding_value=0))
            lengths.append(torch.tensor([len(conversations[i+j][0]) for j in range(self.batch_size) if i+j < len(conversations)]))
            mask.append(batches_y[-1] != Token.PAD_TOKEN.value)
        return batches_X, batches_y, lengths, mask

    @classmethod
    def __load_messages_from_json(cls, path: str, max_message_count: int = None) -> OrderedDict[MessageId, Message]:
        messages: OrderedDict[MessageId, Message] = OrderedDict()
        with open(path, "r", encoding="utf-8") as file:
            chat_json = json.load(file)
            for i, message in enumerate(chat_json["messages"]):
                if max_message_count and i == max_message_count:
                    break
                if message["type"] != "message":
                    continue
                new_message = {
                    "id": message["id"],
                    "text": cls.__normalize(message["text"])
                }
                if not new_message["text"]: # Check for empty message
                    continue
                if "reply_to_message_id" in message.keys():
                    new_message["reply_to_id"] = message["reply_to_message_id"]

                messages[new_message["id"]] = new_message
        return messages

    @classmethod
    def __conversations_from_messages(cls, messages: OrderedDict[MessageId, Message]) -> List[Conversation]:
        # Search for message with `id` in the last `current_id` messages
        def _get_message_by_id(current_id: int, id: int) -> Message:
            for i in range(current_id - 1, -1, -1):
                if messages[i]["id"] == id:
                    return messages[i]
            return None

        conversations: List[Conversation] = []

        messages_values = list(messages.values()) # TODO: try changing this cast to something more applicable
        for i in range(len(messages) - 1): # There's no answer for last message so add -1
            prev_message = messages_values[i]
            if "reply_to_id" in messages_values[i].keys(): # Message is answer for message with `id` of `reply_to_id`
                try:
                    prev_message = messages[messages_values[i]["reply_to_id"]]
                except KeyError:
                    continue
            conversations.append((prev_message["text"], messages_values[i+1]["text"]))
        return conversations

    @classmethod
    def __normalize(cls, text: Union[str, List]) -> List[str]:
        if isinstance(text, List):
            text = " ".join([word for word in text if isinstance(word, str)])
        text = text.lower().strip()
        text = re.sub(r"([.!?])", r" \1 ", text)
        text = re.sub(r"ё", r"е", text)
        text = re.sub(r"[^а-яА-я.!?]+", r" ", text)
        text = re.sub(r"\s+", r" ", text).strip()
        return text.split()

    def __getitem__(self, item):
        return self.batches_X[item], self.batches_y[item], self.lengths[item], self.mask[item]

    def __len__(self):
        return self.length