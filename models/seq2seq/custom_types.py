from typing import List, TypedDict, NotRequired, Tuple
from enum import Enum, auto

MessageId = int
MessageText = List[str]
Conversation = Tuple[MessageText]


class Message(TypedDict):
    id: MessageId
    text: MessageText
    reply_to_id: NotRequired[int]

class Method(Enum):
    DOT = auto()
    GENERAL = auto()
    CONCAT = auto()

class Token(Enum):
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3