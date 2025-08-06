from typing import List, NotRequired, Tuple, TypedDict, Union


MessageId = int
MessageText = Union[List[str], Tuple[str]]
Conversation = Tuple[MessageText]


class Message(TypedDict):
    id: MessageId
    text: MessageText
    reply_to_id: NotRequired[int]