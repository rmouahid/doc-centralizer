from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import uuid4
from .message import Message

@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    messages: List[Message]

    @classmethod
    def create_new(cls, title: str = "Nouvelle conversation"):
        return cls(
            id=str(uuid4()),
            title=title,
            created_at=datetime.now().isoformat(),
            messages=[]
        )