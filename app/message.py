from dataclasses import dataclass

@dataclass
class Message:
    query: str
    response: str
    context: str
    timestamp: str