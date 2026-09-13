from datetime import datetime
import json
import os
from typing import Dict, List, Optional
from .conversation import Conversation
from .message import Message
from .config import HISTORY_FILE

class ConversationHistory:
    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file
        self.conversations: Dict[str, Conversation] = {}
        self.active_conversation_id: Optional[str] = None

        # Create data directory if it doesn't exist
        history_dir = os.path.dirname(self.history_file)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        
        # Initialize empty history file if it doesn't exist
        if not os.path.exists(self.history_file):
            self._initialize_empty_history()
        
        # Load existing history
        self.load_history()
        
        # Create initial conversation if none exists
        if not self.conversations:
            self.create_conversation()
    
    def _initialize_empty_history(self) -> None:
        """Initialize an empty history file with the correct structure"""
        empty_history = {
            "active_conversation": None,
            "conversations": {}
        }
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(empty_history, f, ensure_ascii=False, indent=2)
    
    def create_conversation(self, title: str = "Nouvelle conversation") -> str:
        """Crée une nouvelle conversation et la définit comme active"""
        conversation = Conversation.create_new(title)
        self.conversations[conversation.id] = conversation
        self.active_conversation_id = conversation.id
        self.save_history()
        return conversation.id
    
    def reset(self) -> None:
        """Efface toutes les conversations et repart d'une conversation vide"""
        self.conversations = {}
        self.active_conversation_id = None
        self.create_conversation()

    def delete_conversation(self, conversation_id: str) -> None:
        """Supprime une conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id == self.active_conversation_id:
                self.active_conversation_id = next(iter(self.conversations.keys())) if self.conversations else None
            self.save_history()
    
    def get_conversations_list(self) -> List[dict]:
        """Retourne la liste des conversations pour l'affichage"""
        return [
            {
                "id": conv.id,
                "title": conv.title,
                "created_at": conv.created_at,
                "message_count": len(conv.messages)
            }
            for conv in self.conversations.values()
        ]
    
    def add_interaction(self, query: str, response: str, context: str) -> None:
        """Ajoute une interaction à la conversation active"""
        if not self.active_conversation_id:
            self.create_conversation()
            
        message = Message(
            query=query,
            response=response,
            context=context,
            timestamp=datetime.now().isoformat()
        )
        self.conversations[self.active_conversation_id].messages.append(message)
        self.save_history()
    
    def save_history(self) -> None:
        """Sauvegarde l'historique des conversations au format JSON"""
        data = {
            "active_conversation": self.active_conversation_id,
            "conversations": {
                conv_id: {
                    "id": conv.id,
                    "title": conv.title,
                    "created_at": conv.created_at,
                    "messages": [msg.__dict__ for msg in conv.messages]
                }
                for conv_id, conv in self.conversations.items()
            }
        }
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_history(self) -> None:
        """Charge l'historique depuis le fichier JSON"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure the data has the correct structure
                if not isinstance(data, dict):
                    self._initialize_empty_history()
                    return
                
                self.active_conversation_id = data.get("active_conversation")
                conversations_data = data.get("conversations", {})
                
                self.conversations = {
                    conv_id: Conversation(
                        id=conv_data["id"],
                        title=conv_data["title"],
                        created_at=conv_data["created_at"],
                        messages=[Message(**msg) for msg in conv_data["messages"]]
                    )
                    for conv_id, conv_data in conversations_data.items()
                }
                
                # If no active conversation is set but conversations exist
                if not self.active_conversation_id and self.conversations:
                    self.active_conversation_id = next(iter(self.conversations.keys()))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            self._initialize_empty_history()
            self.conversations = {}
            self.active_conversation_id = None
    
    def get_active_messages(self) -> List[Message]:
        """Returns messages from the active conversation"""
        if not self.active_conversation_id:
            return []
        return self.conversations[self.active_conversation_id].messages