from app.history import ConversationHistory


def test_new_history_creates_default_conversation(tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))

    assert history.active_conversation_id is not None
    assert len(history.conversations) == 1


def test_add_interaction_appends_message_and_persists(tmp_path):
    history_file = tmp_path / "history.json"
    history = ConversationHistory(history_file=str(history_file))

    history.add_interaction("query", "réponse", "contexte")

    messages = history.get_active_messages()
    assert len(messages) == 1
    assert messages[0].query == "query"
    assert messages[0].response == "réponse"

    reloaded = ConversationHistory(history_file=str(history_file))
    assert len(reloaded.get_active_messages()) == 1


def test_delete_conversation_switches_active_conversation(tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    first_id = history.active_conversation_id
    second_id = history.create_conversation("Autre conversation")

    history.delete_conversation(second_id)
    assert history.active_conversation_id == first_id

    history.delete_conversation(first_id)
    assert history.active_conversation_id is None
    assert history.conversations == {}


def test_reset_clears_all_conversations(tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    history.add_interaction("query", "réponse", "contexte")
    history.create_conversation("Autre conversation")
    assert len(history.conversations) == 2

    history.reset()

    assert len(history.conversations) == 1
    assert history.get_active_messages() == []


def test_conversation_ids_are_unique(tmp_path):
    history = ConversationHistory(history_file=str(tmp_path / "history.json"))
    ids = {history.active_conversation_id}
    for _ in range(5):
        ids.add(history.create_conversation())

    assert len(ids) == 6
