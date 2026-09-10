import json
from typing import List, Any

def save_messages_to_json(messages: List[Any], filename: str = "inspect.json") -> None:
    """
    Converts a list of LangChain message objects into a JSON file.

    Parameters:
        messages (List[Any]): List of message objects returned in response["messages"].
        filename (str): The destination JSON file name. Defaults to 'inspect.json'.
    """
    json_data = []

    for msg in messages:
        # Convert message object to dict (supports Pydantic v1 and v2)
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg_dict = msg.dict()
        else:
            msg_dict = dict(msg)

        # Explicitly set 'type' to the class name (e.g., 'HumanMessage', 'AIMessage', 'ToolMessage')
        msg_dict["type"] = msg.__class__.__name__

        json_data.append(msg_dict)

    # Save formatted array to JSON file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(messages)} messages to '{filename}'.")