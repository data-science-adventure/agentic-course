"""Module Utility: Message History Inspector.

This module provides utility functions to serialize and export LangChain / LangGraph
message state histories into formatted JSON files for inspection and debugging.
"""

import json
from typing import Any, List


def save_messages_to_json(messages: List[Any], filename: str = "inspect.json") -> None:
    """Converts a list of LangChain message objects into a formatted JSON file.

    Iterates through message objects (such as `HumanMessage`, `AIMessage`, or `ToolMessage`),
    converts them into Python dictionaries preserving compatibility with both Pydantic v1
    and Pydantic v2 serialization, injects explicit message type annotations, and writes
    the formatted array to a JSON file.

    Args:
        messages (List[Any]): List of message objects returned in `response["messages"]`
            or from a LangGraph state graph.
        filename (str, optional): The destination JSON file path. Defaults to 'inspect.json'.

    Returns:
        None

    Raises:
        OSError: If writing to the specified file fails.
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