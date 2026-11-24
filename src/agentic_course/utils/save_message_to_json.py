"""Module Utility: Message History Inspector.

Serializes and exports LangChain / LangGraph message state histories into formatted
JSON files named after the calling script under `agent_output/`.
"""

import inspect
import json
from pathlib import Path
from typing import Any


def _resolve_caller_stem() -> str:
    """Resolve the filename stem of the external script that invoked this utility."""
    current_file = Path(__file__).resolve()
    for frame_info in inspect.stack():
        caller_path = Path(frame_info.filename).resolve()
        if caller_path != current_file:
            return caller_path.stem
    return "unknown_caller"


def save_messages_to_json(
    messages: list[Any],
    filename: str | None = None,
    output_dir: str | Path = "agent_output",
) -> Path:
    """Convert a list of LangChain message objects into a JSON file in `agent_output/`.

    When `filename` is omitted, the destination path defaults to:
    `agent_output/<caller_script_stem>_messages.json`.

    Args:
        messages: List of message objects returned in `response["messages"]` or from
            a LangGraph state graph.
        filename: Optional explicit filename or path. If a bare filename without a
            parent directory is provided (e.g., `"custom.json"`), it is still placed
            inside `output_dir`.
        output_dir: Directory where message inspection JSON files are stored.
            Defaults to `"agent_output"`.

    Returns:
        Path: The resolved path to the written JSON file.

    Raises:
        OSError: If creating the output directory or writing the JSON file fails.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        caller_stem = _resolve_caller_stem()
        target_path = target_dir / f"{caller_stem}_messages.json"
    else:
        provided = Path(filename)
        # Keep explicit subdirectory paths if provided; otherwise place in output_dir
        target_path = provided if provided.parent != Path(".") else target_dir / provided.name

    json_data: list[dict[str, Any]] = []

    for msg in messages:
        # Support both Pydantic v2 (model_dump) and Pydantic v1 (dict) message models
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg_dict = msg.dict()
        else:
            msg_dict = dict(msg)

        # Explicitly record the message class name for clear trace inspection
        msg_dict["type"] = msg.__class__.__name__
        json_data.append(msg_dict)

    with target_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(messages)} messages to '{target_path}'.")
    return target_path