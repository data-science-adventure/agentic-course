"""Qwen3 local Transformers agent loop with native tool-call parsing.

Run:
    uv add transformers accelerate torch langchain langsmith python-dotenv
    uv run python agent_loop_qwen3_local_native_tools.py

Optional 4-bit CUDA:
    uv add bitsandbytes
    HF_4BIT=true uv run python agent_loop_qwen3_local_native_tools.py

This implementation deliberately bypasses ChatHuggingFace.bind_tools() and
HuggingFacePipeline, using Qwen's native chat template and <tool_call> format.
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.tools import tool
from langsmith import traceable

MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
MODEL_ID = os.getenv("HF_LOCAL_MODEL", "Qwen/Qwen3-4B")
USE_4BIT = os.getenv("HF_4BIT", "false").lower() in {"1", "true", "yes"}
MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price. Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


def get_hf_tool_specs(tools):
    """Convert LangChain tools to the OpenAI-style tool schema Qwen chat templates expect."""
    specs = []
    for t in tools:
        schema = t.args_schema.model_json_schema() if t.args_schema else {"type": "object", "properties": {}}
        schema.pop("title", None)
        specs.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": schema,
            },
        })
    return specs


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs = {"device_map": "auto"}
    if USE_4BIT:
        if not torch.cuda.is_available():
            raise RuntimeError("HF_4BIT=true requires CUDA and bitsandbytes.")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16 if torch.cuda.is_available()
            else torch.float32
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return tokenizer, model


def parse_tool_calls(text):
    """Parse Qwen-style <tool_call>{JSON}</tool_call> blocks."""
    calls = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL):
        try:
            payload = json.loads(match)
            if isinstance(payload, dict) and payload.get("name"):
                calls.append(payload)
        except json.JSONDecodeError:
            print("  [Warning] Could not parse tool-call JSON:", match)
    return calls


def generate_turn(tokenizer, model, messages, tool_specs):
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tool_specs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    # Put inputs on the device holding the model's input embeddings.
    input_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=False)
    # Remove Qwen's end markers from display/parser input, while retaining tool tags.
    response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return response


@traceable(name="Qwen3 Local Transformers Native Tool Agent")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_by_name = {t.name: t for t in tools}
    tool_specs = get_hf_tool_specs(tools)
    tokenizer, model = load_model()

    print(f"Local Hugging Face model: {MODEL_ID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"4-bit quantization: {USE_4BIT}")
    print(f"Question: {question}\n" + "=" * 60)

    messages = [
        {"role": "system", "content": (
            "You are a helpful shopping assistant. Use tools to answer.\n"
            "Rules: Call get_product_price first; do not guess prices.\n"
            "Call apply_discount only after receiving the exact price from get_product_price.\n"
            "Never calculate discounts yourself. If no discount tier is specified, ask.\n"
            "Use one tool call at a time. When calling a tool, emit a native tool call."
        )},
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        response = generate_turn(tokenizer, model, messages, tool_specs)
        print("  [Model output]", response)
        tool_calls = parse_tool_calls(response)

        if not tool_calls:
            # Strip Qwen's thinking block from user-facing answer, if present.
            final = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print(f"\nFinal Answer: {final}")
            return final

        # This experiment intentionally executes only the first tool call per turn.
        call = tool_calls[0]
        name = call.get("name")
        args = call.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        print(f"  [Tool Selected] {name} with args: {args}")
        tool_obj = tools_by_name.get(name)
        if tool_obj is None:
            raise ValueError(f"Unknown tool: {name}")
        observation = tool_obj.invoke(args)
        print(f"  [Tool Result] {observation}")

        # Add the assistant's native tool-call message, then the tool response.
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "name": name, "content": str(observation)})

    print("ERROR: Max iterations reached without a final answer")
    return None


def main():
    print("LangChain tools + local Hugging Face Transformers (native Qwen tool format)")
    run_agent("What is the price of a laptop after applying a gold discount?")

if __name__ == "__main__":
    main()