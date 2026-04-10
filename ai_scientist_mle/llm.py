"""
LLM client abstraction for AI Scientist MLE-Bench.

Supports OpenAI, Anthropic, DeepSeek, and Gemini models.
Adapted from Sakana AI's AI Scientist llm.py.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import backoff

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    # Anthropic models
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-mini",
    "o3-mini",
    # DeepSeek models
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
    # Google Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-pro-preview-03-25",
    # OpenRouter models
    "llama3.1-405b",
]


def create_client(model: str) -> Tuple[Any, str]:
    """Create an LLM client for the given model."""
    if model.startswith("claude"):
        import anthropic
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif "gpt" in model or "o1" in model or "o3" in model:
        import openai
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model in ("deepseek-chat", "deepseek-reasoner", "deepseek-coder"):
        import openai
        print(f"Using DeepSeek API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        ), model
    elif model == "llama3.1-405b":
        import openai
        print(f"Using OpenRouter API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        ), "meta-llama/llama-3.1-405b-instruct"
    elif "gemini" in model:
        import openai
        print(f"Using Gemini API with model {model}.")
        return openai.OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ), model
    else:
        raise ValueError(f"Model {model} not supported.")


def _is_anthropic_model(model: str) -> bool:
    return "claude" in model


def _is_openai_model(model: str) -> bool:
    return "gpt" in model


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("o1") or model.startswith("o3")


def get_response_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict]] = None,
    temperature: float = 0.75,
) -> Tuple[str, List[Dict]]:
    """Get a single response from the LLM."""
    if msg_history is None:
        msg_history = []

    if _is_anthropic_model(model):
        new_msg_history = msg_history + [
            {"role": "user", "content": [{"type": "text", "text": msg}]}
        ]
        import anthropic
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=temperature,
                system=system_message,
                messages=new_msg_history,
            )
        except anthropic.RateLimitError:
            import time
            time.sleep(10)
            response = client.messages.create(
                model=model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=temperature,
                system=system_message,
                messages=new_msg_history,
            )
        content = response.content[0].text
        new_msg_history = new_msg_history + [
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        ]
    elif _is_reasoning_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *new_msg_history,
            ],
            temperature=1,
            max_completion_tokens=MAX_NUM_TOKENS,
            n=1,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif _is_openai_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        # DeepSeek, Gemini, LLaMA, etc. (OpenAI-compatible API)
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        if model == "deepseek-reasoner":
            kwargs.pop("temperature")
            kwargs.pop("stop")
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, m in enumerate(new_msg_history):
            print(f'{j}, {m["role"]}: {str(m["content"])[:200]}')
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def get_batch_responses_from_llm(
    msg: str,
    client: Any,
    model: str,
    system_message: str,
    print_debug: bool = False,
    msg_history: Optional[List[Dict]] = None,
    temperature: float = 0.75,
    n_responses: int = 1,
) -> Tuple[List[str], List[List[Dict]]]:
    """Get N responses from the LLM (for ensembling)."""
    if msg_history is None:
        msg_history = []

    if _is_openai_model(model) and not _is_reasoning_model(model):
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_histories = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_histories = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg, client, model, system_message,
                print_debug=False, msg_history=msg_history,
                temperature=temperature,
            )
            content.append(c)
            new_msg_histories.append(hist)

    return content, new_msg_histories


def extract_json_between_markers(llm_output: str) -> Optional[Dict]:
    """Extract JSON from LLM output between ```json``` markers."""
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                return json.loads(json_string_clean)
            except json.JSONDecodeError:
                continue

    return None


def extract_code_between_markers(llm_output: str) -> Optional[str]:
    """Extract Python code from LLM output between ```python``` markers."""
    code_pattern = r"```python(.*?)```"
    matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if matches:
        return matches[0].strip()

    code_pattern = r"```(.*?)```"
    matches = re.findall(code_pattern, llm_output, re.DOTALL)
    if matches:
        return matches[0].strip()

    return None
