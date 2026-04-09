from dataclasses import dataclass
import re


ROLE_TURN_REGEX = re.compile(
    r"(?:^|\n)\s*(Human|Assistant):\s*(.*?)(?=(?:\n\s*(?:Human|Assistant):)|\Z)",
    flags=re.DOTALL,
)


@dataclass
class PreferenceTriple:
    prompt: str
    chosen: str
    rejected: str


def _extract_turns(conversation: str) -> list[tuple[str, str]]:
    turns = [(m.group(1), m.group(2).strip()) for m in ROLE_TURN_REGEX.finditer(conversation)]
    if not turns:
        raise ValueError("Could not parse Human/Assistant turns from conversation text.")
    return turns


def _render_turns(turns: list[tuple[str, str]]) -> str:
    rendered = "\n\n".join(f"{role}: {content}" for role, content in turns if content or role)
    return rendered.strip()


def split_prompt_response(conversation: str) -> tuple[str, str]:
    if not conversation or not conversation.strip():
        raise ValueError("Conversation text is empty.")

    turns = _extract_turns(conversation)

    final_assistant_idx = None
    for idx in range(len(turns) - 1, -1, -1):
        role, content = turns[idx]
        if role == "Assistant" and content.strip():
            final_assistant_idx = idx
            break

    if final_assistant_idx is None:
        raise ValueError("Could not locate a non-empty final assistant response.")

    response = turns[final_assistant_idx][1].strip()
    if not response:
        raise ValueError("Parsed response is empty.")

    prompt_turns = turns[:final_assistant_idx]
    if not prompt_turns:
        raise ValueError("Prompt is empty after removing final assistant response.")

    prompt = _render_turns(prompt_turns)
    if not prompt:
        raise ValueError("Prompt text is empty after rendering turns.")

    if not prompt.endswith("Assistant:"):
        prompt = f"{prompt}\n\nAssistant:"

    return prompt, response


def parse_hh_example(example: dict) -> PreferenceTriple:
    prompt_chosen, chosen_resp = split_prompt_response(example["chosen"])
    prompt_rejected, rejected_resp = split_prompt_response(example["rejected"])

    prompt = prompt_chosen if len(prompt_chosen) >= len(prompt_rejected) else prompt_rejected
    return PreferenceTriple(prompt=prompt, chosen=chosen_resp, rejected=rejected_resp)


def parse_hh_split(
    dataset_split,
    limit: int | None = None,
    skip_invalid: bool = True,
) -> list[PreferenceTriple]:
    triples: list[PreferenceTriple] = []
    for idx, example in enumerate(dataset_split):
        if limit is not None and idx >= limit:
            break
        try:
            triples.append(parse_hh_example(example))
        except ValueError:
            if not skip_invalid:
                raise
    return triples
