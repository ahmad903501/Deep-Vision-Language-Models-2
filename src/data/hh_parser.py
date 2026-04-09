from dataclasses import dataclass
import re


ASSISTANT_MARKER_REGEX = re.compile(r"(?m)(?:(?<=^)|(?<=\n))[ \t]*Assistant:")


@dataclass
class PreferenceTriple:
    prompt: str
    chosen: str
    rejected: str


def _find_last_assistant_marker(conversation: str) -> re.Match:
    matches = list(ASSISTANT_MARKER_REGEX.finditer(conversation))
    if not matches:
        raise ValueError("Could not locate assistant marker in conversation text.")
    return matches[-1]


def split_prompt_response(conversation: str) -> tuple[str, str]:
    if conversation is None or conversation == "":
        raise ValueError("Conversation text is empty.")

    marker = _find_last_assistant_marker(conversation)

    # Preserve raw formatting exactly: no stripping or normalization.
    prompt = conversation[: marker.end()]
    response = conversation[marker.end() :]

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
