import pytest

from src.data.hh_loader import load_hh_harmless_dataset
from src.data.hh_parser import parse_hh_example, parse_hh_split, split_prompt_response


def _load_small_train_split(max_rows: int = 64):
    try:
        ds = load_hh_harmless_dataset()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load HH-RLHF dataset in this environment: {exc}")

    train = ds["train"]
    if len(train) == 0:
        pytest.skip("HH-RLHF train split is empty.")

    return train.select(range(min(max_rows, len(train))))


def test_parse_single_dataset_example_uses_last_assistant_response():
    train = _load_small_train_split(max_rows=128)

    parsed = None
    for row in train:
        try:
            parsed = parse_hh_example(row)
            break
        except ValueError:
            continue

    assert parsed is not None, "Could not parse any sample from the HH-RLHF train subset."
    assert parsed.prompt.endswith("Assistant:")
    assert "Human:" in parsed.prompt
    assert isinstance(parsed.chosen, str)
    assert isinstance(parsed.rejected, str)


def test_parse_hh_split_on_dataset_subset_returns_valid_triples():
    train = _load_small_train_split(max_rows=128)
    triples = parse_hh_split(train, limit=128, skip_invalid=True)

    assert len(triples) > 0
    for tri in triples[:5]:
        assert tri.prompt.endswith("Assistant:")
        assert isinstance(tri.chosen, str)
        assert isinstance(tri.rejected, str)


def test_parse_hh_split_strict_mode_raises_for_invalid_rows():
    bad_rows = [
        {"chosen": "Human: hi", "rejected": "Human: hi"},
    ]

    with pytest.raises(ValueError):
        _ = parse_hh_split(bad_rows, skip_invalid=False)


def test_split_prompt_response_preserves_raw_whitespace_and_allows_empty_response():
    convo = "Human: A\n\nAssistant: first\n\nHuman: B\n\nAssistant:"
    prompt, response = split_prompt_response(convo)

    assert prompt == "Human: A\n\nAssistant: first\n\nHuman: B\n\nAssistant:"
    assert response == ""


def test_split_prompt_response_preserves_newlines_spaces_in_final_response():
    convo = "Human: hi\n\nAssistant: ok\n\nHuman: again\n\nAssistant:\n   \nThanks"
    prompt, response = split_prompt_response(convo)

    assert prompt == "Human: hi\n\nAssistant: ok\n\nHuman: again\n\nAssistant:"
    assert response == "\n   \nThanks"
