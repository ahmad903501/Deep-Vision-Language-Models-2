import pytest

from src.data.hh_loader import load_hh_harmless_dataset
from src.data.hh_parser import parse_hh_example, parse_hh_split


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
    assert parsed.prompt.strip().endswith("Assistant:")
    assert "Human:" in parsed.prompt
    assert len(parsed.chosen.strip()) > 0
    assert len(parsed.rejected.strip()) > 0


def test_parse_hh_split_on_dataset_subset_returns_valid_triples():
    train = _load_small_train_split(max_rows=128)
    triples = parse_hh_split(train, limit=128, skip_invalid=True)

    assert len(triples) > 0
    for tri in triples[:5]:
        assert tri.prompt.strip().endswith("Assistant:")
        assert len(tri.chosen.strip()) > 0
        assert len(tri.rejected.strip()) > 0


def test_parse_hh_split_strict_mode_raises_for_invalid_rows():
    bad_rows = [
        {"chosen": "Human: hi\n\nAssistant:", "rejected": "Human: hi\n\nAssistant:"},
    ]

    with pytest.raises(ValueError):
        _ = parse_hh_split(bad_rows, skip_invalid=False)
