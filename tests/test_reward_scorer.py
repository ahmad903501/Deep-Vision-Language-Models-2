from types import SimpleNamespace

import pytest
import torch

from src.models.reward_scorer import (
    configure_reward_trainable_params,
    gather_last_token_hidden,
    last_non_pad_indices,
    reward_scores_from_batch,
)


class DummyRewardModel(torch.nn.Module):
    def __init__(self, hidden_state: torch.Tensor):
        super().__init__()
        self._hidden_state = hidden_state
        self.score = torch.nn.Linear(hidden_state.shape[-1], 1, bias=False)
        with torch.no_grad():
            self.score.weight.zero_()
            self.score.weight[0, 0] = 1.0

    def forward(self, input_ids, attention_mask, output_hidden_states=True, return_dict=True):  # noqa: ARG002
        return SimpleNamespace(hidden_states=(self._hidden_state,))


class DummyTrainableRewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.score = torch.nn.Linear(4, 1)


def test_last_non_pad_indices_right_padding():
    attention = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    idx = last_non_pad_indices(attention)
    assert idx.tolist() == [2, 1]


def test_gather_last_token_hidden_uses_provided_indices():
    hidden = torch.tensor(
        [
            [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]],
            [[3.0, 13.0], [4.0, 14.0], [5.0, 15.0]],
        ]
    )
    idx = torch.tensor([1, 0], dtype=torch.long)
    gathered = gather_last_token_hidden(hidden, idx)
    assert gathered.tolist() == [[1.0, 11.0], [3.0, 13.0]]


def test_reward_scores_from_batch_computes_on_last_non_pad_token():
    hidden = torch.tensor(
        [
            [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]],
            [[11.0, 0.0], [21.0, 0.0], [31.0, 0.0], [41.0, 0.0]],
        ]
    )
    model = DummyRewardModel(hidden)

    input_ids = torch.zeros((2, 4), dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    scores = reward_scores_from_batch(model, input_ids, attention_mask)

    # Row0 picks token 1 => 20.0, Row1 picks token 2 => 31.0
    assert torch.allclose(scores, torch.tensor([20.0, 31.0]))


def test_configure_reward_trainable_params_head_only_sets_only_head_trainable():
    model = DummyTrainableRewardModel()
    # Start with all params trainable.
    for p in model.parameters():
        p.requires_grad = True

    configure_reward_trainable_params(model, "head_only")

    backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
    head_trainable = all(p.requires_grad for p in model.score.parameters())

    assert not backbone_trainable
    assert head_trainable


def test_configure_reward_trainable_params_invalid_mode_raises():
    model = DummyTrainableRewardModel()
    with pytest.raises(ValueError):
        configure_reward_trainable_params(model, "unknown_mode")
