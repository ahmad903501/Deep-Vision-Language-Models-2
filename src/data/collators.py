from dataclasses import dataclass

import torch


def assert_tokenizer_contract(tokenizer, expected_padding_side: str) -> None:
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is not set. Set pad token = eos token.")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id is not set.")
    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        raise ValueError("Tokenizer contract violated: pad token must equal eos token.")
    if tokenizer.padding_side != expected_padding_side:
        raise ValueError(
            f"Tokenizer padding_side mismatch. Expected '{expected_padding_side}', "
            f"got '{tokenizer.padding_side}'."
        )


@dataclass
class SFTCollator:
    tokenizer: object
    max_length: int = 1024

    def __post_init__(self) -> None:
        assert_tokenizer_contract(self.tokenizer, expected_padding_side="left")

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [item["prompt"] for item in batch]
        full_texts = [item["full_text"] for item in batch]

        model_inputs = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        prompt_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        labels[model_inputs["attention_mask"] == 0] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
        for i, prompt_len in enumerate(prompt_lengths.tolist()):
            labels[i, :prompt_len] = -100

        response_mask = (labels != -100).to(torch.long)
        model_inputs["labels"] = labels
        model_inputs["response_mask"] = response_mask
        model_inputs["prompt_lengths"] = prompt_lengths
        model_inputs["raw_prompts"] = prompts
        model_inputs["raw_full_texts"] = full_texts
        return model_inputs


@dataclass
class RMCollator:
    tokenizer: object
    max_length: int = 1024

    def __post_init__(self) -> None:
        assert_tokenizer_contract(self.tokenizer, expected_padding_side="right")

    def __call__(self, batch: list[dict]) -> dict:
        chosen_texts = [item["chosen_text"] for item in batch]
        rejected_texts = [item["rejected_text"] for item in batch]

        chosen_inputs = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected_inputs = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        chosen_last_non_pad = chosen_inputs["attention_mask"].sum(dim=1) - 1
        rejected_last_non_pad = rejected_inputs["attention_mask"].sum(dim=1) - 1

        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "chosen_last_non_pad_idx": chosen_last_non_pad,
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
            "rejected_last_non_pad_idx": rejected_last_non_pad,
            "raw_chosen_texts": chosen_texts,
            "raw_rejected_texts": rejected_texts,
        }


@dataclass
class DPOCollator:
    tokenizer: object
    max_length: int = 1024

    def __post_init__(self) -> None:
        assert_tokenizer_contract(self.tokenizer, expected_padding_side="left")

    def __call__(self, batch: list[dict]) -> dict:
        prompts = [item["prompt"] for item in batch]
        chosen_texts = [item["chosen_text"] for item in batch]
        rejected_texts = [item["rejected_text"] for item in batch]

        chosen_inputs = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        rejected_inputs = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        prompt_inputs = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        response_start = prompt_inputs["attention_mask"].sum(dim=1)

        return {
            "chosen_input_ids": chosen_inputs["input_ids"],
            "chosen_attention_mask": chosen_inputs["attention_mask"],
            "rejected_input_ids": rejected_inputs["input_ids"],
            "rejected_attention_mask": rejected_inputs["attention_mask"],
            "response_start": response_start,
            "raw_prompts": prompts,
            "raw_chosen_texts": chosen_texts,
            "raw_rejected_texts": rejected_texts,
        }
