from torch.utils.data import Dataset

from .hh_parser import PreferenceTriple


class SFTDataset(Dataset):
    def __init__(self, triples: list[PreferenceTriple]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        row = self.triples[idx]
        return {
            "prompt": row.prompt,
            "response": row.chosen,
            "full_text": f"{row.prompt} {row.chosen}",
        }


class RMDataset(Dataset):
    def __init__(self, triples: list[PreferenceTriple]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        row = self.triples[idx]
        return {
            "prompt": row.prompt,
            "chosen": row.chosen,
            "rejected": row.rejected,
            "chosen_text": f"{row.prompt} {row.chosen}",
            "rejected_text": f"{row.prompt} {row.rejected}",
        }


class DPODataset(Dataset):
    def __init__(self, triples: list[PreferenceTriple]):
        self.triples = triples

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> dict:
        row = self.triples[idx]
        return {
            "prompt": row.prompt,
            "chosen": row.chosen,
            "rejected": row.rejected,
            "chosen_text": f"{row.prompt} {row.chosen}",
            "rejected_text": f"{row.prompt} {row.rejected}",
        }
