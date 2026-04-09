from torch.utils.data import DataLoader

from src.config.schema import DataConfig
from src.data.collators import DPOCollator, RMCollator, SFTCollator
from src.data.datasets import DPODataset, RMDataset, SFTDataset


def build_dataloaders(
    triples,
    policy_tokenizer,
    rm_tokenizer,
    data_config: DataConfig,
):
    sft_dataset = SFTDataset(triples)
    rm_dataset = RMDataset(triples)
    dpo_dataset = DPODataset(triples)

    sft_loader = DataLoader(
        sft_dataset,
        batch_size=data_config.sft_batch_size,
        shuffle=False,
        collate_fn=SFTCollator(policy_tokenizer, max_length=data_config.max_seq_len),
    )
    rm_loader = DataLoader(
        rm_dataset,
        batch_size=data_config.rm_batch_size,
        shuffle=False,
        collate_fn=RMCollator(rm_tokenizer, max_length=data_config.max_seq_len),
    )
    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=data_config.dpo_batch_size,
        shuffle=False,
        collate_fn=DPOCollator(policy_tokenizer, max_length=data_config.max_seq_len),
    )

    return {
        "sft": sft_loader,
        "rm": rm_loader,
        "dpo": dpo_loader,
    }
