from datasets import DatasetDict, load_dataset


def _validate_hh_split(dataset_dict: DatasetDict) -> DatasetDict:
    required_columns = {"chosen", "rejected"}
    for split_name in ["train", "test"]:
        if split_name not in dataset_dict:
            raise ValueError(f"Expected split '{split_name}' in HH dataset.")
        split_columns = set(dataset_dict[split_name].column_names)
        if not required_columns.issubset(split_columns):
            raise ValueError(
                f"Split '{split_name}' is missing columns {required_columns - split_columns}."
            )
    return dataset_dict


def load_hh_harmless_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    harmless_config: str = "harmless-base",
    cache_dir: str | None = None,
) -> DatasetDict:
    """Load HH-RLHF harmless split with a few robust fallbacks."""
    loaders = [
        lambda: load_dataset(dataset_name, data_dir=harmless_config, cache_dir=cache_dir),
        lambda: load_dataset(dataset_name, harmless_config, cache_dir=cache_dir),
        lambda: load_dataset(dataset_name, cache_dir=cache_dir),
    ]

    last_error = None
    for loader in loaders:
        try:
            dataset = loader()
            if isinstance(dataset, DatasetDict):
                return _validate_hh_split(dataset)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        f"Unable to load HH-RLHF harmless dataset from '{dataset_name}'. Last error: {last_error}"
    )
