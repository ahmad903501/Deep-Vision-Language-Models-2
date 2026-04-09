import json
from datetime import datetime, timezone
from pathlib import Path


def build_manifest(
    stage: str,
    base_checkpoint: str,
    data_domain: str,
    parent_stages: list[str] | None = None,
    notes: str | None = None,
) -> dict:
    parent_stages = parent_stages or []
    lineage = parent_stages + [stage]
    return {
        "stage": stage,
        "base_checkpoint": base_checkpoint,
        "data_domain": data_domain,
        "parent_stages": parent_stages,
        "lineage": lineage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes or "",
    }


def save_checkpoint_manifest(manifest_path: str, manifest: dict) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_checkpoint_manifest(manifest_path: str) -> dict:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_rlvr_sft_only(manifest: dict) -> None:
    forbidden = {"ppo", "dpo", "grpo", "rlvr"}
    lineage = [stage.lower() for stage in manifest.get("lineage", [])]

    if not lineage:
        raise ValueError("Checkpoint manifest has empty lineage.")

    if lineage[-1] != "sft":
        raise ValueError(
            "RLVR must start from SFT checkpoint. "
            f"Current checkpoint stage is '{lineage[-1]}'."
        )

    touched_forbidden = forbidden.intersection(set(lineage))
    if touched_forbidden:
        raise ValueError(
            "RLVR checkpoint lineage is contaminated with prior alignment stages: "
            f"{sorted(touched_forbidden)}"
        )
