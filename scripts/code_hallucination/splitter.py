"""Phase 8: Train/dev/test split by SWE-bench splits.

Since we use SWE-bench splits directly, this phase mostly validates
the splits and selects which instances get hallucination injection.
"""

import random

from .config import HALLUCINATION_RATIO


def select_hallucination_targets(
    instances: list[dict],
    ratio: float = HALLUCINATION_RATIO,
    seed: int = 42,
) -> set[str]:
    """Select which instances get hallucination injection.

    Applies the ratio uniformly within each split to maintain
    consistent class distribution across train/dev/test.

    Returns set of instance_ids that should be hallucinated.
    """
    rng = random.Random(seed)
    targets = set()

    # Group by split
    by_split = {}
    for inst in instances:
        split = inst["split"]
        if split not in by_split:
            by_split[split] = []
        by_split[split].append(inst)

    for split, split_instances in by_split.items():
        n_hall = int(len(split_instances) * ratio)
        rng.shuffle(split_instances)
        for inst in split_instances[:n_hall]:
            targets.add(inst["instance_id"])

        n_clean = len(split_instances) - n_hall
        print(f"  {split}: {n_hall} hallucinated + {n_clean} clean = {len(split_instances)}")

    return targets


def select_format_targets(
    format_entries: list[dict],
    ratio: float = HALLUCINATION_RATIO,
    seed: int = 42,
) -> set[str]:
    """Select which format entries (sub-instances) get hallucination injection.

    Groups by split (read from each entry's ``split`` field, falling back to
    ``"train"``) to maintain a consistent class ratio across splits.

    Returns a set of format entry instance_ids (may be sub-instance IDs).
    """
    rng = random.Random(seed)
    targets: set[str] = set()

    by_split: dict[str, list[dict]] = {}
    for entry in format_entries:
        split = entry.get("split", "train")
        by_split.setdefault(split, []).append(entry)

    for split, entries in by_split.items():
        n_hall = int(len(entries) * ratio)
        rng.shuffle(entries)
        for entry in entries[:n_hall]:
            targets.add(entry["instance_id"])
        n_clean = len(entries) - n_hall
        print(f"  {split}: {n_hall} hallucinated + {n_clean} clean = {len(entries)} format entries")

    return targets


def run(instances: list[dict]) -> set[str]:
    """Run Phase 8: Select hallucination targets."""
    print("=" * 60)
    print("Phase 8: Split & Target Selection")
    print("=" * 60)

    targets = select_hallucination_targets(instances)
    print(f"\nTotal hallucination targets: {len(targets)} out of {len(instances)}")
    print(f"Ratio: {len(targets) / max(len(instances), 1):.1%}")

    return targets


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
