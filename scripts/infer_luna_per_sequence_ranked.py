#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.luna_dataset import LunaOrganized
from core.sdg_depth.net import SDGDepth
from scripts.infer_luna_top5 import (
    EXCLUDED_SEQUENCES,
    infer_sample,
    save_comparison,
)
from scripts.infer_one_luna import model_and_dataset_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Save the top and middle-ranked Luna test samples for each sequence."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    return parser.parse_args()


def select_ranked_samples(metrics, count):
    by_sequence = defaultdict(list)
    for item in metrics:
        by_sequence[item["sequence"]].append(dict(item))

    selections = defaultdict(list)
    summary = {}
    for sequence, items in sorted(by_sequence.items()):
        ranked = sorted(items, key=lambda item: item["mape_percent"])
        for rank, item in enumerate(ranked, start=1):
            item["sequence_rank"] = rank

        selected_count = min(count, len(ranked))
        top = ranked[:selected_count]
        middle_start = max(0, (len(ranked) - selected_count) // 2)
        middle = ranked[middle_start : middle_start + selected_count]
        for group, selected in (("top10", top), ("middle10", middle)):
            for item in selected:
                key = (item["sequence"], item["frame"])
                selections[key].append(
                    {
                        "group": group,
                        "sequence_rank": item["sequence_rank"],
                        "metrics": item,
                    }
                )
        summary[sequence] = {
            "test_samples": len(ranked),
            "top10_ranks": [item["sequence_rank"] for item in top],
            "middle10_ranks": [item["sequence_rank"] for item in middle],
        }
    return selections, summary


def save_selected_result(output_dir, assignment, result):
    metrics = dict(result["metrics"])
    metrics["selection_group"] = assignment["group"]
    metrics["sequence_rank"] = assignment["sequence_rank"]
    sample_dir = (
        output_dir
        / metrics["sequence"]
        / assignment["group"]
        / f'rank{assignment["sequence_rank"]:02d}_frame{metrics["frame"]}'
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_comparison(sample_dir / "comparison.png", result)
    with open(sample_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main():
    cli = parse_args()
    if cli.count <= 0:
        raise ValueError("--count must be positive")
    with open(cli.metrics, "r", encoding="utf-8") as handle:
        all_metrics = json.load(handle)
    selections, sequence_summary = select_ranked_samples(all_metrics, cli.count)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    args = model_and_dataset_args()
    args.luna_val_fraction = 0.2
    args.luna_test_fraction = 0.2
    args.luna_exclude_sequences = EXCLUDED_SEQUENCES
    dataset = LunaOrganized(
        aug_params={},
        root=str(cli.root),
        image_set="test",
        args=args,
    )
    sample_indices = {
        (sample["sequence"], sample["frame"]): index
        for index, sample in enumerate(dataset.extra_info)
    }
    missing = sorted(set(selections) - set(sample_indices))
    if missing:
        raise RuntimeError(f"Selected samples are missing from the test split: {missing}")

    device = torch.device("cuda")
    model = SDGDepth(args.max_disp, use_concat_volume=True, args=args).to(device)
    checkpoint = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    saved_metrics = []
    selected_keys = sorted(selections, key=lambda key: sample_indices[key])
    for progress, key in enumerate(selected_keys, start=1):
        result = infer_sample(model, dataset[sample_indices[key]], device)
        if result is None:
            raise RuntimeError(f"No valid evaluation pixels for {key}")
        for assignment in selections[key]:
            saved_metrics.append(
                save_selected_result(cli.output_dir, assignment, result)
            )
        groups = ",".join(item["group"] for item in selections[key])
        print(
            f"[{progress}/{len(selected_keys)}] {key[0]}/frame{key[1]} "
            f"MAPE={result['metrics']['mape_percent']:.4f}% groups={groups}"
        )

    saved_metrics.sort(
        key=lambda item: (
            item["sequence"],
            item["selection_group"],
            item["sequence_rank"],
        )
    )
    with open(
        cli.output_dir / "selected_metrics.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(saved_metrics, handle, indent=2)
    with open(
        cli.output_dir / "selected_metrics.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(saved_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(saved_metrics)

    summary = {
        "checkpoint": str(cli.checkpoint),
        "source_metrics": str(cli.metrics),
        "split": "test",
        "excluded_sequences": EXCLUDED_SEQUENCES,
        "requested_count_per_group": cli.count,
        "unique_samples_inferred": len(selected_keys),
        "saved_group_entries": len(saved_metrics),
        "sequences": sequence_summary,
    }
    with open(cli.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
