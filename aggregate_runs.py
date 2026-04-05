from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def extract_config_fields(config: dict[str, Any], folder_name: str) -> dict[str, Any]:
    algorithm = config.get("algorithm", {})
    if not isinstance(algorithm, dict):
        algorithm = {}

    row: dict[str, Any] = {
        "folder_name": folder_name,
        "seed": config.get("seed"),
        "max_steps": config.get("max_steps"),
        "game": config.get("game"),
    }

    for key, value in algorithm.items():
        row[f"{key}_config"] = value

    return row


def extract_final_exploitability(csv_path: Path) -> dict[str, Any] | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty:
        return None

    required_cols = {"expl0", "expl1"}
    if not required_cols.issubset(df.columns):
        return None

    # Take the last valid row with non-null expl0 and expl1
    valid_df = df.dropna(subset=["expl0", "expl1"])
    if valid_df.empty:
        return None

    last_row = valid_df.iloc[-1]

    expl0 = last_row["expl0"]
    expl1 = last_row["expl1"]

    return {
        "final_expl0": expl0,
        "final_expl1": expl1,
        "final_avg_exploitability": (expl0 + expl1) / 2,
        "final_global_step": last_row["global_step"] if "global_step" in valid_df.columns else None,
    }


def process_run_folder(folder: Path) -> dict[str, Any] | None:
    config_path = folder / "config.yaml"
    exploitability_path = folder / "exploitability.csv"

    if not config_path.exists() or not exploitability_path.exists():
        return None

    try:
        config = load_yaml(config_path)
        config_fields = extract_config_fields(config, folder.name)
        exploitability_fields = extract_final_exploitability(exploitability_path)

        if exploitability_fields is None:
            return None

        return {**config_fields, **exploitability_fields}
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate selected config fields and final exploitability from run folders."
    )
    parser.add_argument(
        "parent_dir",
        type=str,
        help="Path to the parent directory containing the run folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="summary.csv",
        help="Output CSV file path.",
    )
    args = parser.parse_args()

    parent_dir = Path(args.parent_dir)

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise ValueError(f"Invalid parent directory: {parent_dir}")

    rows: list[dict[str, Any]] = []

    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir():
            continue

        row = process_run_folder(child)
        if row is not None:
            rows.append(row)

    if not rows:
        print("No valid folders found. Nothing written.")
        return

    # Make sure all rows share the same columns
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())

    preferred_order = [
        "folder_name",
        "seed",
        "max_steps",
        "game",
        "final_global_step",
        "final_expl0",
        "final_expl1",
        "final_avg_exploitability",
    ]

    algorithm_columns = sorted(col for col in all_columns if col.startswith("algorithm_"))
    remaining_columns = sorted(col for col in all_columns if col not in set(preferred_order) | set(algorithm_columns))
    final_columns = preferred_order + algorithm_columns + remaining_columns

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
