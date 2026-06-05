from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine stress experiment summaries.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "mixed_stress=results/stress_mixed_200/timeseries_summary.csv",
            "cpu_pressure=results/stress_cpu_200/timeseries_summary.csv",
            "memory_pressure=results/stress_memory_200/timeseries_summary.csv",
            "disk_pressure=results/stress_disk_200/timeseries_summary.csv",
            "rack_pressure=results/stress_rack_200/timeseries_summary.csv",
        ],
    )
    parser.add_argument("--output", default="results/stress_scenario_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in args.inputs:
        scenario, path_text = item.split("=", 1)
        path = Path(path_text)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row = {"scenario": scenario, **row}
                rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output.resolve()}.")


if __name__ == "__main__":
    main()
