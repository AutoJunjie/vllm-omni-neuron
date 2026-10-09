# SPDX-License-Identifier: Apache-2.0
"""Summarize the published position metrics without running model inference."""

import csv
import json
import statistics
from pathlib import Path


def main():
    path = Path(__file__).with_name("position-summary.csv")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    cases = []
    for language in dict.fromkeys(row["language"] for row in rows):
        case = [row for row in rows if row["language"] == language]
        errors = [float(row["raw_relative_l2"]) for row in case]
        maximum = max(case, key=lambda row: float(row["raw_relative_l2"]))
        cases.append(
            {
                "language": language,
                "positions": len(case),
                "raw_l2_failures": sum(error >= 0.03 for error in errors),
                "raw_l2_median": statistics.median(errors),
                "raw_l2_maximum": max(errors),
                "raw_l2_maximum_step": int(maximum["step_0_based"]),
                "exact_top1_matches": sum(
                    int(row["gold_top1"]) == int(row["trn_top1"]) for row in case
                ),
                "public_passes": sum(row["public_passed"] == "True" for row in case),
            }
        )

    cross_tabulation = {
        f"raw_{raw}_public_{public}": sum(
            (float(row["raw_relative_l2"]) < 0.03) == (raw == "pass")
            and (row["public_passed"] == "True") == (public == "pass")
            for row in rows
        )
        for raw in ("pass", "fail")
        for public in ("pass", "fail")
    }
    print(
        json.dumps(
            {
                "scope": "Aggregation of captured primary-reference metrics; no new inference",
                "positions": len(rows),
                "raw_l2_failures": sum(case["raw_l2_failures"] for case in cases),
                "exact_top1_matches": sum(case["exact_top1_matches"] for case in cases),
                "public_passes": sum(case["public_passes"] for case in cases),
                "cases": cases,
                "cross_tabulation": cross_tabulation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
