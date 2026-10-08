from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl
from common.logging_utils import save_json
from task1_dpo.train import git_commit
from task4_safety.generate_responses import CLASSES, POLICIES, adjust_config


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def assign_policies(ids, classes, seed):
    #each audited prompt gets the response of one policy, spread evenly: with 30 ids per class and 4 policies
    #the first class gives 8, 8, 7, 7 and the second starts two policies later, so every policy ends up with 15
    rng= np.random.default_rng(seed)
    assignment= {}

    for c in range(len(CLASSES)):
        class_ids= [i for i in ids if classes[i] == CLASSES[c]]
        order= rng.permutation(len(class_ids)).tolist()

        for position in range(len(order)):
            xstest_id= class_ids[order[position]]

            assignment[xstest_id]= POLICIES[(position + 2 * c) % len(POLICIES)]

    return assignment


def build_audit(generated, per_class, seed):
    #generated= {policy: rows of generated_<policy>.jsonl}
    ids= fixed_audit_ids(generated["sft"], per_class, seed)

    classes= {r["xstest_id"]: r["benchmark_class"] for r in generated["sft"]}
    assignment= assign_policies(ids, classes, seed)

    by_policy= {name: {r["xstest_id"]: r for r in generated[name]} for name in POLICIES}

    #shuffled so neither the class nor the policy can be read off the row order
    rng= np.random.default_rng(seed + 1)
    order= rng.permutation(len(ids)).tolist()

    sheet= []
    key= []

    for audit_id in range(len(order)):
        xstest_id= ids[order[audit_id]]
        policy= assignment[xstest_id]
        row= by_policy[policy][xstest_id]

        #what the labeller sees: no policy name, no judge label
        sheet.append({
            "audit_id": audit_id,
            "category": row["type"],
            "prompt": row["prompt"],
            "response": row["response"],
            "manual_label": "",
        })

        key.append({
            "audit_id": audit_id,
            "xstest_id": xstest_id,
            "policy": policy,
            "benchmark_class": row["benchmark_class"],
            "type": row["type"],
        })

    return ids, sheet, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--examples-per-class", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = adjust_config(load_yaml(args.config), args.smoke, args.examples_per_class)
    outdir = Path(cfg["task4_dir"])

    for name in POLICIES:
        if not (outdir / f"generated_{name}.jsonl").exists():
            raise FileNotFoundError(f"Generate/save {name} responses first: " + str(outdir / f"generated_{name}.jsonl"))

    sheet_path= outdir / "manual_audit_sheet.csv"

    if sheet_path.exists():
        existing= pd.read_csv(sheet_path, dtype=str).fillna("")

        if (existing["manual_label"].str.strip() != "").any():
            #never overwrite labels that were already typed in
            raise SystemExit(f"{sheet_path} already has manual labels in it and was left untouched.")

    generated= {name: read_jsonl(outdir / f"generated_{name}.jsonl") for name in POLICIES}

    ids, sheet, key = build_audit(
        generated,
        int(cfg["manual_audit_per_class"]),
        int(cfg["seed"])
    )

    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    pd.DataFrame(sheet).to_csv(sheet_path, index=False)

    save_json(outdir / "manual_audit_key.json", {
        "git_commit": git_commit(),
        "seed": int(cfg["seed"]),
        "per_class": int(cfg["manual_audit_per_class"]),
        "design": "each audited prompt id is paired with the response of one policy, assigned by the seed so that the policies are equally represented",
        "rows": key,
    })

    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")
    print("Wrote the sheet to label:", sheet_path)
    print("Fill the manual_label column with one of: SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS")
    print("The policy of each row and the judge labels are kept out of the sheet (manual_audit_key.json, judged_*.jsonl).")


if __name__ == "__main__":
    main()
