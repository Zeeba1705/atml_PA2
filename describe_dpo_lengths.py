import json
import statistics
from pathlib import Path

#describes length structure of the four DPO files
#strata rule inferred from the released stratified eval set:
#  matched if chosen/rejected token ratio in [1/1.1, 1.1], clearly longer if ratio >= 1.2 (or <= 1/1.2)

FILES = [
    "data/dpo_standard_train.jsonl",
    "data/dpo_standard_eval.jsonl",
    "data/dpo_length_balanced_train.jsonl",
    "data/dpo_length_stratified_eval.jsonl",
]


def load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok= AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        return tok, "qwen_tokens"
    except Exception as e:
        print("tokenizer unavailable, falling back to word counts:", e)
        return None, "words"


def text_of(response):
    #chosen/rejected may be a string, a message dict, or a list of messages
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content", "")
    if isinstance(response, list):
        return " ".join(m.get("content", "") for m in response if m.get("role") == "assistant") or \
            " ".join(m.get("content", "") for m in response)
    return str(response)


def length(text, tok):
    if tok is None:
        return len(text.split())
    return len(tok(text, add_special_tokens=False)["input_ids"])


def stratum(c, r):
    ratio= c / max(r, 1)
    if 1 / 1.1 <= ratio <= 1.1:
        return "length_matched"
    if ratio >= 1.2:
        return "preferred_longer"
    if ratio <= 1 / 1.2:
        return "rejected_longer"
    return "in_between"


def describe(path, tok, unit):
    rows= [json.loads(l) for l in open(path)]

    chosen= []
    rejected= []
    strata= {}
    labelled= {}

    for row in rows:
        c= length(text_of(row["chosen"]), tok)
        r= length(text_of(row["rejected"]), tok)
        chosen.append(c)
        rejected.append(r)

        s= stratum(c, r)
        strata[s]= strata.get(s, 0) + 1

        if "length_stratum" in row:
            labelled[row["length_stratum"]]= labelled.get(row["length_stratum"], 0) + 1

    n= len(rows)
    diffs= [c - r for c, r in zip(chosen, rejected)]

    info= {
        "file": path,
        "n_pairs": n,
        "unit": unit,
        "chosen_mean": round(statistics.mean(chosen), 1),
        "chosen_median": statistics.median(chosen),
        "rejected_mean": round(statistics.mean(rejected), 1),
        "rejected_median": statistics.median(rejected),
        "mean_chosen_minus_rejected": round(statistics.mean(diffs), 1),
        "frac_chosen_longer": round(sum(d > 0 for d in diffs) / n, 3),
        "inferred_strata": {k: round(v / n, 3) for k, v in sorted(strata.items())},
        "released_strata_labels": labelled or None,
    }

    if "score_chosen" in rows[0]:
        gaps= [row["score_chosen"] - row["score_rejected"] for row in rows]
        info["mean_score_gap"]= round(statistics.mean(gaps), 2)

    return info


def main():
    tok, unit= load_tokenizer()
    out= []

    for path in FILES:
        if not Path(path).exists():
            print("missing:", path)
            continue
        info= describe(path, tok, unit)
        print(json.dumps(info, indent=2))
        out.append(info)

    Path("results/task1_dpo").mkdir(parents=True, exist_ok=True)
    with open("results/task1_dpo/dataset_length_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/task1_dpo/dataset_length_stats.json")


if __name__ == "__main__":
    main()
