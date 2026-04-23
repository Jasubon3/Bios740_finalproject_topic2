import json
from pathlib import Path
from collections import Counter, defaultdict


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entity_key(ent):
    return (ent["start"], ent["end"], ent["type"])


def relation_key(rel, entities):
    h = entities[rel["head"]]
    t = entities[rel["tail"]]
    return (
        rel["type"],
        h["start"], h["end"], h["type"],
        t["start"], t["end"], t["type"]
    )


def span_text(tokens, start, end):
    return " ".join(tokens[start:end])


def analyze_dataset(name, gold_path, pred_path, out_dir):
    gold = load_json(gold_path)
    pred = load_json(pred_path)

    assert len(gold) == len(pred), f"{name}: gold/pred length mismatch"

    ent_fp = Counter()
    ent_fn = Counter()
    rel_fp = Counter()
    rel_fn = Counter()

    correct_relation_examples = []
    missed_relation_examples = []
    extra_relation_examples = []

    for i, (g, p) in enumerate(zip(gold, pred)):
        tokens = g["tokens"]

        gold_ents = {entity_key(e) for e in g["entities"]}
        pred_ents = {entity_key(e) for e in p["entities"]}

        for e in pred_ents - gold_ents:
            ent_fp[e[2]] += 1
        for e in gold_ents - pred_ents:
            ent_fn[e[2]] += 1

        gold_rels = {relation_key(r, g["entities"]) for r in g["relations"]}
        pred_rels = {relation_key(r, p["entities"]) for r in p["relations"]}

        for r in pred_rels - gold_rels:
            rel_fp[r[0]] += 1
        for r in gold_rels - pred_rels:
            rel_fn[r[0]] += 1

        # collect a few examples
        common = list(gold_rels & pred_rels)
        missed = list(gold_rels - pred_rels)
        extra = list(pred_rels - gold_rels)

        if common and len(correct_relation_examples) < 3:
            r = common[0]
            correct_relation_examples.append({
                "index": i,
                "sentence": " ".join(tokens),
                "relation": r[0],
                "head": span_text(tokens, r[1], r[2]),
                "tail": span_text(tokens, r[4], r[5]),
            })

        if missed and len(missed_relation_examples) < 3:
            r = missed[0]
            missed_relation_examples.append({
                "index": i,
                "sentence": " ".join(tokens),
                "relation": r[0],
                "head": span_text(tokens, r[1], r[2]),
                "tail": span_text(tokens, r[4], r[5]),
            })

        if extra and len(extra_relation_examples) < 3:
            r = extra[0]
            extra_relation_examples.append({
                "index": i,
                "sentence": " ".join(tokens),
                "relation": r[0],
                "head": span_text(tokens, r[1], r[2]),
                "tail": span_text(tokens, r[4], r[5]),
            })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": name,
        "entity_false_positives_by_type": dict(ent_fp),
        "entity_false_negatives_by_type": dict(ent_fn),
        "relation_false_positives_by_type": dict(rel_fp),
        "relation_false_negatives_by_type": dict(rel_fn),
        "correct_relation_examples": correct_relation_examples,
        "missed_relation_examples": missed_relation_examples,
        "extra_relation_examples": extra_relation_examples,
    }

    out_path = out_dir / f"{name.lower()}_error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved: {out_path}")
    print(f"\n{name} entity FN by type:")
    for k, v in ent_fn.most_common():
        print(f"  {k}: {v}")

    print(f"\n{name} entity FP by type:")
    for k, v in ent_fp.most_common():
        print(f"  {k}: {v}")

    print(f"\n{name} relation FN by type:")
    for k, v in rel_fn.most_common():
        print(f"  {k}: {v}")

    print(f"\n{name} relation FP by type:")
    for k, v in rel_fp.most_common():
        print(f"  {k}: {v}")


def main():
    analyze_dataset(
        "ADKG",
        "data/spert/adkg/test.json",
        "external/spert/data/log/adkg_eval/2026-04-19_12:16:01.586657/predictions_test_epoch_0.json",
        "results/error_analysis"
    )

    analyze_dataset(
        "MDKG",
        "data/spert/mdkg/test.json",
        "external/spert/data/log/mdkg_eval/2026-04-19_14:19:19.447506/predictions_test_epoch_0.json",
        "results/error_analysis"
    )


if __name__ == "__main__":
    main()
