from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median
import json


OUTDIR = Path("results/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_processed_dataset(dataset_name):
    path = Path("data/processed") / f"{dataset_name}_processed.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def entity_overlap(e1, e2):
    return not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"])


def entity_nested(e1, e2):
    return (
        (e1["start"] <= e2["start"] and e1["end"] >= e2["end"]) or
        (e2["start"] <= e1["start"] and e2["end"] >= e1["end"])
    ) and (e1["start"], e1["end"]) != (e2["start"], e2["end"])


def relation_distance(rel, entities):
    h = entities[rel["head"]]
    t = entities[rel["tail"]]
    if h["end"] <= t["start"]:
        return t["start"] - h["end"]
    elif t["end"] <= h["start"]:
        return h["start"] - t["end"]
    else:
        return 0


def summarize_dataset(dataset_name):
    data = load_processed_dataset(dataset_name)

    overlap_sent_count = 0
    nested_sent_count = 0
    total_sent_count = 0

    relation_distances = []
    relation_type_distances = defaultdict(list)

    sentence_lengths = []
    entity_counts = []
    relation_counts = []
    relation_bucket_counts = Counter()

    pair_type_counter = Counter()

    for split in ["train", "dev", "test"]:
        for ex in data[split]:
            total_sent_count += 1
            tokens = ex["tokens"]
            entities = ex["entities"]
            relations = ex["relations"]

            sentence_lengths.append(len(tokens))
            entity_counts.append(len(entities))
            relation_counts.append(len(relations))

            if len(relations) == 0:
                relation_bucket_counts["0"] += 1
            elif len(relations) == 1:
                relation_bucket_counts["1"] += 1
            else:
                relation_bucket_counts["2+"] += 1

            has_overlap = False
            has_nested = False
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if entity_overlap(entities[i], entities[j]):
                        has_overlap = True
                    if entity_nested(entities[i], entities[j]):
                        has_nested = True

            if has_overlap:
                overlap_sent_count += 1
            if has_nested:
                nested_sent_count += 1

            for rel in relations:
                d = relation_distance(rel, entities)
                relation_distances.append(d)
                relation_type_distances[rel["type"]].append(d)

                h = entities[rel["head"]]["type"]
                t = entities[rel["tail"]]["type"]
                pair_type_counter[(h, rel["type"], t)] += 1

    summary = {
        "dataset": dataset_name,
        "num_sentences": total_sent_count,
        "avg_sentence_length": round(mean(sentence_lengths), 2),
        "median_sentence_length": round(median(sentence_lengths), 2),
        "avg_entities_per_sentence": round(mean(entity_counts), 2),
        "avg_relations_per_sentence": round(mean(relation_counts), 2),
        "sentences_with_overlap_entities": overlap_sent_count,
        "sentences_with_nested_entities": nested_sent_count,
        "pct_sentences_with_overlap_entities": round(100 * overlap_sent_count / total_sent_count, 2),
        "pct_sentences_with_nested_entities": round(100 * nested_sent_count / total_sent_count, 2),
        "avg_relation_distance": round(mean(relation_distances), 2) if relation_distances else 0,
        "median_relation_distance": round(median(relation_distances), 2) if relation_distances else 0,
        "relation_count_bucket_0": relation_bucket_counts["0"],
        "relation_count_bucket_1": relation_bucket_counts["1"],
        "relation_count_bucket_2plus": relation_bucket_counts["2+"],
    }

    print(f"\n===== {dataset_name} EXTENDED EDA =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nTop 10 head-relation-tail patterns:")
    for (h, r, t), c in pair_type_counter.most_common(10):
        print(f"  ({h}, {r}, {t}): {c}")

    print("\nAverage relation distance by relation type:")
    for rel_type, vals in sorted(relation_type_distances.items()):
        print(f"  {rel_type}: {round(mean(vals), 2)}")

    outpath = OUTDIR / f"{dataset_name.lower()}_extended_eda.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "top_head_relation_tail_patterns": [
                    {"head_type": h, "relation": r, "tail_type": t, "count": c}
                    for (h, r, t), c in pair_type_counter.most_common(20)
                ],
                "avg_relation_distance_by_type": {
                    k: round(mean(v), 2) for k, v in relation_type_distances.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {outpath}")


def main():
    summarize_dataset("ADKG")
    summarize_dataset("MDKG")


if __name__ == "__main__":
    main()
