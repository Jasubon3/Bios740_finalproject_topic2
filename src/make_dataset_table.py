from pathlib import Path
from collections import Counter
import pandas as pd

from load_data import load_dataset


OUTDIR = Path("results/tables")
OUTDIR.mkdir(parents=True, exist_ok=True)


def summarize_split(examples):
    entity_type_counter = Counter()
    relation_type_counter = Counter()

    total_entities = 0
    total_relations = 0
    text_lengths = []

    for ex in examples:
        entities = ex.get("entities", [])
        relations = ex.get("relations", [])
        text = ex.get("text", "")

        total_entities += len(entities)
        total_relations += len(relations)
        text_lengths.append(len(text.split()))

        for ent in entities:
            entity_type_counter[ent["type"]] += 1

        for rel in relations:
            relation_type_counter[rel["type"]] += 1

    return {
        "num_examples": len(examples),
        "total_entities": total_entities,
        "total_relations": total_relations,
        "avg_entities_per_example": round(total_entities / len(examples), 2),
        "avg_relations_per_example": round(total_relations / len(examples), 2),
        "avg_text_length_words": round(sum(text_lengths) / len(text_lengths), 2),
        "num_entity_types": len(entity_type_counter),
        "num_relation_types": len(relation_type_counter),
    }


def main():
    rows = []

    for dataset_name in ["ADKG", "MDKG"]:
        data = load_dataset(dataset_name)

        for split in ["train", "dev", "test"]:
            stats = summarize_split(data[split])
            stats["dataset"] = dataset_name
            stats["split"] = split
            rows.append(stats)

    df = pd.DataFrame(rows)
    df = df[
        [
            "dataset",
            "split",
            "num_examples",
            "total_entities",
            "total_relations",
            "avg_entities_per_example",
            "avg_relations_per_example",
            "avg_text_length_words",
            "num_entity_types",
            "num_relation_types",
        ]
    ]

    print(df.to_string(index=False))

    out_csv = OUTDIR / "dataset_summary_by_split.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table to: {out_csv}")


if __name__ == "__main__":
    main()
