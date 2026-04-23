from pathlib import Path
import json

from load_data import load_json


PROCESSED_DIR = Path("data/processed")
OUTDIR = Path("data/spert")
OUTDIR.mkdir(parents=True, exist_ok=True)


# Conservative symmetry choices:
# - associated_with: symmetric
# - everything else: directional
SYMMETRIC_RELATIONS = {
    "associated_with": True
}


def build_types_json(dataset):
    entity_types = set()
    relation_types = set()

    for split in ["train", "dev", "test"]:
        for ex in dataset[split]:
            for ent in ex.get("entities", []):
                entity_types.add(ent["type"])
            for rel in ex.get("relations", []):
                relation_types.add(rel["type"])

    entity_types = sorted(entity_types)
    relation_types = sorted(relation_types)

    types = {
        "entities": {},
        "relations": {}
    }

    for ent_type in entity_types:
        types["entities"][ent_type] = {
            "short": ent_type,
            "verbose": ent_type
        }

    for rel_type in relation_types:
        types["relations"][rel_type] = {
            "short": rel_type,
            "verbose": rel_type,
            "symmetric": SYMMETRIC_RELATIONS.get(rel_type, False)
        }

    return types


def convert_split(split_examples):
    converted = []

    for ex in split_examples:
        doc = {
            "tokens": ex["tokens"],
            "entities": [],
            "relations": []
        }

        for ent in ex.get("entities", []):
            doc["entities"].append({
                "type": ent["type"],
                "start": ent["start"],
                "end": ent["end"]
            })

        for rel in ex.get("relations", []):
            doc["relations"].append({
                "type": rel["type"],
                "head": rel["head"],
                "tail": rel["tail"]
            })

        converted.append(doc)

    return converted


def export_dataset(dataset_name):
    inpath = PROCESSED_DIR / f"{dataset_name}_processed.json"
    with open(inpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset_outdir = OUTDIR / dataset_name.lower()
    dataset_outdir.mkdir(parents=True, exist_ok=True)

    types = build_types_json(data)
    with open(dataset_outdir / "types.json", "w", encoding="utf-8") as f:
        json.dump(types, f, indent=2, ensure_ascii=False)

    for split in ["train", "dev", "test"]:
        converted = convert_split(data[split])
        with open(dataset_outdir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"\nExported {dataset_name} to {dataset_outdir}")
    print("Files:")
    print("- types.json")
    print("- train.json")
    print("- dev.json")
    print("- test.json")


def main():
    export_dataset("ADKG")
    export_dataset("MDKG")


if __name__ == "__main__":
    main()
