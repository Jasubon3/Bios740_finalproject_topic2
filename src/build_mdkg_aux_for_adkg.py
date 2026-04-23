from pathlib import Path
import json
import shutil
from collections import Counter


PROCESSED_MDKG = Path("data/processed/MDKG_processed.json")
ADKG_TYPES = Path("data/spert/adkg/types.json")
OUTDIR = Path("data/spert/mdkg_aux_adkg")
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    mdkg = load_json(PROCESSED_MDKG)
    adkg_types = load_json(ADKG_TYPES)

    allowed_entity_types = set(adkg_types["entities"].keys())
    allowed_relation_types = set(adkg_types["relations"].keys())

    print("Allowed entity types:", sorted(allowed_entity_types))
    print("Allowed relation types:", sorted(allowed_relation_types))

    summary = {}

    exported = {}

    for split in ["train", "dev", "test"]:
        exported[split] = []

        total_examples = 0
        kept_examples = 0
        total_entities_before = 0
        total_entities_after = 0
        total_relations_before = 0
        total_relations_after = 0

        entity_type_counter = Counter()
        relation_type_counter = Counter()

        for ex in mdkg[split]:
            total_examples += 1
            total_entities_before += len(ex["entities"])
            total_relations_before += len(ex["relations"])

            # keep only ADKG-compatible entity types
            kept_entities = []
            old_to_new = {}

            for old_idx, ent in enumerate(ex["entities"]):
                if ent["type"] in allowed_entity_types:
                    old_to_new[old_idx] = len(kept_entities)
                    kept_entities.append({
                        "type": ent["type"],
                        "start": ent["start"],
                        "end": ent["end"]
                    })

            # keep only ADKG-compatible relations whose head/tail survived
            kept_relations = []
            for rel in ex["relations"]:
                if rel["type"] not in allowed_relation_types:
                    continue
                if rel["head"] not in old_to_new or rel["tail"] not in old_to_new:
                    continue

                kept_relations.append({
                    "type": rel["type"],
                    "head": old_to_new[rel["head"]],
                    "tail": old_to_new[rel["tail"]],
                })

            # keep only examples that still contain at least one relation
            if len(kept_relations) == 0:
                continue

            kept_examples += 1
            total_entities_after += len(kept_entities)
            total_relations_after += len(kept_relations)

            for ent in kept_entities:
                entity_type_counter[ent["type"]] += 1
            for rel in kept_relations:
                relation_type_counter[rel["type"]] += 1

            exported[split].append({
                "tokens": ex["tokens"],
                "entities": kept_entities,
                "relations": kept_relations
            })

        summary[split] = {
            "total_examples_before": total_examples,
            "kept_examples_after": kept_examples,
            "pct_examples_kept": round(100 * kept_examples / total_examples, 2) if total_examples else 0,
            "total_entities_before": total_entities_before,
            "total_entities_after": total_entities_after,
            "total_relations_before": total_relations_before,
            "total_relations_after": total_relations_after,
            "entity_type_counts_after": dict(entity_type_counter),
            "relation_type_counts_after": dict(relation_type_counter),
        }

        print(f"\n=== {split.upper()} ===")
        print(f"Examples: {total_examples} -> {kept_examples}")
        print(f"Entities: {total_entities_before} -> {total_entities_after}")
        print(f"Relations: {total_relations_before} -> {total_relations_after}")
        print("Entity types kept:", dict(entity_type_counter))
        print("Relation types kept:", dict(relation_type_counter))

    # save splits
    for split in ["train", "dev", "test"]:
        save_json(exported[split], OUTDIR / f"{split}.json")

    # copy ADKG types so label space matches ADKG exactly
    shutil.copy2(ADKG_TYPES, OUTDIR / "types.json")

    save_json(summary, OUTDIR / "summary.json")

    print(f"\nSaved auxiliary dataset to: {OUTDIR}")
    print("Files:")
    print("- train.json")
    print("- dev.json")
    print("- test.json")
    print("- types.json")
    print("- summary.json")


if __name__ == "__main__":
    main()
