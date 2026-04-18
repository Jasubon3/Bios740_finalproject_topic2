from collections import Counter
from statistics import mean
from load_data import load_dataset


def summarize_split(examples, split_name):
    entity_type_counter = Counter()
    relation_type_counter = Counter()

    n_entities = []
    n_relations = []
    text_lengths = []

    for ex in examples:
        entities = ex.get("entities", [])
        relations = ex.get("relations", [])
        text = ex.get("text", "")

        n_entities.append(len(entities))
        n_relations.append(len(relations))
        text_lengths.append(len(text.split()))

        for ent in entities:
            entity_type_counter[ent["type"]] += 1

        for rel in relations:
            relation_type_counter[rel["type"]] += 1

    print(f"\nSplit: {split_name}")
    print("-" * 40)
    print(f"Number of examples: {len(examples)}")
    print(f"Total entities: {sum(n_entities)}")
    print(f"Total relations: {sum(n_relations)}")
    print(f"Average entities/example: {mean(n_entities):.2f}")
    print(f"Average relations/example: {mean(n_relations):.2f}")
    print(f"Average text length (words): {mean(text_lengths):.2f}")

    print("\nTop 10 entity types:")
    for k, v in entity_type_counter.most_common(10):
        print(f"  {k}: {v}")

    print("\nTop 10 relation types:")
    for k, v in relation_type_counter.most_common(10):
        print(f"  {k}: {v}")


def summarize_dataset(dataset_name):
    data = load_dataset(dataset_name)

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    for split in ["train", "dev", "test"]:
        summarize_split(data[split], split)


def main():
    summarize_dataset("ADKG")
    summarize_dataset("MDKG")


if __name__ == "__main__":
    main()
