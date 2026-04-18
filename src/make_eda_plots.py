from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

from load_data import load_dataset


OUTDIR = Path("results/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


def get_type_counts(examples, key):
    counter = Counter()
    for ex in examples:
        for item in ex.get(key, []):
            counter[item["type"]] += 1
    return counter


def plot_counter(counter, title, xlabel, outpath):
    labels = [k for k, _ in counter.most_common()]
    values = [v for _, v in counter.most_common()]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for dataset_name in ["ADKG", "MDKG"]:
        data = load_dataset(dataset_name)
        train_data = data["train"]

        entity_counts = get_type_counts(train_data, "entities")
        relation_counts = get_type_counts(train_data, "relations")

        entity_out = OUTDIR / f"{dataset_name.lower()}_entity_type_distribution.png"
        relation_out = OUTDIR / f"{dataset_name.lower()}_relation_type_distribution.png"

        plot_counter(
            entity_counts,
            title=f"{dataset_name} Train Split: Entity Type Distribution",
            xlabel="Entity Type",
            outpath=entity_out,
        )

        plot_counter(
            relation_counts,
            title=f"{dataset_name} Train Split: Relation Type Distribution",
            xlabel="Relation Type",
            outpath=relation_out,
        )

        print(f"Saved: {entity_out}")
        print(f"Saved: {relation_out}")


if __name__ == "__main__":
    main()
