from pprint import pprint
from load_data import load_dataset


def inspect_example(dataset_name, split="train", idx=0):
    data = load_dataset(dataset_name)
    example = data[split][idx]

    print(f"\nDataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Index: {idx}")
    print("-" * 60)
    print("Keys:", list(example.keys()))
    print("\nExample preview:")
    pprint(example)


def main():
    inspect_example("ADKG", "train", 0)
    inspect_example("MDKG", "train", 0)


if __name__ == "__main__":
    main()
