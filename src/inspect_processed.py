from pathlib import Path
import json
from pprint import pprint


def inspect_file(path_str, split="train", idx=0):
    path = Path(path_str)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ex = data[split][idx]
    print(f"\nFile: {path.name}")
    print(f"Split: {split}, Index: {idx}")
    print("-" * 60)
    pprint(ex)


def main():
    inspect_file("data/processed/ADKG_processed.json")
    inspect_file("data/processed/MDKG_processed.json")


if __name__ == "__main__":
    main()
