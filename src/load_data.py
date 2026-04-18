from pathlib import Path
import json


RAW_DIR = Path("data/raw")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(dataset_name):
    """
    dataset_name: 'ADKG' or 'MDKG'
    returns: dict with keys train, dev, test
    """
    path = RAW_DIR / f"{dataset_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = load_json(path)

    required_splits = ["train", "dev", "test"]
    for split in required_splits:
        if split not in data:
            raise ValueError(f"Missing split '{split}' in {path.name}")

    return data


def get_split(dataset_name, split):
    data = load_dataset(dataset_name)
    if split not in data:
        raise ValueError(f"Split '{split}' not found in {dataset_name}")
    return data[split]


def list_available_datasets():
    return sorted([p.stem for p in RAW_DIR.glob("*.json")])


def main():
    datasets = list_available_datasets()
    print("Available datasets:", datasets)

    for name in datasets:
        data = load_dataset(name)
        print(f"\n{name}")
        print(f"  train: {len(data['train'])}")
        print(f"  dev:   {len(data['dev'])}")
        print(f"  test:  {len(data['test'])}")


if __name__ == "__main__":
    main()
