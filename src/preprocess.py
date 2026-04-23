from pathlib import Path
import json
import re

from load_data import load_dataset


OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)


TOKEN_PATTERN = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


def tokenize_with_offsets(text):
    tokens = []
    offsets = []
    for m in TOKEN_PATTERN.finditer(text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def char_span_to_token_span(start_char, end_char, offsets):
    overlapping = [
        i for i, (s, e) in enumerate(offsets)
        if not (e <= start_char or s >= end_char)
    ]
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1] + 1


def convert_example(example):
    text = example["text"]
    tokens, offsets = tokenize_with_offsets(text)

    entities = []
    entity_id_to_index = {}

    for ent in example.get("entities", []):
        token_span = char_span_to_token_span(ent["start"], ent["end"], offsets)
        if token_span is None:
            continue

        start_tok, end_tok = token_span
        entity_id_to_index[ent["id"]] = len(entities)

        entities.append({
            "start": start_tok,
            "end": end_tok,
            "type": ent["type"],
            "text": ent["text"],
            "char_start": ent["start"],
            "char_end": ent["end"],
            "id": ent["id"],
        })

    relations = []
    for rel in example.get("relations", []):
        head_id = rel["head"]["id"]
        tail_id = rel["tail"]["id"]

        if head_id not in entity_id_to_index or tail_id not in entity_id_to_index:
            continue

        relations.append({
            "head": entity_id_to_index[head_id],
            "tail": entity_id_to_index[tail_id],
            "type": rel["type"],
        })

    return {
        "doc_id": example["doc_id"],
        "sent_id": example["sent_id"],
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "relations": relations,
    }


def process_dataset(dataset_name):
    data = load_dataset(dataset_name)
    processed = {}

    for split in ["train", "dev", "test"]:
        processed[split] = [convert_example(ex) for ex in data[split]]

    outpath = OUTDIR / f"{dataset_name}_processed.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"Saved: {outpath}")
    for split in ["train", "dev", "test"]:
        print(f"  {split}: {len(processed[split])} examples")


def main():
    process_dataset("ADKG")
    process_dataset("MDKG")


if __name__ == "__main__":
    main()
