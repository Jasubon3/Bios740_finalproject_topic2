from pathlib import Path
import json
import csv
from collections import Counter, defaultdict


ADKG_PRED = Path("external/spert/data/log/adkg_eval/2026-04-19_12:16:01.586657/predictions_test_epoch_0.json")
MDKG_PRED = Path("external/spert/data/log/mdkg_eval/2026-04-19_14:19:19.447506/predictions_test_epoch_0.json")
OUTROOT = Path("results/kg")
OUTROOT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return " ".join(text.lower().strip().split())


def span_text(tokens, start, end):
    return " ".join(tokens[start:end])


def build_graph(name, pred_path):
    data = load_json(pred_path)
    outdir = OUTROOT / name.lower()
    outdir.mkdir(parents=True, exist_ok=True)

    node_counter = Counter()
    node_type_counter = defaultdict(Counter)
    edge_counter = Counter()
    triplets = []

    for ex_idx, ex in enumerate(data):
        tokens = ex["tokens"]
        sentence = " ".join(tokens)
        entities = ex.get("entities", [])
        relations = ex.get("relations", [])

        # build normalized node keys
        entity_nodes = []
        for ent in entities:
            text = span_text(tokens, ent["start"], ent["end"])
            norm = normalize_text(text)
            node_key = (norm, ent["type"])
            entity_nodes.append({
                "node_key": node_key,
                "text": text,
                "type": ent["type"]
            })
            node_counter[node_key] += 1
            node_type_counter[norm][ent["type"]] += 1

        for rel in relations:
            h = entity_nodes[rel["head"]]
            t = entity_nodes[rel["tail"]]

            source_key = h["node_key"]
            target_key = t["node_key"]
            relation = rel["type"]

            edge_counter[(source_key, relation, target_key)] += 1

            triplets.append({
                "example_id": ex_idx,
                "sentence": sentence,
                "source_text": h["text"],
                "source_norm": source_key[0],
                "source_type": source_key[1],
                "relation": relation,
                "target_text": t["text"],
                "target_norm": target_key[0],
                "target_type": target_key[1],
            })

    # assign node ids
    node_id_map = {}
    nodes = []
    for i, (node_key, count) in enumerate(node_counter.most_common(), start=1):
        norm_text, node_type = node_key
        node_id = f"N{i}"
        node_id_map[node_key] = node_id
        nodes.append({
            "node_id": node_id,
            "node_text_normalized": norm_text,
            "node_type": node_type,
            "mention_count": count
        })

    edges = []
    for (source_key, relation, target_key), count in edge_counter.most_common():
        edges.append({
            "source_id": node_id_map[source_key],
            "source_text_normalized": source_key[0],
            "source_type": source_key[1],
            "relation": relation,
            "target_id": node_id_map[target_key],
            "target_text_normalized": target_key[0],
            "target_type": target_key[1],
            "edge_count": count
        })

    # save nodes.csv
    with open(outdir / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["node_id", "node_text_normalized", "node_type", "mention_count"]
        )
        writer.writeheader()
        writer.writerows(nodes)

    # save edges.csv
    with open(outdir / "edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_id", "source_text_normalized", "source_type",
                "relation",
                "target_id", "target_text_normalized", "target_type",
                "edge_count"
            ]
        )
        writer.writeheader()
        writer.writerows(edges)

    # save triplets.csv
    with open(outdir / "triplets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "example_id", "sentence",
                "source_text", "source_norm", "source_type",
                "relation",
                "target_text", "target_norm", "target_type"
            ]
        )
        writer.writeheader()
        writer.writerows(triplets)

    summary = {
        "dataset": name,
        "prediction_file": str(pred_path),
        "num_examples": len(data),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_triplets": len(triplets),
        "top_relations": dict(Counter([e["relation"] for e in edges]).most_common(10))
    }

    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{name}")
    print(f"  Saved to: {outdir}")
    print(f"  Examples: {len(data)}")
    print(f"  Nodes:    {len(nodes)}")
    print(f"  Edges:    {len(edges)}")
    print(f"  Triplets: {len(triplets)}")


def main():
    build_graph("ADKG", ADKG_PRED)
    build_graph("MDKG", MDKG_PRED)


if __name__ == "__main__":
    main()
