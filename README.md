# Biomedical Knowledge Graph Triplet Extraction

This project studies biomedical knowledge graph triplet extraction from text using **SpERT**, a joint entity and relation extraction model. The goal is to extract structured **(subject, relation, object)** triplets from annotated biomedical sentences and convert them into graph artifacts such as node, edge, and triplet tables.

We benchmark the pipeline on two datasets:

- **ADKG**: focused on Alzheimer's disease and related biomedical concepts
- **MDKG**: focused on mental-health and clinical concepts

We also evaluate a small **transfer learning / domain adaptation extension** in which an ADKG-compatible subset of MDKG is used for auxiliary pretraining before fine-tuning on ADKG.

## Repository structure

```text
src/                  # preprocessing, EDA, error analysis, graph construction
configs/              # training and evaluation config files
results/              # tables, summaries, and graph artifacts
data/                 # raw, processed, and SpERT-formatted datasets
external/spert/       # SpERT codebase
paper/ or main.tex    # final report