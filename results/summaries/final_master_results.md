# Final Master Results

## Baseline SpERT
- ADKG dev: NER micro 69.07, NER macro 57.74, RE micro w/o NEC 45.13, RE micro with NEC 40.78
- ADKG test: NER micro 68.22, NER macro 61.89, RE micro w/o NEC 47.31, RE micro with NEC 42.50
- MDKG dev: NER micro 80.99, NER macro 71.80, RE micro w/o NEC 55.25, RE micro with NEC 47.59
- MDKG test: NER micro 81.08, NER macro 71.41, RE micro w/o NEC 56.70, RE micro with NEC 50.48

## Extension: ADKG-compatible auxiliary pretraining
- ADKG dev: NER micro 68.68, NER macro 57.46, RE micro w/o NEC 45.71, RE micro with NEC 40.66
- ADKG test: NER micro 67.85, NER macro 60.61, RE micro w/o NEC 47.55, RE micro with NEC 42.23

## Main takeaways
- MDKG outperformed ADKG under the baseline.
- The auxiliary-pretraining extension produced mixed results on ADKG.
- On ADKG test, auxiliary pretraining slightly improved RE micro F1 without NEC (47.31 -> 47.55), but slightly reduced NER micro F1 (68.22 -> 67.85) and RE micro F1 with NEC (42.50 -> 42.23).
