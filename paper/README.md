Frozen numbers for the shipped harnesses. See `../REPRODUCE.md`.

These JSON files are **scores we measured**, not a redistribution of benchmark
videos or full annotations. LVBench-derived split files live in `../manifests/`
and are CC-BY-NC-SA-4.0. Dataset terms: `../DATASETS.md`.

Packed-context sidecars are in the supplementary archive, not this git tree
(70 MB for WeakFT). See that pack's `dumps/CONTEXTS.md`.

| File | Contents |
|---|---|
| `scores.json` | Dev 350 / held-out 882 / AKS-90 / Video-MME / MLVU |
| `mcnemar.json` | AKS↔CardinalityLedger, AKS-90, Uniform↔WeakFT |
| `mcnemar_legacy_hybrid_timestamped.json` | Previous main-table pairs |
| `search_aks_seeded.json` | AKS-seeded trajectory |
| `search_k40.json` | Historical 15-candidate k40-evolution |
