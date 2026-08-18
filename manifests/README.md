# Manifests (not Apache-2.0)

These files are **derived from [THUDM/LVBench](https://huggingface.co/datasets/THUDM/LVBench)** and are licensed under **CC-BY-NC-SA-4.0**, the same terms as LVBench. They are **not** covered by this repository’s Apache-2.0 license.

| File | What it is |
|---|---|
| `lvbench_videos.json` | 83 accessible / 20 inaccessible YouTube ids from a ModelScope snapshot (2026-08-08) |
| `lvbench_split_seed42.json` | 1,232-QA index after `random.Random(42).shuffle`: val 350 + held-out 882, including answer letters |

Academic / non-commercial use only. ShareAlike applies to adaptations. Videos are not in this repository; fetch them yourself (see `../DATASETS.md`).

LVBench does not own the copyright of the raw videos. YouTube ids are listed so a reviewer can rebuild the same local subset; they are not a grant to download or redistribute those videos.
