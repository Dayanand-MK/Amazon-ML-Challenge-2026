# Final model card

Model: lightgbm_31. Framework: LightGBM 4.6.0 (MIT). Artifact: models/final_model.pkl (MIT; see LICENSE).
Locally trained from supplied training data only; no pretrained weights. 240 trees, 14,640 nodes, conservative tree parameter bound 468,480 (<8 billion).
Threshold 0.750; source-balanced retrieval True; max 24 scored targets/S1.

| Metric | Held-out evaluation |
|---|---:|
| Macro per-S1 F0.5 (challenge metric) | 0.891918 |
| Micro F0.5 (diagnostic only) | 0.930619 |
| Pooled pair precision | 0.980943 |
| Pooled pair recall | 0.772163 |
| Pair candidate recall | 0.862289 |
| Zero-match S1 accuracy | 0.962121 |

See Documentation_template.md for split, selection, features, provenance and limitations. Country shift to France is not quantified by this US/India holdout. Only load trusted pickle files. Model serialization includes the ordered features, threshold and measured metadata.
