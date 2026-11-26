# Live progress
Last updated: 2026-11-28 | Documentation set r75

**PAUSED by user (2026-11-28).** Compression validation complete; all ten compressed candidates within proposed .10pp loss tolerance. P+ST FP16 recommended, not adopted. Original AMP EGTEA test results locked; compressed held-out accuracy unverified. No experiments, test scoring or deployment decisions until explicit resumption. Approved maintenance cleanup verified complete: 105 result files, 13 experiment Python files, 23/23 regression tests before and after. No scientific changes.

| Milestone | Status | Remaining |
|---|---|---|
| M1 Data and splits | EGTEA complete | External dataset/access, annotation and loader protocol |
| M2 Predictors | Complete | None; bounded replacement search closed |
| M3 Adaptation | Complete | None; no further head training approved |
| M4 Caches | Complete for selected models | Preserve active assets; historical training caches partly deleted |
| M5 Fusion/validation | Complete | None; architecture/threshold lock frozen |
| M6 Ablations | Complete | Publication presentation of results |
| M7 Efficiency/compression | Validation complete | Precision adoption; compressed held-out scoring only if separately approved; streaming unmeasured |
| M8 Held-out evaluation | EGTEA complete | External evaluation deferred; GTEA access blocked, Ego4D alternate pending scope/license/protocol |
| M9 Reproducibility/demo | Partial | README/docs sync and local Git deletion commit pending; external recovery audit; full environment reproduction not established; demo/Q&A |
| M10 Paper | Pending | Figures, tables, report and rubric check |

## Resumption queue (user's order)
1. Review deployment precision: FP16 vs INT8 storage, proposed tolerance, whether separate compressed test reporting is justified. Retain original AMP lock.
2. Qualitative overlays and failure-case analysis; use validation for exploratory diagnostics, log any additional test access; no test tuning.
3. Demo: video overlay, branch maps, optional ground truth, measured latency/VRAM; distinguish offline benchmark from streaming.
4. Publication-quality figures/tables: adaptation, fusion/15 subsets, negative results, locked scores/uncertainty, compression and measured costs.
5. External dataset: retry authorized GTEA access or approve Ego4D alternative; audit overlap and lock eligibility/alignment before scoring; no target adaptation.
6. Final report, reproducibility and rubric audit.

Protocols, metrics, provenance and limitations: PROJECT.md. Registry: EXPERIMENTS.csv r43, 199 unique runs. No new registry records from maintenance. Historical result paths may refer to verified members of `results/historical_evidence.tar.gz`; source recovery tag `acv-pre-cleanup-20261128`.
