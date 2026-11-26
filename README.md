# Resource-Constrained Egocentric Gaze Estimation

Research on complementary gaze predictors within an 8 GiB inference VRAM budget. The selected efficient ensemble is P+ST (training-only empirical prior, ViNet-S, GLC); P+STD adds DINOv2. The original weighted CSAT ensemble and T/P+T are comparators. See `agents/PROJECT.md` for locked protocols, decisions, provenance and limitations, `agents/MILESTONES.md` for status, and `agents/EXPERIMENTS.csv` for 199 recorded experiments.

## Results

EGTEA official test, locked AMP models, 2,021 clips / 993 fixation targets, fixed validation thresholds:

| Model | F1 (%) |
|---|---:|
| T | 45.570 |
| Weighted CSAT | 45.766 |
| P+T | 45.614 |
| P+ST | 45.969 |
| P+STD | 46.048 |

P+ST FP16 storage is recommended but not adopted: 51.500% validation F1, 152.354 MiB serialized weights, 511.108 MiB peak allocated GPU memory, 72.831 ms model latency. INT8 storage (77.346 MiB) dequantizes to FP16, not native CUDA INT8. Both preserve ViNet-S BatchNorm floating state in FP32. Compressed held-out accuracy, optimized streaming and external transfer remain unverified. Subject overlap and possible checkpoint exposure limit generalization claims; this is not a reproduction of the published GLC evaluation protocol.

## Setup and layout

Linux, Conda `gaze` (Python 3.10, PyTorch 2.5.1, CUDA 12.1, Torchvision 0.20.1), Quadro RTX 4000 8 GB. Preserve the working environment; snapshots in `environment/`. Obtain data and checkpoints from authorized sources; do not commit them. Patched upstream source must be reconstructed and verified using `patches/` and the supplemental GLC patch in the locked evaluation results. Full environment reconstruction has not been verified.

| Path | Purpose |
|---|---|
| `agents/` | Research rules, protocol, status and registry |
| `datasets/`, `models/`, `fusion/`, `utils/` | Data, model adapters, fusion and metrics |
| `experiments/` | Active entry points and shared helpers (13 Python files) |
| `scripts/`, `tests/` | Preparation, reproducibility and synthetic regression |
| `environment/`, `patches/` | Environment and upstream provenance |
| `data/`, `checkpoints/`, `external/`, `results/` | VM-only assets, ignored by Git |

Required assets include `data/egtea/{cropped_clips,gaze_data,parsed_gaze,splits}`, `external/GLC~`, `external/vinet_v2`, `external/dinov2`, original selected weights in `checkpoints/`, DINOv2 pretrained weights in the Torch Hub cache, and selected adapted Huang state `results/adaptation/huang_clean/best.pt`. Do not regenerate locked splits or reset modified upstream trees. See the provenance in PROJECT.md for exact revisions and hashes.

## Safe installation checks

Run from the repository root:

```bash
conda activate gaze
python -B -m unittest discover -s tests -v
```

The 23 synthetic CPU regression tests passed before and after maintenance on the VM. They are not real CUDA inference or end-to-end reproduction tests. No held-out scoring is authorized while the project is paused.

## Active experiment entry points

- `experiments/finalize_validation.py`: aligned validation, final fusion, standardized profiling.
- `experiments/evaluate_locked_v3.py`: locked EGTEA evaluator; depends on prior lock and patch evidence.
- `experiments/screen_compression.py`, `experiments/complete_compression.py`: validation-only compression and BatchNorm-safe follow-up.
- `experiments/screen_dinov2.py`, `experiments/screen_glc_ego4d.py`, `experiments/train_fusion.py`, `experiments/audit_geometric_fusion.py`, `experiments/audit_metrics.py`: retained shared helpers, diagnostics and regression dependencies.
- `experiments/train.py`, `experiments/cache_predictions.py`, `experiments/evaluate.py`: original adaptation/caching/prior evaluation. Historical defaults may refer to archived assets; inspect arguments and provenance before use.

Invoke modules from the root with `python -m experiments.<module>`. Keep source files unchanged where SHA-256 gates bind them to the original results. Use fresh outputs for approved new work; do not overwrite evidence or tune against test results.

## Evidence and recovery

The approved 2026-11-28 maintenance retained 104 original result files live, plus one verified historical evidence bundle. Keep the selected caches, `final_selection/`, all three `locked_evaluation*` directories, both compression directories, `dinov2_s/head_clean/`, selected Huang state, and empirical prior/validation provenance at their existing paths.

Historical result paths in the registry are original paths. If a file is absent live, find its exact member in `results/historical_evidence.tar.gz`. The bundle contains 404 original files and an embedded `_acv_cleanup_manifest.json` recording paths, sizes and SHA-256 digests. The reported bundle SHA-256 is `2a8a053f38cc2ac2468c60520bf38466b2300bcfb0cb757b20ea47fabf353d43`.

Example: inspect or recover a historical metric without restoring all loose files:

```bash
tar -tzf results/historical_evidence.tar.gz
tar -xOzf results/historical_evidence.tar.gz results/ablation_uniform/validation.json
```

For multiple related files, extract only the needed members into a separate temporary directory after verifying the bundle digest; do not extract over the active tree. Preserve the bundle separately from disposable run outputs. Older deletions predating this bundle mean some historical experiments cannot be directly rerun without regenerating assets.

Eleven retired source scripts are recoverable from Git commit `41efda480a46471fddd1559a839dbec5dcd7acce`, retained on the VM by tag `acv-pre-cleanup-20261128`. Example:

```bash
git show acv-pre-cleanup-20261128:experiments/audit_f1_oracle.py
```

The evidence bundle and recovery tag are VM-local; preserve them and their recoverability when synchronizing. External hand-object checkout and unused Ego4D checkpoint remain pending a separate recovery audit. Do not delete active model repos, datasets, weights, or research evidence without approval.
