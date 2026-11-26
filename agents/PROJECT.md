# Resource-Constrained Egocentric Gaze Estimation
Last updated: 2026-11-28 | Documentation set r75

## Objective and research decisions
Test whether complementary gaze cues improve the strongest same-protocol singleton within 8 GiB inference VRAM (target 7 GiB). Study adaptation/augmentation, fusion, ablation, transfer and compression; deliver reproducible code, paper and demo. No test-driven model selection.
- 2026-11-27: approved ablation gate: full ensemble must exceed the strongest same-protocol singleton and published GLC-MViT 44.80% numerically on validation. Weighted CSAT met this gate; not published superiority. Later optimization was bounded and exploratory.
- 2026-11-28: after final validation, before test scoring, user locked P+ST AMP as efficient candidate, P+STD AMP as four-slot research comparator and T, weighted CSAT, P+T as controls. Cue search closed. Original AMP held-out results are immutable; no retraining or threshold changes.
- 2026-11-28: user deferred external evaluation, approved validation compression, then paused after compression completion. FP16 recommendation is not adoption. Compressed held-out scoring requires a separate approved protocol; no new experiments while paused (status in MILESTONES.md).

## Data, models and evaluation protocol
| ID | Predictor | Selected state / role |
|---|---|---|
| C | Gaussian spatial prior | Original training-fit mean/covariance, fixed |
| S | [ViNet-S](https://github.com/ViNet-Saliency/vinet_v2), video saliency | Frozen; adaptation used existing 1,590,689-param decoder |
| A | [Huang 2018](https://github.com/hyf015/egocentric-gaze-prediction), spatial + late fusion | Clean update 200, existing 12,321-param head |
| T | [GLC](https://github.com/BolinLai/GLC), global-local temporal gaze | Frozen EGTEA checkpoint; adaptation used existing 97-param classifier |
| D | DINOv2 ViT-S/14 spatial gaze | Frozen backbone, trained 43,137-param head, clean epoch 8 |
| P | Empirical spatial prior | Train-only mean normalized target heatmaps with uniform pseudo-observation |
A is the released fixation-assuming spatial + late fusion model, not full Attention Transition (no LSTM/flow; full checkpoint unverified). E = Ego4D GLC; H = hand-object contact detector (blocked); G = matched Gaussian; Q/M = motion-conditioned prior / compensated motion; R/O = object-conditioned prior / object saliency. These alternatives are not final models.

**Locked data protocol (11-26):** EGTEA Gaze+ subject-grouped 80/20 subdivision of official train, official test held out. Train/val/test 6,639/1,660/2,021 clips, zero pairwise clip overlap. Train/val 26/6 subjects, disjoint, union equals official train 8,299 clips. Train 4,720 valid/3,202 fixation; val 1,136 valid/675 fixation; test 1,521 valid/993 fixation targets. All 26 test subjects overlap official train subjects. Manifest `data/egtea/splits/manifest.json`. RGB 256² center crop; one common target per clip (random train, deterministic val/test); S 32 consecutive frames, T 8 stride-8 frames, C/A/D last frame. All maps normalized probabilities on 64²; preserve gaze geometry, masks, row IDs.

**Locked metrics (11-27):** `glc_minmax_v1` primary: per-map `(p-min)/(max-min+1e-6)` after probability normalization; fixation-only, macro frame precision/recall then harmonic F1, 11 thresholds 0–.02, target threshold .001. Legacy probability F1 secondary; retain probability maps for fusion/KL. Report fixed validation-threshold test F1 and adaptive F1 descriptively. AUC-J, NSS, KL; nominal GLC AAE is not calibrated angular error. Metric correction followed validation inspection and reversed rankings; earlier probability-F1 choices retained as historical. Only retained epochs could be reselected.

**Locked final fusion (11-28, pre-test):** T anchor, multiply T by exponentiated weighted clipped log-ratios of auxiliary maps to uniform `1/4096` and normalize; eps `1e-8`, log bounds ±4, gamma1. Exact implementation in `experiments/finalize_validation.py`. S=.30, D/P=.05. P+T uses P only; P+ST uses P/S; P+STD uses P/S/D. Weighted CSAT = normalize(.95*T + .05*mean(C,S,A)), gamma 1. C/P precomputed. FP16 CUDA autocast with FP32 stored weights, inputs/maps/fusion. No new fitting. Final primary thresholds: T/P+T .006, weighted CSAT .008, P+ST/P+STD .004; legacy .002 all. External transfer planned with the same locked AMP models/thresholds, without target adaptation, subject to comparable labels/protocol.

## Completed experimental studies (11-26–28)
F1 values below are primary percentages unless stated. CSV is the compact run registry; exact configurations, metrics, histories and arrays are in live `results/` or the verified historical evidence bundle, and source history. Historical r72 preserves detailed decision timing, hyperparameters, verification and hashes (see provenance). Do not treat exploratory validation gains as confirmatory.

### Adaptation, augmentation and fusion
- Approved 11-26: frozen backbones/BN, existing heads, matched clean/aug eligibility; batch1, accum4, AdamW lr1e-4, 200 updates, seed42, KL, FP16 AMP scale128 (after S smoke failure), val100/200. Six runs share 800-fixation manifests. Full clean val reselected C fixed, S/T frozen, A clean200 (best/last tensors identical, user audit). S frozen/clean/aug20.970/20.660/20.313; mixed augmentation effects. Selected C/S/A/T15.032/20.970/14.768/51.164. `results/metric_audit/validation.json`, `results/adaptation/`, `results/cache_selected/`.
- Clip-coherent augmentation: flip p=.5, roll ±8°, x/y shift ±4%, brightness gain .85–1.15/offset ±.06, blur p=.25 (3/5/7px), lower-field rectangle occlusion p=.20 (width10–22%, height10–25%, synthetic). Transform gaze, invalidate out-of-frame, regenerate heatmap. Clean shares RNG/eligibility without pixel changes; independent RNG, workers=0; clean val/test. Cache actual transformed-video predictions; frozen predictors in fusion.
- `fusion/strategies.py`: uniform, static weights, dynamic confidence/agreement MLP, CNN log-map residual, logpool, spatial weights, anchored residual. Initial FP32 KL, AdamW .003, batch32, 15 epochs, seed42; spatial clean/aug43.610/45.150. Uniform CSAT22.895 vs T51.164. Concentration audit T vs uniform: support350/1304px, target mass53.554/25.439%, peak error8.478/8.344px; broadening and target-mass loss coexist with slight peak improvement, not causal. `results/concentration_audit/validation.json`.
- Approved 42 weighted/sharpening configs: best .95T+.05mean(C,S,A), gamma1, 51.356 vs sharpened T51.173. All 15 uniform and 15 fixed-weight subsets: uniform best multi ST35.372; weighted CSAT51.356, ST51.287; removing C/S/A/T costs .079/.085/.055/36.606pp (not independent). Renormalize fixed weights, no retuning. `results/weighted_sharpening/validation.json`, `results/ablation_{uniform,weighted}/validation.json`.
- CNN 8-run sweep best46.654; anchored 16-run study (KL or KL+2 soft F1, full CSAT or repeated T, clean/aug) all choose identity epoch0 51.164; best trained49.902. Bounded residual cannot add mass at T zeros; soft surrogate differs from exact F1. `results/cnn_sweep/`, `results/anchored_sweep/`. Stop learned fusion tuning.
- Geometric 63-config screen after anchored results: T anchor, uniform/C reference, S/A exponents 0/.05/.15/.30, gamma1/1.25, eps1e-8, clip±4. Plain SAT(.30,.05,1)51.4965 vs ST51.4797, AT51.2024, T51.1638. Prior-relative CST51.248, full CSAT51.211. Correlated experts not independent Bayesian evidence. `results/geometric_fusion/validation.json`. No further broad fusion search.

### Complementarity and routing
- Peak oracle: T8.478px mean error vs GT-selected branch5.811px (−31.45%); auxiliary strictly better316/675, ≥2px225, ≥5px128. Not an F1 bound. `results/peak_oracle/validation.json`.
- Metric-aligned oracle (fixed C/S/A/T/SAT maps, official threshold grid): feasible/bound F1 singletons57.079734/57.081450, T-or-SAT52.435305/52.435412, extended58.063587/58.064510. Tangent majorization (201 weights); conditional finite-map/grid bound, not deployable or proof of learnability. `results/f1_oracle/validation.json`.
- Approved train-only utility routing after oracle: regress gain vs T at train threshold .006 using macro F1 derivatives, 62 full/11 GLC-only features, ridge and MLPs, clean/aug, 8 fits/32 policies. Best linear51.3686, GLC-only51.3394, below ST/SAT. Conditional compute unmeasured. `results/expert_router/`. Stop routing.

### Bounded predictor replacement (approved 11-27, refined 11-28)
| Cue | Design and outcome | Evidence |
|---|---|---|
| E | Ego4D GLC, same architecture; E26.920, best TE51.397 below SAT. Never use for Ego4D transfer. | `results/glc_ego4d/{val,fusion}/` |
| H | 100DOH blocked by Torch/CUDA/GCC and missing CUDA headers; no accuracy/latency measured, retain isolated env. GazeFlow exceeds budget (~16GB). | r72 build history |
| D | Frozen ViT-S/14, 256→252, ImageNet norm, 384×18×18 features; 384→64→32→1 head, GELU, 64² softmax. Train3,202 fixations, KL AdamW .001/WD .01, batch32,10 epochs, seed42; epoch8 D27.6517, best TD51.3866. Slot-matched STD51.5010, DAT51.3765, SDAT51.5127. | `results/dinov2_s/{head_clean,screen,replacement_screen}/` |
| G/P | G refit Gaussian on same 3,202 fixation coordinates, min sigma .02; P=(sum target heatmaps+one uniform unit-mass prior)/3203. 15 fixed candidates, prior exponent .05. G15.057/P13.407; P+STD51.532690 vs G+STD51.532689, STD51.501043. No meaningful estimator advantage. | `results/spatial_prior_screen/validation.json` |
| Q/M | Q five direction-conditioned train priors shrunk to P by64; M camera-compensated local flow (GFTT/LK/RANSAC + Farneback); unreliable Q→P, M→uniform. 13 fixed candidates; Q14.392/M5.748, Q+STD51.5168, P+MDT51.1828 below P+STD. Unreliable42.67% val fixations, motion processing17.668ms, not E2E. | `results/motion_screen/validation.json` |
| R/O | Frozen COCO MobileNetV3 320 FPN Faster R-CNN; O Gaussian box-center mixture, R dominant-class train prior shrunk to P by64. 13 fixed candidates; O12.325/R15.015, R+STD51.533561 vs P+STD51.532690. Val85.36% nonempty, detector13.301ms/106.486MiB, not E2E; no hand class or established gaze relevance. | `results/object_screen/screen/validation.json` |
Fixed replacement/addition studies used T anchor and slot-matched coefficients without retuning; use exact source configurations. Exact grids and cue implementations in `experiments/{screen_dinov2,audit_dino_replacements,audit_spatial_prior,screen_motion,screen_objects}.py` and r72. Close replacement search; preserve negative results, avoid causal or equivalence claims.

## Final validation and profiling (11-28)
`experiments/finalize_validation.py`; full `results/final_selection/summary.json` verified. All ten live FP32 F1 exactly match aligned caches; all AMP/profile stages passed. Max absolute primary AMP drift .008855pp; thresholds unchanged. DT(.05) duplicates earlier TD(.05). All values below are validation; latency and memory are measured AMP, not theoretical.

| Candidate | FP32 F1% | AMP F1% | Model/E2E ms | Peak allocated MiB | Params M |
|---|---:|---:|---:|---:|---:|
| T | 51.163842 | 51.164318 | 43.06/109.32 | 455.28 | 70.182 |
| ST | 51.479680 | 51.485865 | 75.38/142.70 | 663.69 | 79.683 |
| P+ST | 51.494837 | 51.495139 | 75.48/142.74 | 663.71 | 79.683 |
| DT | 51.232503 | 51.236024 | 48.25/113.43 | 541.27 | 92.282 |
| P+DT | 51.258821 | 51.267183 | 48.28/113.32 | 541.28 | 92.282 |
| P+T | 51.198467 | 51.206181 | 43.03/108.24 | 454.42 | 70.182 |
| STD | 51.501043 | 51.509898 | 81.17/148.47 | 749.65 | 101.782 |
| P+STD | 51.532690 | 51.531345 | 81.33/149.38 | 749.66 | 101.782 |
| SAT | 51.496506 | 51.502381 | 83.70/151.33 | 786.17 | 111.490 |
| weighted CSAT | 51.355716 | 51.358151 | 83.71/151.85 | 786.18 | 111.490 |

Protocol: batch1, 48 evenly spaced val indices, 8 warmup twice around allocator cleanup, synchronized GPU input→map model latency; E2E common loader/decode incl histories/gaze prep + transfer + inference/fusion + D2H, excluding scoring, alignment, display and initialization. PyTorch allocated/reserved distinct from total device use. No streaming optimization or live FPS claim. P+ST AMP vs FP32 reduces model116.39→75.48ms, peak747.10→663.71MiB, same stored weights. P+ST vs P+STD: −.036206pp, 21.71% fewer params; vs weighted CSAT: +.136988pp, 28.53% fewer params, worse KL.

Validation paired subject bootstrap, 10,000 seed42, fixed candidate thresholds, pool frame P/R then F1, six unbalanced subject clusters (52/40/218/150/201/14 fixations). All nine 95% intervals include zero. FP32 pp: ST−T +.315838 [−.132021,.667546]; P+ST−ST +.015157 [−.064362,.079883]; P+STD−P+ST +.037853 [−.052403,.112104]; P+STD−T +.368848 [−.011636,.590028]. Exploratory, neither independent confirmation nor equivalence.

## Locked EGTEA test (11-28; original AMP only)
`experiments/evaluate_locked_v3.py`, `results/locked_evaluation_v3/summary.json`. One locked inference pass; no adaptation or retuning. All 2,021 deterministic target IDs/labels/branch alignment checked; 32 sequential decode checks; all included filename length differences zero. Official 2,022 clips: excluded OP03 historically considered corrupt now decodes all 361 frames and passes loader/sequential checks. User-approved 2,021 cohort retained before scoring, excluded clip not inferred/scored; do not claim present corruption. Full-cohort sensitivity requires separate approval. v1 stopped at omitted GLC source patches, v2 passed supplemental patch reconstruction then stopped at exclusion review; preserve both attempts and patches. v3 completed; uploaded summary, locks, counts, fixed F1 and bootstrap independently checked. Full environment reproduction/raw VM arrays not independently verified.

| Model | Fixed F1% | Adaptive F1% | Fixed legacy% | AUC-J | NSS | KL | Nominal AAE° |
|---|---:|---:|---:|---:|---:|---:|---:|
| T | 45.569797 | 45.569797 | 33.341944 | .912432 | 2.504254 | 3.327868 | 8.787176 |
| weighted CSAT | 45.765737 | 45.765737 | 33.151176 | .915447 | 2.523569 | 2.545682 | 8.713147 |
| P+T | 45.614276 | 45.614276 | 33.410265 | .914018 | 2.507975 | 3.335248 | 8.773378 |
| P+ST | 45.968535 | 46.035608 | 33.691708 | .917979 | 2.541483 | 3.464370 | 8.684970 |
| P+STD | 46.047717 | 46.071259 | 33.744242 | .918888 | 2.545428 | 3.486795 | 8.662813 |

Paired subject bootstrap (10,000 seed42, fixed thresholds, 95% unadjusted percentile intervals, pp): P+ST−T +.398738 [.126578,.644394]; P+ST−weighted CSAT +.202798 [−.004439,.406073]; P+STD−P+ST +.079182 [.035120,.122302]; P+T−T +.044479 [.018655,.078614]; P+STD−weighted CSAT +.281980 [.066455,.493975]. Descriptive; P+ST improves localization but worsens KL vs T. No independent-subject generalization, equivalence, across-metric dominance, published GLC superiority or streaming claim.

## Compression (11-28; validation complete)
`experiments/screen_compression.py` → `results/compression_screen/summary.json`; `experiments/complete_compression.py` → `results/compression_bn_safe/summary.json`. No test reads, training, dependency changes or threshold tuning. Five locked models, all 1,660 val clips/675 fixations, same profiling protocol; fresh AMP control max drift .00192745pp (<.01 gate). FP16 stores floating state in FP16 except protected state; symmetric signed INT8 per-output-channel Linear/Conv1d/2d/3d weights with FP32 scales, remaining floats FP16 except protected state; integer buffers unchanged. Dequantize on CPU to FP16 before GPU transfer, not native INT8 CUDA. FP32 inputs/static maps/output/fusion, CUDA autocast. Strict saved/reloaded state verification, original factory/weights still needed (not standalone). No Gaussian quantization. Resume gates verify hashes.

Initial S pack/smoke failed both formats: BN running variance overflow (`net.backbone.base3.0.branch2.0.bn.running_var`, max179166.71875). Corrective batch preserved 42,432 S BN floating elements in FP32, no clamping; strict restore, reused hash-verified T/A/D maps/checkpoints, rescored reuse controls, inferred S only. Both stages/six profiles completed; v1 and corrective evidence retained. Uploaded summaries/source hashes, thresholds, populations, provenance and arithmetic checked; VM binary contents not directly inspected.

| Config | Fresh AMP F1% | FP16 F1% | INT8 storage F1% | FP16/INT8 file MiB | FP16/INT8 peak MiB | FP16/INT8 model ms |
|---|---:|---:|---:|---:|---:|---:|
| T | 51.166246 | 51.158313 | 51.149352 | 133.963/67.932 | 315.839/315.839 | 41.736/41.901 |
| P+T | 51.206241 | 51.202698 | 51.153347 | See v1 | See v1 | See v1 |
| weighted CSAT | 51.358638 | 51.355920 | 51.315930 | 213.071/107.785 | 574.766/574.766 | 79.112/79.756 |
| P+ST | 51.496580 | 51.500036 | 51.457809 | 152.354/77.346 | 511.108/511.108 | 72.831/73.100 |
| P+STD | 51.530204 | 51.526854 | 51.495467 | 194.546/99.204 | 554.613/554.613 | 77.936/78.499 |

P+ST fresh AMP 304.28MiB, 663.71MiB peak, model76.05ms. FP16 file−49.93%, runtime state−49.97%, peak−22.99%, model latency−4.24%, F1 +.003456pp; E2E140.114ms. INT8 storage file−74.58%, same FP16 runtime memory, model73.100ms/E2E140.294ms, F1−.038771pp. All ten candidates within proposed .10pp engineering loss tolerance (not equivalence; tolerance not separately approved). Recommend FP16, INT8 storage as size alternative; adoption pending and explicitly post-original-test decision. Compressed held-out accuracy unverified.

Native dynamic INT8 CPU feasibility: T only, x86, 92 Linear modules, two threads/four deterministic val inputs; state280884285→74813581 bytes, mean1345.674→1082.376ms, L1 map difference .0478–.1144. No CPU full F1 or CUDA INT8 result. INT4 not demonstrated. Do not conflate CPU feasibility, storage compression and GPU compute.

## Validity and reporting constraints
- Released EGTEA GLC checkpoint likely exposed to project validation, possible test exposure; exact training manifest/selection unknown. Upstream val/test use same official test list; upstream 8 linspace frames over nominal64, all eight scored vs our last target, preprocessing/decoder/label differences not fully audited. Published44.80% is context, not direct superiority. [Lai et al.](https://doi.org/10.1007/s11263-023-01879-7).
- Test clip-disjoint but subject overlap; validation selection exploratory, small/unbalanced validation clusters; metric correction after validation; in-sample stacking; Huang checkpoint exposure unresolved. Transfer is no target adaptation, not assured zero-shot. Preserve excluded-cohort discrepancy, negative results and KL tradeoff. No causal/equivalence claims or significance based solely on unadjusted descriptive intervals.
- Profiling standardized batch1 offline, not deployed throughput; FP16/INT8 storage results validation only; current factory not standalone. No test-driven tuning, including qualitative inspection; log any additional held-out access and distinguish validation diagnostics from reporting.
- External GTEA (2012, 17 sequences/14 participants) absent; author [dataset](https://ai.stanford.edu/~alireza/GTEA_Gaze_Website/GTEA_Gaze.html) [archive](https://webshare.ipat.gatech.edu/coc-rim-wall-lab/web/afathi3/GTEA_Gaze/GTEA_Gaze_Dataset.zip) HTTP502/direct fetch; VM wget DNS failure. GTEA Gaze+ 2015 is subsumed by EGTEA, not independent. Ego4D alternate pending user scope/license/access; [start](https://ego4d-data.org/docs/start-here/), [gaze docs](https://ego4d-data.org/docs/data/gaze/), [GLC preprocessing](https://github.com/BolinLai/GLC/blob/main/slowfast/datasets/DATASET.md). Raw gaze CSV lacks explicit fixation class; lock timestamp alignment/eligibility/sampling before scoring; report separately if no independently defined comparable fixation proxy. Audit recording and pretrained checkpoint overlap; exclude E from Ego4D transfer. Bounded UID download, separate environment/disk audit, never share credentials. No external scoring yet.

## Active assets and provenance
Repository (read-only): https://github.com/hady-mahmoud-aus/gaze_estimation_ensemble. VM Linux `~/ACV Project`, Conda `gaze`, Python3.10/Torch2.5.1/CUDA12.1/TV0.20.1/OpenCV4.10/einops0.8.2, Quadro RTX4000 8GB, driver535.86.10. Protect env; optional FairScale/Detectron2 imports patched; source revisions/patches in `patches/`, Conda specs in `environment/`. `handobj` isolated and blocked. No new installs approved.

Layout: `agents/`, `datasets/`, `models/`, `fusion/`, `experiments/`, `scripts/` (capture_reproducibility.py), `tests/`, `utils/`, `environment/`, `patches/`, README/.gitignore. Ignored `data/`, `checkpoints/`, `external/`, `results/`. Run `python -m experiments.<module>` from root. Active final entry points: `finalize_validation.py`, `evaluate_locked_v3.py`, `screen_compression.py`, `complete_compression.py`; retain `screen_dinov2.py`, `screen_glc_ego4d.py`, `train_fusion.py`, `audit_geometric_fusion.py`, `audit_metrics.py` for shared helpers/tests. The remaining 13 Python files include `__init__.py`, `train.py`, `cache_predictions.py`, `evaluate.py`. Historical evaluators v1/v2 are not active source files; their lock/patch evidence remains live. VM deletions uncommitted/unpushed. Historical migration and prior sync details in r72. Source patch reconstruction passed (not full environment reconstruction).

Selected cache `results/cache_selected/{val,train_clean,train_aug}`; D `results/dinov2_s/head_clean/`; P `results/spatial_prior_screen/{P.npy,validation.json}`; keep active weights/priors/maps/manifests. Historical cleanup approved 11-27: VM54→25GiB, removed disposable outputs, original archives, 11 unselected adaptation weights, original/adapted train maps; selected weights/caches, validation caches, extracted data, compact evidence retained. Archive metadata checked, byte recovery unverified; historical training cache reruns not directly available. Subsequent verified consolidation below.

Key SHA-256 anchors (full detailed manifests/digests in result JSONs and historical r72):
| Artifact | SHA-256 |
|---|---|
| GLC EGTEA weight | `0e2004d4b029c781f8bc7a29a6549cf0b70ddbd178f1aa75873b6934d7cc19e8` |
| S weight | `5d097a6b145b2cff7f08aa141a91e7aec4ac967504b439f4b04110c7e475cbbd` |
| Final validation summary | `fc8ba7abd96d5955eac117b5c0a21a460f91050eeb3f84ba833a48c2930e4942` |
| finalize_validation.py | `b9765a776fe671c7e324aeea426afc3527d72163051332d7a1f3be71ec94a5a3` |
| Locked test summary | `ffa5fb845b915a96effe160a0e674cacb414b2db4a51858ec4652172f2e25345` |
| evaluate_locked_v3.py | `40f19506213504723acb30b547c1b93d49058b5e33a8579c9b651dc3fa41e375` |
| Compression v1 summary | `791149469a9ce2381ac5719b4968229e237f659ffa2a0d76c4c63b86b3b2fbfa` |
| screen_compression.py | `e05842ccb3fea17425029b16e3d33280ad7f7ecb90a4ecfdba830c996228205a` |
| BN-safe summary | `6eb32f9a9f7c945c0bb5362ce3e7a6a107890e8b6576aeec09aaf03ddd71d490` |
| complete_compression.py | `18aac6f32751a46381ab5b0debbfc74f6d6ed3553cea7c5a23c38c681e146261` |

Registry `EXPERIMENTS.csv` r43, 199 unique runs, SHA-256 `f0246f2f630229fddb8c17e63c21297af2f844f593608a4f045c959ff1bb15f2`. Rounded F1%, unknown fields blank. Five locked test rows use fixed thresholds. Ten compressed validation rows; no repeated controls, failed packs, CPU probe, or oracle/smoke rows. Weighted/geometric grids registered by winners, full grids in JSON. Do not duplicate run IDs or reinterpret historical legacy metrics.

Documentation cleanup r73 (11-28, user approved while paused): consolidated r72 PROJECT/MILESTONES without changing research decisions or CSV. Exact prior Git blobs (read-only history; `git show <blob>` in a clone containing them): PROJECT r72 `192b7cc6d0c34a02ab28f222a03f2c337074313f`; MILESTONES r72 `e563fe93147e7eb42036c957911e7e5be44ee130`. Retain history for detailed prior grids, source/weight/cache hashes, audit logs, and literature leads; verify these blobs remain accessible before manual replacement. No separate history file needed. AGENTS r62 and registry r43 unchanged.

Maintenance (2026-11-28, approved and executed while experiments paused): VM cleanup report PASS. 508→105 result files (404 consolidated, 104 original files live), 111→27 result directories, 24→13 experiment Python files; 30 listed bytecode files and empty GTEA download removed per script. Bundle `results/historical_evidence.tar.gz`, 2,512,382,062 bytes, SHA-256 `2a8a053f38cc2ac2468c60520bf38466b2300bcfb0cb757b20ea47fabf353d43`; embedded `_acv_cleanup_manifest.json` preserves original paths, sizes and hashes; every archived file verified before removal. Registry paths remain unchanged: if absent live, look for exact member in bundle; extract to temporary location, not over active results. Source recovery tag `acv-pre-cleanup-20261128` at `41efda480a46471fddd1559a839dbec5dcd7acce`; 11 retired scripts recoverable there. CPU regression tests 23/23 before and after, 19 integrity anchors and 199 unique registry rows passed, retained direct import check passed. No model inference or scoring. Report uploaded as `acv_cleanup_execution.txt`; actual archive not uploaded/independently rehashed here. External hand-object checkout, Ego4D weights and upstream backups untouched pending separate recovery audit. README r75 prepared locally, sync pending; no remote writes.

## Deliverables and paper priorities
| Rubric | Weight | Evidence |
|---|---:|---|
| Problem | 10% | Objective, constraints, related work |
| Method/code | 30% | Alignment, augmentation examples, manifests, reproducible repo, setup/weights/configs/commands |
| Results/analysis | 30% | Learning curves, 15-subset rankings, fusion/augmentation, F1 vs VRAM/latency/size, both datasets, qualitative failures |
| Report | 20% | Abstract, intro, related work, methods, results, discussion/limitations, conclusion |
| Demo/Q&A | 10% | Video overlays, branch maps, optional GT, measured speed/VRAM; label offline inference |

Literature leads (not systematic review): Lai above; [Kümmerer 2018](https://doi.org/10.1007/978-3-030-01270-0_47) metric-specific readout; [Hinton 2002](https://doi.org/10.1162/089976602760128018) products of experts; [He 2016](https://doi.org/10.1109/CVPR.2016.90) residuals; [Zhao 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Zhao_Optimizing_the_F-Measure_for_Threshold-Free_Salient_Object_Detection_ICCV_2019_paper.html) F-measure; [Narasimhan 2015](https://proceedings.mlr.press/v37/narasimhana15.html) nondecomposable metrics; [Tavakoli 2019](https://doi.org/10.1109/WACV.2019.00035) manipulation; other leads in r72. No directly comparable published results established. Prioritize controlled evidence, honest limitations and demo/report over unsupported low-bit debugging.
