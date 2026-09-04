# ChandraMatch Autonomous R&D State

## Generation
ChandraMatch v2 redesign.

## Frozen evidence
- Pair004: frozen LRO NAC ↔ TMC-2 baseline; automatic coarse selector converges to near-identity.
- Pair006: frozen geographically distinct LRO NAC ↔ TMC-2 generalization scene; automatic selector converges to ~692 px and the selected tiled-LightGlue branch passed the controlled perturbation response test.
- Pair007: held-out failure of the frozen v1 end-to-end matcher. Automatic selector converges to ~900 px, while tiled LightGlue and LoFTR both fail controlled perturbation robustness. A separate appearance diagnostic strongly improves after the ~903 px warp, so the coarse basin is likely meaningful but fine correspondence is unstable.

## Scientific rules
1. Never treat RANSAC reprojection RMSE as independent ground-truth accuracy.
2. Controlled cross-sensor perturbation response is robustness/equivariance evidence, not absolute GT accuracy for the unperturbed pair.
3. Preserve Pair007 as a v1 held-out failure; diagnostics must not retroactively turn it into a pass.
4. No manually supplied offsets or scene-specific parameter tuning in held-out evaluation.
5. Use identical v2 parameters for Pair004, Pair006, and Pair007 replay unless a change is declared globally and all three are rerun.
6. Prefer reproducible automated experiments and machine-readable verdicts over one-off scripts.

## v2 target architecture
1. Multiscale structural coarse search (SFOC/CFOG-style structural representation + FFT/correlation search).
2. Automatic top-K coarse-hypothesis scoring using geometry, overlap, structural similarity, and ambiguity measures.
3. Uniform control-point grid with structural local matching; RIFT-style or learned matchers used as fallback/corroboration.
4. Constrained robust transform plus local residual-consistency analysis.
5. Interior-ROI perturbation validation that keeps the same physical terrain in view.
6. Automatic PASS / AMBIGUOUS / FAIL confidence report.
7. Learned matchers are fallback/corroboration rather than the only registration mechanism.

## Replay targets
- Pair004 expected coarse basin: near identity.
- Pair006 expected coarse basin: approximately +692 px translation magnitude.
- Pair007 expected coarse basin: approximately +904 px translation magnitude, but v1 fine matcher is not validated.

## Current next task
Bring the self-hosted Windows/RTX 4070 runner online, pass the runner health check, then implement the v2 structural coarse-search module and replay Pair004/006/007 automatically.
