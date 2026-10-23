# Recovery notes — 2026-10-23

Selected the physics-informed model instead of an adjacent data-only baseline. Removed module-level execution from the imported model and fit envelope statistics on training simulations only.

## Shared changes

Added an explicit data root, deterministic CPU seeds, dependency declarations, synthetic demonstration, finite loss/gradient/prediction checks, held-out evaluation, checkpoint replay with protocol checks, and source provenance. Train/test preprocessing is separated. Archival code is not executed at import.

## Deviations and unverified parts

The runner covers healthy training and held-out reconstruction scores. The archived threshold and MATLAB SPRT code are preserved but are not integrated into this runner. Fault trajectories, threshold calibration and the full paper detection comparison have not been rerun.
