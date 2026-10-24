# Recovery notes — 2026-10-23

Selected the physics-informed model instead of an adjacent data-only baseline. Removed module-level execution from the imported model and fit envelope statistics on training simulations only.

## Shared changes

Added an explicit data root, deterministic CPU seeds, dependency declarations, synthetic demonstration, finite loss/gradient/prediction checks, held-out evaluation, checkpoint replay with protocol checks, and source provenance. Train/test preprocessing is separated. Archival code is not executed at import.

## Deviations and unverified parts

The runner covers healthy training and held-out reconstruction scores. The archived threshold and MATLAB SPRT code are preserved but are not integrated into this runner. Fault trajectories, threshold calibration and the full paper detection comparison have not been rerun.

## 2026-10-24 release-readiness audit

- Specify Python 3.12, update the pip bootstrap, align default dependency installation with tested pins, and record a macOS transitive snapshot.
- Validate CLI values, reject empty/nonfinite normalization inputs, protect existing outputs, and record environment metadata.
- Add version-2 checkpoint protocols with input-file fingerprints. Older checkpoint protocols require retraining.
- Add regression tests and a CPU CI matrix; CI execution status is reported separately from local tests.
- Clarify runnable versus archival code, data confidentiality, reproducibility limits and release rights in the documentation.

## Follow-up release hardening — 2026-10-24

- Protocol v3 removes absolute machine paths, supports data-root relocation and binds replay to the repository and maintained source hashes. Recreate v1/v2 checkpoints.
- Add replay privacy/provenance regressions and dependency advisory checks on all three hosted operating systems.
- Avoid duplicate push/PR checks and cancel superseded workflow runs.
- Record explicit release decisions; no rights, dataset license or scientific approval is assumed.
