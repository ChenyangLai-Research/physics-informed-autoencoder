# Archival source files

These files document the original research implementation. They are not installed or executed by `run.py`.
They may execute training or filesystem writes on import. Do not import them as a library.

The curated Python entry point is `../run.py`; its dependencies are specified at the repository root.
The archive was syntax-checked, but its original experiments were not executed in this audit.

Historical module names are not importable under the numbered archive filenames: `thr_definition_new`. The archive is incomplete as a standalone execution environment.

The MATLAB SPRT script requires MATLAB and the original evaluation inputs. MATLAB/Octave execution was not verified.

Original source hashes and relative locations are recorded in `../project.json`. Original hashes remain provenance records. `archived_sha256` records each reviewed copy; `audit_edits` identifies safety changes. Original research folders were not changed.
