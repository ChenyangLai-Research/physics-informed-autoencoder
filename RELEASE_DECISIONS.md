# Release decisions requiring owner confirmation

Status: **pending — keep private; do not merge or publish**. Recorded 2026-10-24.
Passing execution tests does not answer the following questions. Potential contributors named below are parties to verify, not a legal determination of ownership.

## AE-R — Code rights and license

Confirm author/institution and actuator-simulation contributor rights for the Python models and MATLAB SPRT implementation. Identify any separately licensed detection or threshold code.

Required response: identify the rights holder and approver, the permitted code/legacy-file scope, the written approval or applicable license, and the exact project-wide license authorized. State whether publication of embedded constants and source provenance is permitted. No project-wide license will be inferred from a paper or a nearby license file.

## AE-D — Data and disclosure

For healthy.mat (AME_dis_mat/current_mat) and any fault/calibration sets, identify owners and permitted access/reuse. Confirm whether physics-envelope constants may be published.

Required response: dataset name/version, rights holder, access URL or contact process, permitted uses and redistribution terms. Choose **code plus synthetic data only**, **code plus an approved request-access route**, or **code plus specifically licensed data**. Approve disclosure of schemas/identifiers separately. Until answered, no research data or trained weights are included.

## AE-M — Scientific scope

Current tests cover healthy reconstruction only. Provide the independent calibration set, threshold selection, SPRT alpha/beta and stopping rules, and fault evaluation protocol if a fault-detection claim is intended; otherwise approve reconstruction-only scope.

Required response: approve the documented limited scope, or supply the authoritative equations/script/experimental protocol needed for a change. Smoke tests are not evidence that the scientific choice is correct.

## AE-H — Original Git metadata

The original commit retains a local machine email/hostname. New audit commits use the GitHub noreply address. Confirm either (a) retaining the original metadata, or (b) preparing a separately reviewed sanitized public history with required authorship/attribution preserved. No existing branch/history will be force-rewritten as part of this review.

## AE-A — Archive disclosure

Choose whether the eventual public release should include the selected `legacy/` files, a reviewed/redacted subset, or only the maintained runner plus provenance/attribution. Review historical paths, filenames, names and scientific constants before deciding. Original research folders remain unchanged.

Record decisions here with approver, date, scope and evidence reference. Do not paste confidential agreements or original datasets into this repository. An answer to these questions is not an instruction to change repository visibility or merge this PR.
