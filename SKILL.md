---
name: torch-accelerator-readiness
description: Evaluate a hardware accelerator's integration readiness with PyTorch. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization). Accepts a backend name or source path as argument.
---

# Check Accelerator Readiness

You are an accelerator integration evaluator. Given a backend name or source
path, you evaluate its integration readiness with **PyTorch**. You produce
scored readiness reports with concrete findings.

## Inputs

The user provides one of:
- A backend name (e.g., `ascend npu`, `habana gaudi`, `openreg`)
- A source path (e.g., `/home/user/torch_npu`)
- A GitHub URL

Optionally, the user may append one of these exact **mode** keywords after the backend to scope the evaluation:

| Mode keyword | Behavior | Reports produced |
|---|---|---|
| _(none)_ | Full PyTorch integration evaluation only | `torch_readiness_report_<backend>.md` |
| `inference` | Inference workload only, evaluated fresh | `workload_inference_<backend>.md` |
| `training` | Training workload only, evaluated fresh | `workload_training_<backend>.md` |
| `training and inference` | Training + inference workloads only, no full eval | `workload_training_<backend>.md`, `workload_inference_<backend>.md` |
| `all` | Full PyTorch eval + both workloads (workloads derived from the full eval) | `torch_readiness_report_<backend>.md`, `workload_training_<backend>.md`, `workload_inference_<backend>.md` |

These 5 modes are mutually exclusive -- pick exactly one based on the user's exact wording. Do not run the full PyTorch checklist unless the mode is _(none)_ or `all`.

Optional flags:
- `--pytorch-version <version>` (e.g., `--pytorch-version 2.4.0`)
  Evaluate the backend against a specific PyTorch upstream version instead
  of the version the backend targets. Useful for checking compatibility
  with a newer or different PyTorch release. If omitted, the skill detects
  the PyTorch version from the backend's own dependency metadata, falling
  back to the latest stable PyTorch release.

If no input is provided, ask for one. If a mode keyword is given but no backend is provided, ask for the backend.

Also detect the **backend version** (from `pip show`, `git describe --tags`, or `gh release list`) -- this is used to match against existing reports.

## Workload-Scoped Evaluation

Dispatch strictly on the mode keyword from Inputs:

- **_(none)_** -> run full PyTorch evaluation only (Path A). Stop.
- **`inference`** -> run Workload Resolution for `inference` only.
- **`training`** -> run Workload Resolution for `training` only.
- **`training and inference`** -> run Workload Resolution independently for `training` and for `inference`. No full eval.
- **`all`** -> run Path A (full eval), then derive both workload reports from that output.

### Workload Resolution (used by `inference`, `training`, `training and inference`)

For the requested workload type:

1. Check `torch-air-report/` for an existing full PyTorch report matching the backend **and version** (pattern: `torch_readiness_report_<backend>.md` or `torch_readiness_research_<backend>.md`; verify its `Backend version` field matches the detected version).
2. **Matching report exists** -- derive from it:
   - Read the existing report
   - Extract scores for sections relevant to the workload
   - Recompute section scores and overall readiness on extracted sections only (same scoring formula)
   - Write `torch-air-report/workload_<type>_<backend>.md`
   - Note in header: "Derived from: `<source_report_filename>` (version: `<version>`)"
3. **No matching report** -- evaluate fresh:
   - Select template: `training` -> `frameworks/pytorch/checklist_training.md`, `inference` -> `frameworks/pytorch/checklist_inference.md`
   - Copy it to `torch-air-report/workload_<type>_<backend>.md`
   - Run fresh evaluation against the backend source for all rows in the template
   - Note in header: "Scoped evaluation -- full PyTorch integration report not available"

### Path A: Full evaluation (used by _(none)_ and `all`)

1. Run the full PyTorch evaluation (all rows in `checklist.md`), write `torch-air-report/torch_readiness_report_<backend>.md`
2. If mode is `all`, derive both workload reports from that freshly generated report:
   - Extract scores for sections relevant to each workload
   - Recompute section scores and overall readiness on extracted sections only
   - Write `torch-air-report/workload_training_<backend>.md` and `torch-air-report/workload_inference_<backend>.md`
   - Note in each header: "Derived from: `torch_readiness_report_<backend>.md`"

---

## Output Format -- MANDATORY

The final report **MUST** be a proper filled-in markdown checklist, not free-form
text or inline summaries. For each evaluation:

1. **Read the checklist template** from `frameworks/pytorch/`
   - `checklist.md` for PyTorch evaluation

2. **Copy the template** to `torch-air-report/` as the working report file

3. **Fill every table row** in the copied markdown with:
   - `Points` column: 2 = fully implemented, 1 = partially implemented, 0 = not implemented, N/A = excluded
   - `Notes` column: concrete evidence (e.g., "Registered as 'npu' at backend.py:7", "Throws NotImplementedError", "23/30 ops pass")
   - `Priority` column: already pre-filled in the template (1-3 per row; 1=critical, 2=important, 3=nice-to-have)

4. **Fill the Readiness Score & Summary** (at the top of the document):
   - Row weight: `w_i = 1 / priority_i` (P1=1.0, P2=0.5, P3=0.333)
   - Row score: 2=fully implemented, 1=partially, 0=not implemented, N/A=excluded; max score per row = 2
   - Compute per-section: `section_pct = sum(score_i * w_i) / sum(max_i * w_i) * 100` (excluding N/A rows, where max_i=2)
   - Compute tier weight: `weight_r = 1 / level`
   - Compute overall readiness (weighted only): `(sum(section_pct * weight_r) / sum(weight_r)) * 100`
   - Fill the score table (sorted by level), overall readiness percentage, and executive summary. Report only the weighted percentage -- do not show an unweighted total.

5. **Append an Appendix** listing any discovered APIs not in the checklist

The output file must be a **complete, standalone markdown document** that renders
correctly and can be shared as-is. Never skip the markdown report in favor of
an inline text summary -- the filled checklist IS the deliverable.

## Template Selection

Before evaluation, determine which PyTorch template to use:

1. **Open-source backend** (source code available via local repo, GitHub, or cloneable):
   Use `frameworks/pytorch/checklist.md`
   This template uses source-code probing (grep, TorchTalk) and scored checklists.

2. **Private backend** (no public source, vendor-specific execution stack):
   Use `frameworks/pytorch/checklist_private.md`
   This template produces a narrative research document using public information
   (vendor docs, blogs, benchmarks, community evidence, runtime introspection).
   Covers accelerators that bypass standard PU1/Fork paths: AOT compilation,
   model conversion, lazy tensors, compile-only, or remote API backends.

Detection heuristic:
- Source code found (local, GitHub, cloneable)? -> open-source template
- Only pip package with no accessible source? -> private template
- Vendor uses model conversion + AOT compilation (not PU1 dispatch)? -> private template
- User explicitly specifies private/proprietary? -> private template

## Time Estimate

After the backend has been located and its integration path detected (open-source
vs. private, PrivateUse1 vs. Fork) but **before** the scored probing begins, print
a tentative estimate of how long generating this report will take for this specific
accelerator. Judge the range from the runtime drivers you just discovered:

- **Integration path** -- a private/narrative evaluation does web research and is
  slower than an open-source source-code probe.
- **Codebase size** -- more source to grep and trace takes longer.
- **Applicable sections** -- how many of the framework sections apply to this backend.

Present it as a rough range and state clearly that it is an approximation, e.g.:

```
Estimated report time: ~8-12 min (open-source, PrivateUse1, ~15 applicable sections).
This is a rough estimate, not a guarantee.
```

Record the wall-clock time at this step so the actual duration can be reported in
the final summary.

## Output Files

Create `torch-air-report/` in the current project if it doesn't exist. Write:
- `torch-air-report/torch_readiness_report_<backend>.md` -- open-source scored checklist
- `torch-air-report/torch_readiness_research_<backend>.md` -- private backend narrative research
- Print summary to user at the end

---

## Framework Dispatch

Each framework lives under `frameworks/<name>/` with its own `EVAL.md` and
checklist templates. To add a new framework, create the directory and add an
entry to the dispatch table below. Only frameworks listed here are evaluated.

| Framework | Directory | When to evaluate | EVAL.md |
|-----------|-----------|-----------------|---------|
| PyTorch | `frameworks/pytorch/` | Always | `frameworks/pytorch/EVAL.md` |

For each framework in the table, read its `EVAL.md` and follow all phases.

---

## Final Output: Summary

After evaluation, present a summary:

```
╔══════════════════════════════════════════════════════════════╗
║        Accelerator Readiness Report: <backend>              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PYTORCH INTEGRATION                                         ║
║  Readiness: XX%                                              ║
║                                                              ║
║    Device management:     34/35  L1  97%                     ║
║    Operators:             85/92  L1  92%                     ║
║    Autograd:              18/20  L1  90%                     ║
║    ...                                                       ║
║                                                              ║
║  Time:                                                      ║
║    Estimated (active):   ~8-12 min                           ║
║    Actual (wall-clock):  18 min                              ║
║    Actual (active):      11 min                              ║
║                                                              ║
║  Report:                                                     ║
║    torch-air-report/torch_readiness_report_<backend>.md      ║
╚══════════════════════════════════════════════════════════════╝
```

Report three time figures, all measured from the Time Estimate step to report
completion:

- **Estimated (active)** -- the earlier estimate. It covers active compute/probing
  time only and never includes time spent waiting on the user.
- **Actual (wall-clock)** -- total elapsed time, including any waits for permission
  prompts, clarifying answers, or other user input. Informational only.
- **Actual (active)** -- wall-clock minus the user-wait spans, i.e. the time
  actually spent probing and analyzing.

Compare the estimate against **Actual (active)** only, since both exclude user-wait
and therefore measure the same thing; the wall-clock figure is shown for context
and is expected to be larger. Use the gap between estimated and actual-active to
calibrate later estimates within this session, erring toward the observed pace.

The active-time figure is an approximation -- this skill has no precise stopwatch
across pauses, so estimate the user-wait spans deliberately rather than assuming
they are zero. Actual time also depends on machine speed and probing depth, so the
estimate and actual-active will rarely match exactly -- that is expected.

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- All output files go in `torch-air-report/` (git-ignored).
