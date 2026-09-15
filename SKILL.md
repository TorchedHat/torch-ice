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

| Mode keyword | Behavior |
|---|---|
| _(none)_ | Full PyTorch integration evaluation only -- no workload summaries |
| `inference` | Full evaluation + Inference Readiness Summary section |
| `training` | Full evaluation + Training Readiness Summary section |
| `both` | Full evaluation + both Training and Inference Readiness Summary sections |

All modes run the same full PyTorch evaluation and write a single report: `torch_readiness_report_<backend>.md`. The mode only controls whether workload summary section(s) are added to that report -- it never changes what gets evaluated or produces separate files.

Optional flags:
- `--pytorch-version <version>` (e.g., `--pytorch-version 2.4.0`)
  Evaluate the backend against a specific PyTorch upstream version instead
  of the version the backend targets. Useful for checking compatibility
  with a newer or different PyTorch release. If omitted, the skill detects
  the PyTorch version from the backend's own dependency metadata, falling
  back to the latest stable PyTorch release.

If no input is provided, ask for one. If a mode keyword is given but no backend is provided, ask for the backend.

Also detect the **backend version** (from `pip show`, `git describe --tags`, or `gh release list`) -- this is used to match against existing reports.

## Workload Readiness Summaries

Always run the full PyTorch evaluation first (all rows in `checklist.md`), fill every row's Points and Notes as usual. Then, based on the mode keyword:

- **_(none)_** -- do not add any workload summary. Report is done.
- **`inference`** -- add a **Inference Readiness Summary** subsection.
- **`training`** -- add a **Training Readiness Summary** subsection.
- **`both`** -- add both subsections.

### Computing a workload summary

For the requested workload, using the already-scored rows in the same report:

1. Identify which sections/rows are relevant to the workload (e.g., training relies on Autograd, AMP, Distributed Training, Serialization/checkpointing; inference relies on torch.compile/Inductor, Quantization, Serialization/model loading, Dtype Support). Rows/sections irrelevant to the workload are excluded from this calculation.
2. Recompute overall readiness for just those rows using the same formula (`w_i = 1/priority_i`, `section_pct = sum(score_i*w_i)/sum(max_i*w_i)*100`, `weight_r = 1/level`, weighted average across the included sections).
3. Write a short narrative: **Overall <Workload> Readiness: X%**, **Key strengths**, **Key gaps**, **Recommendations** -- same structure and tone as the main Executive Summary, but scoped to the workload.
4. Do **not** print a per-section score table for the workload -- narrative + single percentage only. The full per-row detail already lives in the main checklist below.

### Placement

Insert the workload summary subsection(s) immediately **above** the "### Section Scores" table, after the main Executive Summary. Order: Executive Summary -> Training Readiness Summary (if requested) -> Inference Readiness Summary (if requested) -> Section Scores.

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
