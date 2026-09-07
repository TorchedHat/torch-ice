---
name: torch-accelerator-readiness
description: Evaluate a hardware accelerator's integration and security readiness with PyTorch. Use when checking if an accelerator (XPU, NPU, openreg, HPU, custom) supports PyTorch's PrivateUse1/fork integration (device management, hooks, operators, AMP, autograd, torch.compile, distributed, profiler, serialization), or when assessing security posture (isolation, encryption, firmware, PyTorch integration surface). Accepts a backend name or source path, with optional --security or --all flags.
---

# Check Accelerator Readiness

You are an accelerator integration evaluator. Given a backend name or source
path, you evaluate its readiness with **PyTorch** — functional integration,
security posture, or both. You produce scored readiness reports with concrete findings.

## Inputs

The user provides:
- A backend name (e.g., `ascend npu`, `habana gaudi`, `openreg`)
- A source path (e.g., `/home/user/torch_npu`)
- A GitHub URL
- An optional evaluation mode flag (default: functional only)

### Evaluation Modes

| Flag | Mode | What runs |
|------|------|-----------|
| _(none)_ | Functional only (default) | PyTorch integration checklist |
| `--security` | Security only | PyTorch security checklist |
| `--all` | Both | Functional + security checklists |

### Optional flags

- `--pytorch-version <version>` (e.g., `--pytorch-version 2.4.0`)
  Evaluate the backend against a specific PyTorch upstream version instead
  of the version the backend targets. Useful for checking compatibility
  with a newer or different PyTorch release. If omitted, the skill detects
  the PyTorch version from the backend's own dependency metadata, falling
  back to the latest stable PyTorch release.

Examples:
```
/torch-air:torch-accelerator-readiness <accelerator name>
/torch-air:torch-accelerator-readiness <accelerator name> --security
/torch-air:torch-accelerator-readiness <accelerator name> --all
/torch-air:torch-accelerator-readiness /path/to/torch_npu --all
```

If no backend input is provided, ask for one.

## Output Format -- MANDATORY

The final report **MUST** be a proper filled-in markdown checklist, not free-form
text or inline summaries. For each evaluation run:

1. **Read the checklist template** from the appropriate path:
   - Functional: `frameworks/pytorch/checklist.md`
   - Security: `frameworks/pytorch/security/checklist.md`

2. **Copy the template** to `torch-air-report/` as the working report file

3. **Fill the metadata table** at the top of the report:
   - **torch-air version**: read from `VERSION` in the torch-air repo root; if missing, use `git describe --tags --always` in that repo
   - **PyTorch version**: the PyTorch version supported or tested by the backend — check `setup.py`, `pyproject.toml`, `requirements*.txt`, README compatibility table, or CI config
   - **Model**: record the AI model name used to generate the report (e.g., the model identifier from the runtime environment)

4. **Fill every table row** in the copied markdown with:
   - `Points` column: 2 = fully implemented/evidenced, 1 = partially, 0 = not implemented/evidenced, N/A = excluded
   - `Notes` column: concrete evidence (e.g., "Registered as 'npu' at backend.py:7", "VFIO IOMMU isolation documented", "23/30 ops pass")
   - `Priority` column: already pre-filled in the template (1-3 per row; 1=critical, 2=important, 3=nice-to-have)

5. **Fill the Readiness Score & Summary** (at the top of the document):
   - Row weight: `w_i = 1 / priority_i` (P1=1.0, P2=0.5, P3=0.333)
   - Row score: 2=fully, 1=partially, 0=not, N/A=excluded; max score per row = 2
   - Compute per-section/domain: `section_pct = sum(score_i * w_i) / sum(max_i * w_i) * 100` (excluding N/A rows, where max_i=2)
   - Compute tier weight: `weight_r = 1 / level`
   - Compute overall readiness (weighted only): `(sum(section_pct * weight_r) / sum(weight_r)) * 100`
   - Fill the score table (sorted by level), overall readiness percentage, and executive summary. Report only the weighted percentage -- do not show an unweighted total.

6. **Append an Appendix** listing any discovered APIs not in the checklist (functional only)

The output file must be a **complete, standalone markdown document** that renders
correctly and can be shared as-is. Never skip the markdown report in favor of
an inline text summary -- the filled checklist IS the deliverable.

## Template Selection (functional only)

Before functional evaluation, determine which PyTorch template to use:

1. **Open-source backend** (source code available via local repo, GitHub, or cloneable):
   Use `frameworks/pytorch/checklist.md`
   This template uses source-code probing (grep, TorchTalk) and scored checklists.

2. **Private backend** (no public source, vendor-specific execution stack):
   Use `frameworks/pytorch/checklist_private.md`
   This template produces a narrative research document using public information (vendor
   docs, blogs, benchmarks, community evidence, runtime introspection). Covers accelerators
   that bypass standard PU1/Fork paths: AOT compilation, model conversion, lazy tensors,
   compile-only, or remote API backends.

Detection heuristic:
- Source code found (local, GitHub, cloneable)? -> open-source template
- Only pip package with no accessible source? -> private template
- Vendor uses model conversion + AOT compilation (not PU1 dispatch)? -> private template
- User explicitly specifies private/proprietary? -> private template

## Time Estimate

After the backend has been located and the evaluation mode determined (open-source 
vs. private, PrivateUse1 vs. Fork) but **before** the scored probing begins, print 
a tentative estimate of how long generating the report(s) will take. Judge the range from:

- **Evaluation mode** -- `--all` runs both checklists; `--security` does web research
  and doc analysis; functional-only does source-code probing.
- **Integration path** -- private/narrative evaluation is slower than open-source probe.
- **Codebase size** -- more source to grep and trace takes longer.
- **Applicable sections** -- how many sections/domains apply to this backend.

Present it as a rough range and state clearly that it is an approximation, e.g.:

```
Estimated report time: ~8-12 min (functional only, open-source, PrivateUse1, ~15 sections).
Estimated report time: ~15-25 min (--all, open-source, ~15 functional + 6 security domains).
This is a rough estimate, not a guarantee.
```

Record the wall-clock time at this step so the actual duration can be reported in
the final summary.

## Output Files

Create `torch-air-report/` in the current project if it doesn't exist. Write:

| Mode | Output file |
|------|-------------|
| Functional (default) | `torch-air-report/torch_readiness_report_<backend>.md` |
| Functional (private) | `torch-air-report/torch_readiness_research_<backend>.md` |
| `--security` | `torch-air-report/torch_security_readiness_report_<backend>.md` |
| `--all` | Both functional and security report files above |

---

## Framework Dispatch

Each framework lives under `frameworks/<name>/` with its own `EVAL.md` and
checklist templates. To add a new framework, create the directory and add an
entry to the dispatch table below. Only frameworks listed here are evaluated.

Security is an evaluation **dimension** nested inside the
PyTorch framework (`frameworks/pytorch/security/`), not a peer framework.

| Framework | Directory | When to evaluate | EVAL.md |
|-----------|-----------|-----------------|---------|
| PyTorch (functional) | `frameworks/pytorch/` | Default, or `--all` | `frameworks/pytorch/EVAL.md` (Part 1) |
| PyTorch (security) | `frameworks/pytorch/security/` | `--security` or `--all` | `frameworks/pytorch/security/EVAL.md` |

**Dispatch rules:**
- No flag (default): run Part 1 (functional) only. Do NOT run security.
- `--security`: run security evaluation only. Do NOT run functional.
- `--all`: run Part 1 then security evaluation. Produce both reports.

For each evaluation that runs, read its `EVAL.md` and follow all phases.

---

## Final Output: Summary

After evaluation, present a summary. Adapt the box to the mode used.

### Functional only (default)

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

### Security only (`--security`)

```
╔══════════════════════════════════════════════════════════════╗
║     Accelerator Security Readiness Report: <backend>        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  SECURITY READINESS                                          ║
║  Readiness: XX%                                              ║
║                                                              ║
║    Multi-Tenant Isolation:      L1  XX%                      ║
║    Device Memory Encryption:    L1  XX%                      ║
║    Data Scrubbing:              L1  XX%                      ║
║    Host-Device Transit:         L2  XX%                      ║
║    Firmware & Driver:           L2  XX%                      ║
║    PyTorch Integration:         L3  XX%                      ║
║                                                              ║
║  Report:                                                     ║
║    torch-air-report/torch_security_readiness_report_<backend>.md   ║
╚══════════════════════════════════════════════════════════════╝
```

### Both (`--all`)

```
╔══════════════════════════════════════════════════════════════╗
║        Accelerator Readiness Report: <backend>              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PYTORCH INTEGRATION          Readiness: XX%                 ║
║  SECURITY                     Readiness: XX%                 ║
║                                                              ║
║  Reports:                                                    ║
║    torch-air-report/torch_readiness_report_<backend>.md      ║
║    torch-air-report/torch_security_readiness_report_<backend>.md   ║
╚══════════════════════════════════════════════════════════════╝
```

Report three time figures (when applicable), all measured from the Time Estimate step to report completion:

- **Estimated (active)** -- the earlier estimate. Active compute/probing time only.
- **Actual (wall-clock)** -- total elapsed time, including user-wait spans. Informational only.
- **Actual (active)** -- wall-clock minus user-wait spans.

Compare the estimate against **Actual (active)** only. Use the gap to calibrate later estimates within this session.

## Important Notes

- Every probe must be wrapped in try/except. One failure must not stop the evaluation.
- If the backend is only partially implemented, produce a partial report.
- For items that cannot be checked (e.g., "CI pipeline"), mark as "Requires manual verification".
- All output files go in `torch-air-report/` (git-ignored).
- Security evaluation is based on publicly available information. Reports require human security engineer review before external use.
- Distinguish "not documented" from "not implemented" in security Notes.
