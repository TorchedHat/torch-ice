# Torch-ICE Architecture Review Checklist

Reference checklist for reviewing torch-ice PRs that touch `SKILL.md`,
`skills/`, `frameworks/`, or `.claude/skills/` — used by the
`torch-ice-review` skill and by human reviewers. Skip items with
nothing in the diff to check them against.

**This file is reference material, not a fillable template.** The reviewer
(human or agent) reads it, checks the diff against each item, and writes up
whatever's actually wrong as a fresh review — see
`SKILL.md` for the output format. Passing
items are never written up; if nothing under a category is violated, that
category doesn't appear in the review at all.

## Framework vs. Evaluation Dimension

Get this right first — most items below depend on it. **Neither case ever
justifies a new skill** — `torch-integration-capability-evaluation` is the single entry
point for all accelerator evaluation, frameworks and dimensions alike. A new
skill is only for a genuinely different *capability*, entirely outside
accelerator evaluation (e.g. `torch-ice-review`, which reviews
Torch-ICE's own PRs, not accelerator backends).

- **New framework**: evaluates a fundamentally different integration
  target/runtime than anything Torch-ICE already covers (e.g. a wholly new
  ML framework, not a new lens on PyTorch accelerator integration). Gets its
  own top-level `frameworks/<name>/` and a row in the existing skill's
  Framework Dispatch table.
- **Evaluation dimension**: adds a scoring lens (security, performance,
  compliance, ...) on top of a target Torch-ICE already evaluates. Must
  nest under the existing framework (`frameworks/<framework>/<dimension>/`)
  and be reached via a flag on the existing skill — never a dispatch-table
  row of its own.
- Still unclear after that? Default to **dimension** — the safer failure
  mode is asking the author to fold it in and nest it, not letting parallel
  framework trees multiply.

## Skill Structure

- [ ] **One skill for all accelerator evaluation** — a new evaluation
  dimension (security, performance, compliance, ...) extends
  `torch-integration-capability-evaluation` via a flag (`--security`); a new framework
  extends it via a Framework Dispatch table entry. Neither ever gets its
  own skill or command name — a standalone skill is only for a genuinely
  different capability outside accelerator evaluation entirely (see
  "Framework vs. Evaluation Dimension" above)
- [ ] **Flag syntax documented** — `SKILL.md` shows invocation examples for
  the default run, the new flag alone, and the combined (`--all`) run
- [ ] **Default invocation unchanged** — running the skill with no flags
  still produces only the base functional evaluation; a new dimension is
  never folded into the default
- [ ] **Plugin-packaging symlink intact** —
  `skills/torch-integration-capability-evaluation/SKILL.md` is a symlink to `../../SKILL.md`
  (for plugin discovery), not a second copy — they can never structurally
  diverge as long as the symlink exists. Flag a diff that replaces the
  symlink with a real file: that silently turns it into a second copy that
  *can* drift, with nothing enforcing they stay in sync afterward
- [ ] **No redirect to a separate skill** — description/body text should
  point users at a flag on the existing skill, never "use `<other-skill>`
  instead" for something that should be a flag

## Framework Nesting

```
frameworks/pytorch/
├── checklist.md
├── EVAL.md
└── security/
    ├── checklist.md
    └── EVAL.md
```

- [ ] **Dimensions nest under their parent framework** — path is
  `frameworks/<framework>/<dimension>/` with its own `checklist.md` +
  `EVAL.md`; never a peer `frameworks/<dimension>/`
- [ ] **Framework-specific content stays scoped** — content referencing a
  specific framework's internals (PrivateUse1, `torch.compile`, autograd,
  DeviceGuard, ...) lives under that framework's directory, not elsewhere
- [ ] **New frameworks are fully wired** — a genuinely new framework needs
  `checklist.md`, `EVAL.md`, and a Framework Dispatch table entry all
  together, or it's only half-usable
- [ ] **Orchestrator output stays framework-generic** — adding a framework
  shouldn't require hardcoding its name or section list into a shared
  output template. `SKILL.md`'s "Final Output: Summary" box is currently
  hardcoded to PyTorch (e.g. the literal line `PYTORCH INTEGRATION` and
  PyTorch's own section names) because only one framework exists today; the
  PR that adds framework #2 is the one that must generalize this template
  to render per-framework, not bolt on a second hardcoded block next to the
  first
- [ ] **README's repo tree matches the actual layout** — update it in the
  same PR as any structural change under `frameworks/`/`skills/`
- [ ] **Checklist items have a matching `EVAL.md` phase** — an item with
  nothing telling the evaluator to check it will never get filled with real
  evidence
- [ ] **No duplicate/overlapping items across sections** — a new
  checklist's items shouldn't restate ground already covered by another
  section (e.g. a security section re-checking something the framework's
  own checklist already scores) — merge or cross-reference instead

## Scoring Consistency

**Reference scoring model** (`frameworks/pytorch/checklist.md`):

```
Row weight:     w_i = 1 / priority_i
Section %:      sum(score_i * w_i) / sum(2 * w_i) * 100   (N/A rows excluded)
Tier weight:    weight_r = 1 / level
Overall %:      sum(section_pct * weight_r) / sum(weight_r) * 100
```

- [ ] **Numeric row scoring** — rows score `0`/`1`/`2`/`N/A`, not
  qualitative labels like PASS/PARTIAL/FAIL or READY/CONDITIONAL/NOT READY,
  so results can be weighted and compared across frameworks
- [ ] **Weighted overall percentage** — overall readiness is a computed
  weighted percentage, not a qualitative-only verdict
- [ ] **Priority pre-filled per row** — every row's Priority (1=critical,
  2=important, 3=nice-to-have) is fixed in the template, not decided per
  evaluation run
- [ ] **Levels and tier weighting documented** — sections have Levels (1-3)
  with `weight = 1/level` stated
- [ ] **Formula matches the reference model** — reuses the formulas above
  unless a divergence is explicitly justified in the PR description
- [ ] **Standard summary block present** — Executive Summary + Section
  Scores + overall % at the top, before the detailed rows
- [ ] **One scoring system, not two in parallel** — domain-level results
  roll up into section %, which rolls up into overall % — not a separate,
  unrelated numeric or qualitative score running alongside

## Dispatch & Orchestration

- [ ] **Dispatch table lists frameworks only** — never a row for an
  evaluation dimension; everything in this table runs on every evaluation
  automatically
- [ ] **Dimensions routed by flags** — a dimension not in the dispatch table
  needs a corresponding flag in `SKILL.md`, or it's unreachable
- [ ] **No "follow all phases" without flag scoping** — orchestrator
  instructions must explicitly scope which phases run for which active
  flags; this is an instruction to an LLM, not compiled code, so unscoped
  language gives the model no signal to skip anything
- [ ] **Combined evaluation needs an explicit flag** — running multiple
  dimensions together requires a documented `--all`-style flag, not implicit
  default behavior
- [ ] **Dimension `EVAL.md` loads only when its flag is active** —
  unconditional loading wastes context and risks the model following it
  without the flag
- [ ] **Output naming follows convention** —
  `torch-ice-report/<type>_report_<backend>.md`
- [ ] **Dimension phases don't pollute the base framework's `EVAL.md`** —
  keep them in the dimension's own nested `EVAL.md`

## General Conventions

- [ ] **Reports target `torch-ice-report/`** — the git-ignored output
  directory
- [ ] **README updated when the assessment surface changes** — a new skill
  or dimension invisible in the README effectively doesn't exist to users
- [ ] **`EVAL.md` strips internal instructions from output** — a ground rule
  that scoring rubrics and phase instructions never leak into the generated
  report
- [ ] **Probes are failure-isolated** — a new `EVAL.md` instructs wrapping
  each probe so one failure doesn't abort the whole evaluation, matching
  `SKILL.md`'s existing "every probe must be wrapped in try/except" rule for
  the base framework
- [ ] **Partial implementations still produce a partial report** — a new
  `EVAL.md` doesn't bail out entirely when the backend/target is only
  partially implemented; it should say so and still fill in what it can,
  matching the base framework's convention
- [ ] **Unverifiable items are marked, not skipped** — a new checklist/EVAL
  pair handles items it can't check (e.g. "CI pipeline") by marking them
  "Requires manual verification," not omitting them or guessing a result
- [ ] **`plugin.json`'s skills path unchanged** (`./skills/`) — this is the
  one field the plugin loader uses to discover every skill in the repo
- [ ] **Checklist refinements preserve table structure** — content fixes and
  structural changes (renumbering, column layout) should be separate,
  clearly labeled edits
- [ ] **Evidence is concrete, not asserted** — filled rows in any scored
  checklist cite a file path + line, a command's output, or a URL — never a
  bare claim like "supported"
- [ ] **Reports record generation metadata** — Torch-ICE version, framework
  version (e.g. PyTorch version), and the model used to generate the report
  are captured, so a report is reproducible and its provenance is clear
- [ ] **No collateral removal of existing documentation** — an edit to
  README/SKILL.md/checklist prose doesn't silently drop an existing,
  still-accurate line (e.g. a scoring rule, a caveat) as a side effect of
  editing nearby text; removals need to be deliberate and called out
- [ ] **Backwards-compatible report semantics for existing consumers** — a
  report format already consumed by external vendors or downstream tooling
  keeps its existing meaning; a new dimension or feature doesn't repurpose
  or redefine what an existing report already means
- [ ] **Extension scaffolding in `SKILL.md` stays intact** — instructions
  that document how to add a new framework or dimension aren't removed or
  trimmed as part of an unrelated edit; they're what makes the pattern
  self-documenting for the next contributor
- [ ] **No unrelated changes to plugin manifests** — edits to
  `.claude-plugin/marketplace.json` or `.claude-plugin/plugin.json` are
  flagged unless the PR's actual change requires them (e.g. a new plugin
  entry, a changed skills path)

---

## Always Request Changes

Some failures are structural enough that nothing else about the PR
outweighs them:

For the manual skill, these require a **Request Changes** recommendation.
When the checklist is supplied to `@torch-ice-review-agent`, it reports them
as advisory findings only; the agent never submits a formal GitHub review.

- A new evaluation dimension shipped as its own standalone skill instead of
  a flag
- A new framework shipped as its own standalone skill instead of a
  Framework Dispatch table entry on the existing skill
- A dimension nested as a peer `frameworks/<dimension>/` instead of under
  its parent framework
- A new checklist using qualitative-only scoring instead of the numeric
  weighted model
- A dimension listed in the Framework Dispatch table, or reachable without
  an explicit flag

## Example Anti-Pattern Combinations

Illustrative only — not exhaustive, and not a substitute for checking each
item on its own evidence. Real PRs commonly trip several related items
together because they share one root cause; these exist so a reviewer can
sanity-check that related items are being read consistently, not as a fixed
list of "the" patterns to look for.

| Anti-pattern | Typically also violates |
|--------------|-------|
| A dimension shipped as its own standalone skill instead of a flag | Flag syntax documented, No redirect to a separate skill |
| A new framework shipped as its own standalone skill instead of a Framework Dispatch table entry | One skill for all accelerator evaluation |
| A dimension's directory created as a `frameworks/<name>/` peer instead of nested under its framework | Framework-specific content stays scoped |
| Qualitative-only scoring (e.g. PASS/PARTIAL/FAIL, READY/CONDITIONAL/NOT READY) instead of the numeric model | Weighted overall percentage |
| A dimension added as its own dispatch-table row with unconditional "follow all phases" | Dimensions routed by flags, No "follow all phases" without flag scoping |
