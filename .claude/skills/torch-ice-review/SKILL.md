---
name: torch-ice-review
description: Review a torch-ice pull request for architectural alignment against checklist.md. Use when asked to review, audit, or validate a torch-ice PR (by number, URL, or branch) that adds or changes a skill, framework, or evaluation dimension under SKILL.md, skills/, frameworks/, .agents/skills/, or .claude/skills/. Produces a problems-only review and, optionally, posts it to the PR as a GitHub review.
---

# Torch-ICE Architecture Review

Review torch-ice PRs that touch `SKILL.md`, `skills/`, `frameworks/`,
`.agents/skills/`, or `.claude/skills/` for architectural alignment with Torch-ICE's conventions —
not accelerator backends themselves (that's `torch-integration-capability-evaluation`;
don't confuse the two).

## Usage Modes

### No Argument

If invoked with nothing, don't guess — ask:

> What would you like me to review?
> - A PR number or URL (e.g. `16`, or the full GitHub URL)
> - A local branch (diffed against `main`)

### PR Number or URL

```
gh pr view <pr> --json number,title,body,author,headRefName,baseRefName,files,additions,deletions,reviews
gh pr diff <pr>
base_repo=$(gh pr view <pr> --json url -q .url | sed -E 's#https://github.com/([^/]+/[^/]+)/pull/.*#\1#')
git fetch https://github.com/$base_repo.git pull/<number>/head:pr-<number>-head
```

Fetch by explicit HTTPS URL parsed from `gh pr view <pr>`'s own `url` field —
never from a separate `gh repo view` call, and never from a local remote
named `origin`/`upstream`. `gh repo view` resolves whatever repo is
"current" for the local clone's remote config, independent of which PR was
asked for; in a clone with `origin` pointing at a personal fork and
`upstream` pointing at the canonical repo, it can resolve to a *different*
repo than the one the PR argument actually points at, silently fetching an
unrelated PR with the same number from the wrong repo. Deriving `base_repo`
from the PR's own resolved `url` instead ties the fetch to the exact repo
`gh pr view` already used for metadata, so it can't diverge.

This fetch matters even for a dry run: most PRs come from forks, so the
changed files won't exist on disk without it, and it's what inline-comment
line numbers (see Posting, below) get anchored against.

Wrap all of this in error handling: if `gh` isn't authenticated, the PR
doesn't exist, or the repo can't be resolved, report the exact error and
stop — don't guess at PR content.

### Local Branch Mode

```
git branch --show-current
git diff --name-only main...HEAD
git diff main...HEAD
git log main..HEAD --oneline
```

Use the branch name in place of a PR number throughout; there's no `gh pr`
context (title/body/author), so build the Summary from commit messages and
the diff instead.

### GitHub Actions Review Agent

`@torch-ice-review-agent` is a separate, comment-triggered Actions agent. It
posts an advisory PR comment and does not run this skill or submit a formal
GitHub review. This skill remains manual; use `--post` when a formal review
is wanted. Any future Actions wrapper for this skill would be separate.

## Applicability

Compare the changed-files list against `SKILL.md`, `skills/**`,
`frameworks/**`, `.agents/skills/**`, and `.claude/skills/**`. If none match, say so and stop — don't
produce a review.

## Review Philosophy

1. **Only report problems.** The review's job is to say what's wrong and
   what to change. Never write "X is fine," never praise a correct decision,
   never mention a passing check. If a category has nothing wrong, omit it
   entirely.
2. **Investigate, don't guess.** Read the actual diff and the referenced
   files before deciding something violates (or satisfies) a checklist item.
   A guess that's wrong is worse than no comment.
3. **Get the classification right first.** The "Framework vs. Evaluation
   Dimension" call at the top of the checklist decides which items even
   apply — most findings follow from getting this right.
4. **Fail closed on uncertainty.** If a claim can't be verified from the
   diff (e.g. it requires running code you don't have access to), say so
   explicitly — "requires manual verification" — rather than treating it as
   passing.
5. **Be specific.** Every finding cites a file:line or a diff excerpt, and
   names the fix. "This should use flags instead" beats "structure could be
   improved."
6. **No repetition.** Each finding appears in exactly one place in the
   output, under the one category it best fits.

## Review Workflow

### Step 1: Gather context

Use the commands from the relevant Usage Mode above.

### Step 2: Classify

Determine PR type using this precedence — check top to bottom, stop at the
first match (a PR commonly matches more than one bullet, e.g. adding a
`frameworks/` entry *and* editing the dispatch table in the same diff; that
dispatch edit is still checked under Dispatch & Orchestration, it doesn't
make the PR `mixed`):

1. Diff adds or substantively changes content under `skills/` or
   `frameworks/` → apply the "Framework vs. Evaluation Dimension" test from
   the checklist → `new-framework` or `new-evaluation-dimension`. Wins even
   when the same diff also touches `SKILL.md`.
2. Else, diff touches only `SKILL.md` dispatch/routing/flag logic, no new or
   changed `skills/`/`frameworks/` content → `orchestrator-change`.
3. Else, diff touches only `README.md`/prose → `docs-only`. Only the
   General Conventions category applies (specifically README/structure
   sync); skip straight to Step 6.
4. Else, diff is scoped to `.agents/skills/torch-ice-review/**` or
   `.claude/skills/torch-ice-review/**`
   (the review skill's own files) with no changes to `skills/`,
   `frameworks/`, or root `SKILL.md` → `review-tooling-change`. Skill
   Structure, Framework Nesting, Scoring Consistency, and Dispatch &
   Orchestration are all N/A — they govern the accelerator-evaluation
   surface this skill reviews, not its own structure. Only General
   Conventions applies; skip straight to Step 6.
5. Else → `mixed` — genuinely orthogonal changes bundled together, not the
   routine dispatch-table edit a new framework/dimension is expected to
   include.

Skill Structure's "one skill for all accelerator evaluation" item applies
equally to `new-framework` and `new-evaluation-dimension` — neither ever
gets its own skill. What differs: Framework Nesting's "dimensions nest
under their parent framework" item is N/A for `new-framework` (a framework
legitimately gets its own top-level `frameworks/<name>/`); and Dispatch &
Orchestration's expectations invert — a framework *should* get a Framework
Dispatch table row (that's correct, not a violation), while a dimension
should never get one and must be routed by a flag instead.

### Step 3: Check against the checklist

Go through `checklist.md` category by category (Skill
Structure, Framework Nesting, Scoring Consistency, Dispatch & Orchestration,
General Conventions — skip categories Step 2 ruled out). For each item, read
the actual diff/files and decide: does this PR violate it?

- Violated → capture a candidate finding: `(file:line, one-line claim,
  category, checklist item)`.
- Satisfied, or the item doesn't apply to this diff → move on, record
  nothing. Passing items are never written up (see Review Philosophy).
- Can't tell from the diff alone → read the referenced file directly rather
  than assuming either way.

Cross-check every candidate against **Always Request Changes** in the
checklist — if it matches, flag it as such; this is what later decides the
Recommendation and the posted review's `event`.

### Step 4: Consolidate

Before drafting anything, collapse the candidate list:

- **Same root cause → one finding.** E.g. a standalone skill for a
  dimension typically trips the "one skill per lens," "flag syntax
  documented," and "no redirect" items at once — that's one finding
  ("shipped as a standalone skill instead of a flag") with three
  consequences, not three findings.
- **Same fix → one finding.** If one edit resolves several checklist items,
  it's one finding.
- **Same file:line twice → merge**, unless they're genuinely independent
  defects.

### Step 5: Fact-check

Re-read the cited file:line (or diff hunk) for each surviving finding and
confirm it still holds against the actual diff — not a stale assumption
from Step 3. Drop anything that doesn't hold up on a second look; reword
anything imprecise. Only fact-checked findings go in the output.

### Step 6: Write the review

Follow **Output Format** below. Write it fresh — this is not a filled copy
of the checklist file.

## Output Format

```markdown
## Architecture Review: PR #<number>
<!-- or, for local branch mode: -->
## Architecture Review: <branch-name> (vs main)

### Summary
What the PR does (1 sentence), then the overall verdict. If no issues were
found, say so explicitly here and stop — no further sections.

### Skill Structure
[Problems only — omit this section if none]

### Framework Nesting
[Problems only — omit this section if none]

### Scoring Consistency
[Problems only — omit this section if none]

### Dispatch & Orchestration
[Problems only — omit this section if none]

### General Conventions
[Problems only — omit this section if none]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Any finding matching "Always Request Changes" in the checklist forces
**Request Changes** regardless of everything else. Otherwise, judge
holistically — [1-2 sentence justification focused on what, if anything,
blocks approval].
```

Each finding: cite the file:line, state the checklist item it violates, and
name the fix — e.g. "`skills/torch-security-readiness/SKILL.md` (new file):
ships security evaluation as a standalone skill instead of a `--security`
flag on `torch-integration-capability-evaluation`. Fold the content into
`frameworks/pytorch/security/` and add the flag; delete the standalone
skill." One finding, one bullet, no repetition across categories.

## Posting to the PR (`--post`)

Without `--post`, the output above is the full deliverable — nothing is
sent to GitHub.

With `--post`, still show the rendered review and get explicit user
confirmation before calling GitHub — a posted review is visible to others
and isn't something a local revert undoes.

Determine an anchor line for each finding against the PR's actual head
content, not the local working tree (most PRs are from forks, and even a
local copy's line numbers may not match the PR's version):

```
git show pr-<number>-head:<path> | grep -n <pattern>
```

using the `pr-<number>-head` ref fetched in Step 1. If the fetch failed, or
no confident single line can be found, don't force a bad anchor — fold that
finding into the review body instead of an inline comment.

Submit one atomic review via stdin so the JSON isn't mangled by shell
quoting:

```
gh api repos/<owner>/<repo>/pulls/<number>/reviews --input - <<'EOF'
{
  "event": "REQUEST_CHANGES",
  "body": "<Summary + all category sections + Recommendation>",
  "comments": [
    {"path": "skills/<name>/SKILL.md", "line": 12, "body": "<finding text>"}
  ]
}
EOF
```

`event` mapping: Recommendation **Request Changes** → `REQUEST_CHANGES`.
Recommendation **Approve** or **Needs Discussion** → `COMMENT` — never post
`APPROVE` automatically; a merge approval is a human's call even when the
review found nothing wrong. State the Recommendation clearly in the body so
a human can act on it.

If the API rejects an inline comment (e.g. the line isn't part of the
diff), retry the review without that comment and fold its content into the
body — one bad anchor must never block the whole review from posting.

If the API rejects the whole review with "Can not request changes on your
own pull request" (422), the authenticated `gh` user is the PR's author —
GitHub only permits `COMMENT` from a PR's own author, never
`REQUEST_CHANGES`/`APPROVE`. Retry with `event: COMMENT`, keep the
Recommendation text ("Request Changes" / "Approve" / "Needs Discussion")
in the body unchanged so the verdict is still clear, and note in the body
that it was posted as `COMMENT` for this reason.

**Avoid duplicate reviews on re-invocation:** check `reviews` from Step 1
for a prior review from this agent. If the new findings are identical to
the last posted review, skip posting and tell the user nothing changed.

## Output Files

- `torch-ice-report/architecture_review_pr<number>.md` (or
  `..._<branch-name>.md` for branch mode) — the rendered review, git-ignored
- Printed summary to the user

## Important Notes

- This skill reviews Torch-ICE's own architecture — never run it against an
  accelerator backend's source code.
- Every `gh`/`git` call must be wrapped so one failure (rate limit, auth,
  missing PR) doesn't silently corrupt the rest of the review.
- Never post to GitHub without the explicit confirmation required above,
  regardless of flags passed.
