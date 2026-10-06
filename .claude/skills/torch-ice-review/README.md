# Architecture Review

Assessment PRs that touch `SKILL.md`, `skills/`, `frameworks/`, or
`.claude/skills/` should be
reviewed against `.claude/skills/torch-ice-review/checklist.md`
before merge. It's a reference checklist (skill structure, framework
nesting, scoring consistency, dispatch logic) — not a form to fill in.

Run it with the `torch-ice-review` skill:

```
/torch-ice-review <pr-number-or-url-or-branch>
```

This checks the PR's diff against the checklist and writes up whatever's
actually wrong as a fresh, problems-only review, organized by category, with
a final Recommendation (**Approve** / **Request Changes** / **Needs
Discussion**). Categories with nothing wrong are omitted — a clean PR gets a
short review, not a wall of passing checkmarks. By default it's a dry run —
nothing is posted to GitHub. Add `--post` to have it post the review to the
PR (inline comments per finding, verdict in the review body) after you
confirm the rendered output:

```
/torch-ice-review 16 --post
```

The separate `@torch-ice-review-agent` GitHub Actions agent is a
comment-triggered, read-only advisory flow. It posts a normal PR comment; it
does not run this skill or submit a formal review.

To review manually instead, read
`.claude/skills/torch-ice-review/checklist.md` and write up
findings the same way. See
[issue #17](https://github.com/TorchedHat/torch-ice/issues/17) for a
potential separate Actions wrapper for this skill.
