---
description: Pull main, read a plan's State block, do its Next in reviewable commits
argument-hint: [plan-name]
---

Continue an in-flight initiative. Plan: "$ARGUMENTS"

1. **Sync first**: fetch and update local `main` — PRs merged since the last session
   must be present. If the working tree is dirty, stop and report what is there
   before touching anything.
2. **Read the plan**: `.claude/plans/<plan-name>.md` § State. If no plan was named,
   list the open plans' Status and Next lines and ask which one to continue.
3. **Check State against reality before acting**: if Next assumes a PR merged or a
   branch exists, verify it (`gh pr view`, `git branch -a`). If State and reality
   disagree, correct the State block first and say so. If the Next's branch already
   exists with commits newer than State's **Updated** date, another session likely
   owns it — check `ListAgents` and coordinate before touching the tree.
4. **Do the single Next action** — not the rest of the plan. A non-trivial Next
   starts with its plan statement as the first commit (CLAUDE.md § Plans: claims
   run before they are written, verification per claim).
5. **Commit as you go, on a branch**: one commit per concern, so each can be
   reviewed, edited, or given feedback individually. Never one bulk commit at the
   end. `make lint` and the full suite green before each commit. Base the branch on
   `main` unless the State block says otherwise; if stacked, record "merge X first"
   in both the State and the PR body — the base only auto-retargets if the earlier
   branch is deleted at merge. Keep the plan's State block current
   (Done / Next / Surprises) as part of the work, so it can be handed off at any
   point; the State update rides in the same PR.
6. **Stop at the review boundary**: when the Next action is done, push the branch,
   open or update the PR, summarize what each commit contains, and stop — do not
   start the following item.

Parallel jobs each get their own `git worktree` — two sessions in one checkout
collide on the working tree between commits.
