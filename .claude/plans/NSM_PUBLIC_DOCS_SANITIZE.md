# Plan: remove private-deployment details from the public repo

**Repo:** `gattia/nsm`. **Created:** 2026-09-23.

## State

**Updated:** 2026-09-23 · **Status:** open

- **Next:** Step 1, the inventory.
- **Blocked on:** nothing.
- **Done:** nothing yet.
- **Surprises:** none yet.

## Goal

NSM is public. Its docs, comments, tests and plans name the private application that
uses it in production (the one in this checkout's parent directory): its repo name, file
paths and line numbers, the production server, the job worker, job IDs and PIDs. NSM users
do not need any of that, and some of it should not be public.

When this is done, the public repo can say *"a downstream application"* where a
decision depends on one. It never names that application or describes its internals.

## What counts as private

Remove or make generic:

- the application's name, and paths or line numbers inside it
- the production server, worker, website, job IDs, PIDs, archive sizes
- facts that only matter for deploying that application (for example, "delete line N
  before pulling")

Keep, made generic:

- that a real caller depends on an API, when that is the reason for a decision. Write
  "a downstream caller builds `TriplanarDecoder` directly", not the file and line.
- numbers that describe NSM's behaviour, such as BScore tolerances, with the source
  described as "a production reconstruction".

## Where the private facts go

Anything still useful to the maintainer moves to the application's own repo, or to the
maintainer's notes. Nothing is lost; it just stops living here.

## Steps

1. **Inventory.** Search the tracked files for the application's name and the terms
   above. On 2026-09-23 that is 25 files: `docs/` (3), `CHANGELOG.md`, 2 files in `NSM/`,
   14 in `testing/`, and 5 plans. List each hit as *delete*, *make generic*, or *move*.
   Show the list to the maintainer before editing.
2. **Rewrite `docs/`, `CHANGELOG.md`, `NSM/` and `testing/`.** One PR. No behaviour
   change. Test names that mention the application are renamed.
3. **Plans.** Edit the active plans the same way. For the history file and
   `completed/`, the maintainer decides between editing them and leaving them.
4. **Guard.** Add a test that fails if the application's name appears in `docs/`,
   `CHANGELOG.md`, `NSM/` or `testing/`, and a line in `CLAUDE.md` § Writing.

## Decisions for the maintainer

- **Git history.** Old commits, PR descriptions and closed issues still contain these
  details. Rewriting history would break every clone and PR link. Proposed: leave history
  alone and clean only the current tree.
- **Plans directory.** `.claude/plans/` is public. Should it stay tracked, or become
  local-only?
- **Is the application's name itself sensitive,** or only its internals? If only the
  internals, Step 4's guard checks for paths and server details instead of the name.

## Size

Net-negative lines. Permanent additions: one guard test (~15 lines) and one `CLAUDE.md`
line.

## Verification

- The Step 1 search returns no hits outside the files the maintainer chose to keep.
- `make lint` and the full suite pass.
- The guard test fails when the name is added to a file in `docs/`.
