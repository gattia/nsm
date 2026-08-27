# Plan: NSM code-health audit and refactor

**Repo:** `gattia/nsm` (NSM). **Created:** 2026-08-14.

## State

**Updated:** 2026-08-27 · **Status:** open

- **Next:** execute **§8.0.K** — `reconstruct_latent` internals: #75, the 185-line nested
  `compute_loss`, the hybrid Adam/LBFGS branch. Its statement is commit 1 of that slice
  and is not written yet. Nothing blocks it; §8.0.J stopped at the
  `reconstruct_latent(**reconstruct_inputs)` call and left `latent_fit.py` untouched.
  **Read §8.0.J's size finding before writing the statement** — K opens a 744-line module
  with the same shape, and the cost of a keyword-only extraction is two lines per
  parameter, not one.
- **§8.0.J executed (2026-08-27), PR open:** commits 2–9 whole — the characterization, the
  keyword refusal, `register_similarity`, the reference mesh, the ungating plus dead code,
  the timing recorder, the extraction, and this update. Suite 864 → 884 passed / 1 skipped
  / 3 xfailed at both ends: **all 12 strict xfails commit 2 raised were retired inside the
  slice**, and the 3 that remain are the regression harness's. One § History entry (20,
  the swallowed keyword), one CHANGELOG Breaking entry and four Changed, `SCOPE` §3.1's
  coverage claim corrected, `ARCHITECTURE` §5's table row and §7's accepted-and-ignored
  row updated, `KNOWN_ISSUES` § Open's F401 count recomputed (43 → 44, having been 54).
  `reconstruct_mesh`'s executable body is 322 → 189 lines.
- **§8.0.J, review round 1 (maintainer, 2026-08-27): the keyword refusal was trying to be
  helpful and that was bloat.** The first version ran `difflib.get_close_matches` over the
  live signature and appended "did you mean ...?" to every unknown key — 18 lines and two
  imports. The statement had defended it as "part of the fix, not a courtesy", and that was
  overreach: the signal that was missing is the *refusal*, and a caller holding a
  `TypeError` that quotes their own typo does not need the function to guess for them. The
  general point, since this is the second slice to add prose or machinery answering a
  question nobody asked (§8.0.I round 2's connectivity docstring was the first): **"the
  error could be more helpful" is not evidence that it should be.** Applied by rewriting
  commits 1, 2, 3, 8 and 9 in place and force-pushing, per `CLAUDE.md`; the branch is
  otherwise unchanged, verified by diffing the replay against the pre-review branch.
- **The slice's own headline was wrong, and answering why settled the State block's open
  question.** The row says "the 61-parameter signature"; it is 58 named plus `**kwargs`.
  More usefully, the signature is not §8.0.J's to shrink at all: `reconstruct_mesh` is
  frozen public API and kneepipeline calls it with 27 keyword arguments, which is the same
  release-boundary constraint §8.0.I cited for `create_mesh` (17) and `create_mesh_adaptive`
  (26). **All three shrink together at §8.0.O or none of them does** — taking a third of a
  set early is how a release boundary gets crossed twice. What was left for §8.0.J was the
  internals, and the hole that made 58 parameters dangerous *now*: a `**kwargs` that
  accepted every misspelling of them.
- **The size budget was missed for the third slice running, and this time the mechanism is
  nameable.** Budget +75 net in `NSM/`, ceiling +95, actual **+120** (it was +135 before
  review round 1 cut the suggestion machinery). It priced three helper
  signatures and their docstrings and **did not price the call sites**: a keyword-only
  boundary writes every parameter name twice, and three helpers needing 38 parameters
  between them is 76 lines that are nothing but names. That cost is not avoidable by
  writing it better — keyword-only is §8.0.I review round 2's requirement — so **the rule
  for §8.0.K and §8.0.L is to budget an extraction at two lines per parameter the stage
  needs, plus the body.** §8.0.H's miss was transitional code priced at zero and §8.0.I's
  was refusals priced by net lines; all three are the same error of budgeting the part you
  are thinking about and not the part the language makes you write.
- **The deletion pass deleted two of the five planned helpers, on a criterion worth
  keeping: does the call site state something the inline form cannot?** `_decode_meshes`
  failed it — its call site repeats what the loop says, for 35 lines — and so did the
  argument coercion, which reads `reconstruct_mesh`'s own parameters and rebinds them, so
  a helper handing five values back for the caller to rebind is plumbing, and a 5-tuple
  unpack of two numbers and three sequences is the positional coupling this slice removes.
  The three that survived each pass it: `_build_reference_mesh` lets the call site say
  `... if register_to_mean else None`, which is the fix, where inline the same fact is a
  45-line `if/else`; `_sample_subject` puts one name on a two-reader branch; and
  `_assemble_result` was moved *further* than planned, taking the return-type switch with
  it, because a six-flag `if` at the call site and a six-flag body 40 lines away is
  `time_calc_recon_loss`'s defect in a second location.
- **Two defects were the same defect in different places, which is what made them findable.**
  `time_calc_recon_loss` was measured on every scored call and returned by nothing, because
  recording a stage and returning it were two unrelated edits 90 lines apart. The
  return-type switch had the identical shape. `_StageTimings` is a `dict` rather than a
  wrapper around one precisely so that the second edit stops existing: `return_timing` is
  `result.update(timings)`.
- **The characterization's source scan would have gone green by measuring nothing.** It
  matched `time_x = toc - tic` to derive the stages the body measures — and commit 7
  deleted every instance of that spelling. Left alone it would have passed against an empty
  set. It now matches both spellings and asserts it matched something. **Any test that
  derives its expectation from the source is one refactor away from asserting a tautology;
  the fix is an assertion that the derivation found anything at all.**
- **§8.0.G's conversion left a residue in every file it touched, and this slice cleared one.**
  Ten of `reconstruct_mesh`'s fifteen log records sat under `if verbose is True:` — faithful
  to the `print` gates they replaced, and the reason a host that ran the exact replacement
  the deprecation notice names saw none of them. Ungated here; **~180 sites of the same
  shape remain**, one file at a time, with whichever slice opens each. §8.0.N is where the
  last of them should be checked for.
- **Measured and *not* a defect, recorded so it is not re-investigated** (and asked about
  in review): the single- and multi-object readers were passed
  `mean_mesh if register_similarity else None` and `mean_mesh` respectively. Inert, two
  ways. By reading, `mean_mesh` has exactly one reader in either sampler
  (`mesh_sampling.py:141`, `:552`) and both are inside `if register_to_mean_first is True:`.
  By running, a **decoy** mean mesh — 3× the radius, shifted by (5, 5, 5), so any
  registration against it would be unmissable — gives byte-identical points, SDF,
  `icp_transform`, `scale` and `center` against `None` on both readers with
  `register_to_mean_first=False`. The `else None` was defensive habit and the file said so
  already: the multi-object call passed `mean_mesh` plain and always had, so a load-bearing
  guard would have left that path broken all along. After the reference-mesh commit the
  question closes outright — `mean_mesh` *is* `None` when `register_to_mean` is false, so
  the conditional is unreachable, not merely inert.
- **Still positional and pre-existing, out of this slice** (unchanged from §8.0.I's list):
  `interpolate._advance` (9), `interpolate._tangent_laplacian_step` (8),
  `interpolate.interpolate_common` (8), `correspondence_metrics._tri_tri_intersect` (6).
- **§8.0.I merged to `main` in PR #91 (2026-08-27):** both review rounds applied as
  commits on top, at the maintainer's instruction; #54, #57 and #60 closed by the merge.
- **§8.0.I executed (2026-08-26), PR #91:** commits 2–9 whole — the
  characterization, #57's accessor, #60 in two commits, #54 in two, the shared
  tail, and this update. Suite 812 → 857 passed / 1 skipped, 3 xfailed at both
  ends: **all 19 strict xfails commit 2 raised were retired inside the slice**.
  Three § History entries (17 #57, 18 #54's `score_correspondence`, 19 #60's
  fallback grid), five CHANGELOG Breaking entries, SCOPE §2.3's three conditions
  all executed and §2.6 given new input, ARCHITECTURE §2.1/§3/§6/§7 corrected.
  `mesh/main.py` is net −5 lines with the deduplication in it.
- **§8.0.I, review round 1 (maintainer, 2026-08-27): the accessor is `get_faces`,
  and `refine_mesh` imports it rather than forwarding.** The slice left
  `refine_mesh.py` calling *two* names for one operation — `get_faces` at three
  sites and the new `triangle_faces` at the connectivity warning — and named the
  accessor against the convention of the file it moved into
  (`get_triangle_area`, `get_edge_lengths`). The maintainer's move settles both
  in one edit and beats the two options that had been offered (a marked
  back-compat alias, or a Breaking deletion deferred to §8.0.O): a named import
  binds `get_faces` in `refine_mesh`'s namespace, so
  `NSM.mesh.refine_mesh.get_faces` resolves to the *same object* — verified, with
  `__module__` reporting `NSM.mesh.triangle_metrics` — and there is no second
  docstring or wrapper to keep in step. **The general point: "preserve the public
  path" and "have one definition" are not in tension when the path can be a
  re-export.** Reaching for an alias or a deprecation was solving a problem the
  import system does not have. (This is unrelated to ARCHITECTURE §5's re-export
  trap, which is about *star* imports with no `__all__`.) Also renamed here:
  `_volume_and_origin` → `_prepare_sdf_grid`, because the name described the
  return tuple rather than the job, and "volume" is `crop_sdf_to_narrow_band`'s
  local word — `main.py` says `grid` 68 times and `volume` 11, 8 of them inside
  that one function. **Landed as a commit on top at the maintainer's instruction,
  not as a rewrite of commits 3/4/6.**
- **The re-export exposed a false negative in `test_docs_references`, older than
  this slice.** `_qualnames` collected only `def`/`class` nodes, so a name a module
  *re-exports* was invisible to it and any doc citing `refine_mesh.get_faces`
  would have failed — as would `deep_sdf.Sine`, which §8.0.H created and nothing
  has cited since. Fixed by registering NSM-internal `from ... import ...` names.
  **The narrowing to NSM-internal is the part worth remembering:** the first
  version registered every `from x import y`, which put `nn` into `TOP_LEVEL` and
  made the docs' `nn.Sequential` / `nn.Embedding` / `nn.ModuleList` citations look
  like NSM symbols — three failures, caught by running it. A checker that decides
  *whether a reference is ours* from the same index it uses to *resolve* it will
  fail that way every time the index widens.
- **§8.0.I, review round 2 (maintainer, 2026-08-27): the slice reintroduced the
  defect it had just fixed, one commit later.** Commit 5 turned
  `create_mesh_adaptive`'s 17 positional arguments into keywords because
  positional forwarding across a call boundary *is* #60 — ARCHITECTURE §7's named
  example of the LR bug's shape. Commit 8 then wrote a **14-positional** call to
  the new `_finish_meshes`, twice, in the same file. `_finish_meshes` and
  `_prepare_sdf_grid` are now **keyword-only** (`*`), so the list cannot be
  supplied in an order at all; every call site names its arguments. Bitwise
  output unchanged against the same pre-refactor baseline. **The lesson is about
  the fix, not the miss:** a fix applied at the *call sites* leaves the next
  author free to re-create the defect, and a fix applied at the *signature*
  does not. CLAUDE.md's "fix the class of defect" has a stronger reading than the
  one taken here — enumerate the sites, then move the guard to where new sites
  are born. **Still positional and pre-existing, out of this slice:**
  `interpolate._advance` (9), `interpolate._tangent_laplacian_step` (8),
  `interpolate.interpolate_common` (8), `correspondence_metrics._tri_tri_intersect`
  (6). Same shape; they belong with whichever slice next opens those files.
- **§8.0.I, review round 2, second item: the connectivity warning's docs answered
  the wrong question.** The maintainer read `_warn_if_connectivity_differs` and
  asked why the two meshes have to match when the point of the module is that
  `mesh` is a *changed* `base_mesh`. The check was right — it compares face arrays,
  and a warp moves every vertex while touching no face — but the module docstring
  said only "must share connectivity and cell ordering" and never said the
  geometry is *supposed* to differ arbitrarily. A doc that is technically accurate
  and does not preempt the first question a reader has is not doing its job.
  Verified and now pinned: the documented warp pass is silent, the iterative
  warp→subdivide→re-warp loop is silent, and reusing a **stale** `mesh` against a
  refined base warns — which is the mistake an iterative caller actually makes,
  and which "succeeded" at 624 → 1110 cells with the wrong triangles.
- **§8.0.I diverged from its slice-index row before any code, and the statement
  says so.** The row reads "`mesh/main.py`". #57 has **zero** sites there — its
  five are in `correspondence_metrics` (2), `interpolate` (2) and `refine_mesh`
  (1) — and #54's two mesh-side sites are also outside it. Re-running the greps
  is what caught it. The general lesson: a slice index row names the *issue* set
  reliably and the *file* only as a guess, because the index was written from
  issue titles.
- **Two of §8.0.I's own claims came back different from the issue text.**
  (1) #57 says a quad mesh "raises a bare reshape `ValueError`". It raises for 3
  quads and **silently fabricates five triangles** for 4 — the reshape succeeds
  exactly when the flat length divides, which is a fact about the cell count mod
  4. That moved #57 from a hygiene fix to a § History entry. (2) #60's differing
  `narrow_band` defaults are **behaviourally inert** — 6.2e-08 (skimage) /
  7.5e-08 (VTK) on a 32³ sphere against a 0.065 voxel — so aligning them needed
  no History entry, where the issue's framing implied one. Both were settled by a
  ten-line script before the statement was written.
- **The size budget was wrong again, the same way §8.0.H's was, and the statement
  is the thing that keeps being wrong.** `NSM/` is **+140** net against a stated
  ceiling of +30. Computed split: **+95 docstrings, +29 `raise`/`warn` message
  text, +16 everything else** — so the *logic* is +16 and the budget was measuring
  the wrong quantity. Two of the three biggest additions were mandated by the
  statement itself: SCOPE §2.3 condition 3 is a 43-line module docstring for
  `refine_mesh`, and condition 2 is a 15-line warning. **The rule to carry into
  §8.0.J:** a slice that adds documentation or refusals by name has to budget them
  by name; a single net-lines ceiling will be missed every time. §8.0.H recorded
  this once and it was not enough to prevent it.
- **A second, narrower lesson: extraction pays the signature twice.**
  `_finish_meshes` first landed at 72 lines and made `mesh/main.py` **+25** for
  removing two copies of a 45-line tail — a 14-parameter signature at the
  definition and again at each of two call sites, plus an `Args` block restating
  `create_mesh`'s own parameters. Cutting the docstring to what the reader cannot
  look up took it to −5. §8.0.J and §8.0.K extract from 61- and 26-parameter
  functions; expect the same, and check the budget *after* the docstring.
- **`create_mesh_adaptive` is still 223 lines** (from 243), and that is deliberate
  — the statement scoped only the shared tail. What remains is a ~55-line
  docstring plus three sequential passes (coarse → fallback → dense). Its
  26-parameter signature is §8.0.O.
- **§8.0.H merged to `main` in PR #90 (2026-08-26):** no review comments to apply.
- **§8.0.H executed (2026-08-26), PR merged:** commits 2–11 whole — the option
  matrix, #46 in three commits, #45, #26, the #20 sweep, one `Sine`, #34, and
  this update. Suite 787 passed (from 704) / 1 skipped / 3 xfailed (from 5); the
  12 strict xfails commit 2 raised were all retired inside the slice. **Verified
  end to end against the real shipped models** (not in CI — they are not in the
  repo): both `model_params_config.json` files are refused for omitting
  `padding`, and with `"padding": 0.1` added both load through `load_model` on
  CPU and forward (647: 20,801,924 parameters, output width 2; 551: 20,801,410,
  width 1). That same run settles `SCOPE.md` §2.6's open question — the consumer
  *could* switch to `load_model`, once those two files state `padding`.
  Three § History entries (14 `layer_split`/progressive depth, 15 #45, 16 #26),
  seven CHANGELOG Breaking entries, and `NSM_TRAINING_IDEAS.md` Idea 5 unblocked.
- **§8.0.G merged to `main` in PR #89 (2026-08-26):** #1 closed by the merge,
  no review comments to apply. Commits 2–8 whole — characterization,
  the `basicConfig` deletion, the `verbose` bridge, five per-subpackage
  conversions, verify-and-close #1, this update. Suite 678→704 passed; 5 xfailed
  both ends (the two new strict xfails were raised and retired inside the slice).
  Measured on the branch: **1** `print` survives in `NSM/` outside
  `train/deprecated/` — `configs/generate_sdf_default_config.py`'s, under its
  `__main__` guard — against 257 before; 277 `logger` calls (204 debug, 34 info,
  39 warning) across 15 module-scope loggers. Three structural pins in
  `testing/NSM/test_observability.py` hold all of that: no `print`, no log call
  that builds its first argument, and a `getLogger(__name__)` in every module
  that speaks.
- **§8.0.H, review round 3 (maintainer, 2026-08-26): the VAE activation is now
  *repairable*, not just documented.** The maintainer's objection was that the slice
  left the library unable to do what it was written to do, and it was right.
  `conv_activation` lands, `None` by default — byte-identical module list, every
  existing checkpoint loads, verified against 647 and 551 — and `load_model` requires
  the config to state which architecture it means, the third instance of that pattern
  after `padding` and `conv_norm_type`. **Not a version number, deliberately:** the
  maintainer proposed one, and a `version: 1|2` bundles this change with whatever the
  next architecture change is, needs a lookup table to read, and asserts that v2 is
  better — which Idea 13 says nobody has measured. A named field says what differs and
  lets the retrain pick the value. Placement (`conv → norm → activation`) is pinned by
  a test *as provisional*, so changing it is a decision rather than a diff. **Three
  required keys now land in one release**, all needing the same one-line edit to the
  same files; a combined "this config predates v0.3.0, here are the lines to add"
  message is worth considering at the release, and is not in this slice.
- **§8.0.H, review round 2 (maintainer, 2026-08-26): `conv_norm_type` had four
  defaults and they disagreed.** Asking what the ShapeMedKnee config actually says
  turned up the divergence: `"batch"` in the `VAEDecoder`/`TriplanarDecoder`
  signatures, `_get_triplanar_params` and the triplanar template; `"layer"` in the
  two_stage branch, `two_stage`'s defaults and `default_config.json`. **The value
  three of them chose is the one nothing has ever trained** — 647, 551 and
  `ShapeMedKnee_2024_config.json` all say `"layer"`, and the shipped default only
  agrees because `651a810` regenerated it from the 647 run; before that the key was
  absent. Same remedy as #26: `load_model` requires it, templates say `"layer"`,
  signatures untouched (breaking for a public-stable class → release slice). Not the
  same *defect* as #26, checked rather than assumed: a mismatch against a checkpoint
  is loud, because `BatchNorm2d` and `LayerNorm` differ in key set and shape. What the
  silent default cost was a **fresh** run from the template. Also settled here: the
  activation was never wired in — `71df387`, Aug 2023, the first triplanar commit,
  appends conv and norm and builds the activation without appending it — so no
  triplanar model NSM has produced has ever had one, and this is not a regression.
- **§8.0.H, review round 1 (maintainer, 2026-08-26): the `norm_layers` refusal was
  too wide, and the framing was wrong.** The maintainer recognised the option as
  something they had deliberately set up, which sent us to the history: commit
  `01d774a` (Jun 2023) introduced the branch with the message *"separate wieght
  norm and batch norm so can use both"* — and made it an `elif`, which is exactly
  what prevents using both. So it was never dead weight; it was a feature the code
  did not deliver, reachable only with `weight_norm=False`. Checked against a real
  `361_nsm_femur_cartilage` training config the maintainer produced: it carries
  `layers_with_norm: [0..7]` with `weight_norm: true`, so the defect never touched
  it — and the first implementation **refused it anyway**. Now the two cases differ
  (warn when provably inert, raise only where LayerNorm was really built), and that
  config builds bitwise-identically before and after. Two lessons, both general:
  a guard on "the key is set" is not the same as a guard on "the key did
  something"; and `git log` on the line, not just the line, is what tells you
  whether you are deleting dead code or someone's intent. Delivering "use both" is
  new capability and is filed for §8.1.
- **§8.0.H diverged from its statement in four places.** (1) **Two defects the
  audit had never recorded**, both found by re-running the claims:
  `layer_split: false` — the value `default_config.json` and *both shipped model
  configs* carry — is tested with `is not None`, so it meant *split at layer 0*
  and moved every state-dict key; and `TwoStageDecoder` mutated its module-level
  default dicts **before** raising, so even a failed construction changed the
  module process-wide. (2) **The size budget was wrong and the reason is
  structural.** NSM/ is **+31** net against a stated +10 — but `raise` blocks in
  `models/` went 30 → 78 lines (+48) and comments +32, so the *logic* is **−49**.
  The statement costed "roughly +25 lines of refusal" without counting that the
  slice contains **eight** separate refusals; a slice whose whole thesis is
  "works or refuses at construction" cannot be net-negative on message text.
  (3) **#34 needed no xfail** — the assertion it wants already held, it was
  simply never asserted, so the characterization commit had nothing to record and
  the measurement landed with the fix. (4) **`normalize_coordinates`' `padding`
  was still there although #20 is closed** — the issue closed on the enumeration,
  not on the deletions; the `models/` instances were carried only by
  `KNOWN_ISSUES` § Open, which is why "closed" was not the same as "gone".
- **Two maintainer calls ride alongside and gate nothing in §8:** (1) **file the
  drafted Mesh-subject issue** — approved text in PR #85's body, tracker still
  has no such issue (checked 2026-08-26); (2) the v0.3.0 timing ("soonish, or at
  the end of this cleanup") — latest tag is still v0.2.0, nothing gates the cut,
  and §8.0.F's ship-together constraint is satisfied inside `main` now that #19
  and #27 are both merged. Slices G–N do not touch a release boundary; §8.0.O,
  §8.0.P (the quarantine move changes import paths) and §8.0.Q do.
- **Scope re-drawn 2026-08-26 (maintainer).** Research and upgrades wait until the
  refactor closes. The Idea 4 coupled decision left this plan's **Next** and lives
  only in `NSM_TRAINING_IDEAS.md`, where it already was; §8.1 and §8.2 are marked
  deferred in place and indexed in the new **§8.3**. What triggered the re-draw:
  §8's four checkboxes read three-quarters done while every one of the repo's
  largest functions was a function the slices had *moved* but never opened, and
  the whole `models/` package — half the consumer's public contract — had had no
  pass of any kind. §8.0 now carries a slice index (G–Q) so that stops being
  invisible.
- **PR #85 merged to `main` (2026-08-25):** #19 and #27 closed by the merge,
  no review comments to apply, branch deleted. That consumed this plan's
  "review of #85" Next; the research-queue item behind it —
  `NSM_TRAINING_IDEAS.md` Idea 4(a), the no-retrain latent-norm diagnostic —
  was executed the same day (results and provenance in that file's Idea 4
  entry; summary: training-shell saturation confirmed on the one shipped
  latent-code file, fitted production norms median ~7.3 against a bound of 10,
  and the error–norm correlation has opposite signs within vs across
  subjects).
- **§8.0.F executed (2026-08-24), PR #85 open:** commits 2–7 whole, per the
  recalibrated /next (#83) — characterization, the dead-backfill fix, the #19
  key rewrite, the index decoupling, #27, the shell split, this update. Suite
  664→678 passed; 12→5 xfailed (every #19/#27 xfail passes unmarked; two new
  Mesh-subject pins). All 16 verification-table rows green. #19's three Open
  entries → History §13; #27's Open entry retired with no History entry
  (aliases shared one storage — disk, not accuracy); CHANGELOG carries both
  Breaking entries; SCOPE §4's cache row is versioned. **Diverged from the
  statement:** the single class's pos_idx backfill condition was DEAD
  (`unpack_numpy_data` always sets the key; pre-index-layout caches crashed
  `__getitem__` with IndexError) — fixed as its own commit, no History entry
  (always a crash); a Mesh subject has never built end to end in either class
  (silently dropped via the readers' `os.path.exists` gate, or `KeyError` in
  `mesh_content_key` when seeded) — pinned strict-xfail
  (`TestMeshSubjects`), issue drafted in the PR body, and `_identity` already
  routes a Mesh through the geometry digest so a later build fix cannot
  resurrect the per-object key; docs retirement split across the two #19
  commits so each commit's docs stay true; the multiprocessing determinism
  test now writes its meshes once in the parent (rewriting per subprocess
  moved the mtime-bearing key its pairing depends on); budgets: sdf_dataset
  +63 net of ≤+90, triplanar +26 of ≤+25 (black's wrapping of the 8-argument
  hook signature).
- **§8.0.F statement merged to `main` in PR #82 (2026-08-24):**
  every claim in it re-run against `main` at `8ae0081` first (suite 664 passed /
  12 xfailed; the aliasing probe, the `parameters()` dedup, the `forward` path,
  the hash-machinery caller grep). Two design calls that diverge from the issues'
  first framing, argued in the statement: `subsample` is *decoupled* from cache
  content (raw indices cached, padding at draw) rather than keyed — so the
  subsample-key xfail gets rewritten, not unmarked — and #27's alias-strip hook
  is deliberately permanent rather than a delete-when module, because the three
  shipped model checkpoints are pre-fix forever. Rebased onto `main` post-#84
  (clean); commit 5's scope gains §7.1's save/load round-trip, the venue #84's
  annotations name.
- **§8.0.E merged to `main` in PR #79 (2026-08-24):** #5 closed by the merge;
  `wandb-optional` and `clamp-gradient-known-issue` both deleted. The first
  landing attempt — PR #78, stacked on #77's branch — merged into its
  surviving stale base instead of `main` (issue #5 staying open was the
  tell); #79 re-landed the same eight commits, reviewed on #78, and the trap
  is now a working convention in CLAUDE.md (PR #80).
  The work: statement → characterization → #5
  fix → amendment → dead-pair deletion → split, suite green and
  lint clean at every commit (660→664 passed; the #5 probe xfail passes unmarked
  from the fix commit on). #5: six guarded imports (schedulefree's sentinel
  pattern), raise-by-name at entry for every explicit request, and the one
  skip-not-raise — `get_mean_errors`' histogram tail yields `None` without wandb
  so training validation survives; payload byte-identical with wandb installed
  (pinned). Split: `get_mean_errors` alone moves to `recon_evaluation.py`, with
  a call-time import of `reconstruct_mesh`/`NoZeroLevelSetError` (no
  module-level cycle; the `test_predictive_validation` monkeypatch seam
  survives, green untouched). **Diverged from the statement:** the "trio move"
  became delete-two-move-one — the docs-reference checker went red mid-split
  and SCOPE §2's dead ruling (2026-08-22) had already deferred
  `tune_reconstruction` + `compute_correlation_coefficient` deletion to exactly
  this pass, so an amendment (committed before the split) deleted them instead
  (CHANGELOG Breaking; frozen namespace lists narrowed by two with the ruling
  cited in place; raise sites nine → eight; the tuning *intent* is parked as
  `NSM_TRAINING_IDEAS.md` Idea 12 — maintainer, 2026-08-24); the histogram-present pin fakes
  `reconstruct_mesh` with clean metric values rather than using the #29
  degenerate fixture — NaN metrics make `wandb.Histogram` raise `ValueError`
  and yield `None` even with wandb installed, so the degenerate path cannot pin
  Histogram presence; `recon_evaluation.py`'s split budget (≤ +15) came out
  ~+27 because the statement's budget line omitted the module docstring that
  8.0.C-style moves carry.
- **PR #77 merged to `main` (2026-08-24):** the `enforce_minmax` Open entry has
  its full gradient mechanics; Idea 11 carries the clamp-form axis.
- **§8.0.D merged to `main` in PR #76 (2026-08-24):** #28, #42, #49, #52, #59
  closed by the merge.
  The work: statement → characterization → #42 →
  #49 → #59 → sweep → #28 → #52 → split, suite green and lint
  clean at every commit (654→660 passed; 19→12 xfailed — every conversion a fix
  passing unmarked). #42: the warm-up unpacks the way `train_epoch` does, pinned
  via a stubbed `schedulefree` (AdamW + `train()`/`eval()`) since the crash was
  the trainer's, not the library's. #49: one boundary — `resume_epoch` names the
  last completed epoch, `>= 1` loads (History §11). #59: `+=`, and
  `add_plain_lr_to_config` loses its positional override (History §12 —
  wandb-only, ~×n_batches). Sweep: `train_epoch(return_loss, verbose)` deleted
  live-trainer-only (multi_head is #51's, `deprecated/` is 0b's). #28: history
  returned (wandb payload + epoch/lrs/targets/latent_norms per epoch, payload
  byte-identical); the harness's `train_epoch` wrapper deleted, baselines
  unmoved — the regression suite now tests the public contract. #52: names
  declared on `MultiSurfaceSDFSamples` in mesh-path-list order, trainer adopts
  or refuses at entry; deliberately not in the cache key. Split: four private
  helpers, `train_epoch` whole, net +57 of +80. **Diverged from the statement:**
  `_schedule_free_eval_warmup` also takes `optimizer` (the statement's signature
  omitted it, but `optimizer.eval()` lives in the block); #52's fix had to touch
  the pre-existing validation tests — a bare `MagicMock` dataset auto-creates a
  truthy `mesh_names` the adoption path would read, so those mocks now pin
  `mesh_names=None`; a stale `train_deep_sdf.py:401` line-number citation in the
  clamp characterization docstring rode along (symbols only, per #31).
- **§8.0.C merged to `main` in PR #74 (2026-08-23):** #15, #16, #29 closed by
  the merge.
  The work: statement → characterization → #15 →
  #16 → class sweep → #29 → split, suite green and lint clean at
  every commit. #15: readers unified on `"pts"`; the two dataset-class probes
  collapsed, their dead `"gt_sdf"` fallback with them. The `"xyz"` transitional
  alias was then **deleted on the same branch** (maintainer, 2026-08-23:
  sooner-than-later — and since no tagged release ever carried it, waiting for
  v0.3.0 would have meant it never shipped at all). #15 is now a clean Breaking
  rename in the CHANGELOG; external `["xyz"]` readers get a loud `KeyError`. #16: honoured (History §9) — the
  memory said delete, but the measured blast radius is only never-shipped
  multi-object sampled runs and the intent is config-plumbed; the harness's
  sampled tests shrank ~1000× and the suite dropped ~11s. #29: raises
  `NoZeroLevelSetError` by name; `get_mean_errors` catches and scores NaN
  (History §10). The split: `latent_fit.py` + `wandb_logging.py`, `main.py` keeps
  `reconstruct_mesh` + evaluation + a permanent re-import block, net +67 of the
  +80 budget, both namespaces frozen by test. **Diverged from the statement:**
  the sweep had to reach all four trainer call sites and the default-config
  generator (the statement named only `train_deep_sdf.py:238`) — the plumbing was
  wider than enumerated; #29's planned NaN-latent-row into `Regress` was replaced
  by skip-plus-NaN-r² because sklearn refuses NaN design rows and degeneracy is
  decoder-level, so the failure is all-or-nothing; the three Open table rows
  retired together in the #29 commit rather than one per fix commit; a
  single-object end-to-end test rode with #16 (the branch was never runnable
  before #15+#16, so its first execution belonged with the fix that made it
  cheap).
- **#48 remnants merged to `main` in PR #73 (2026-08-23):**
  `norm_penalty_type='barrier'` raises by name outside its `(min, max)` range
  instead of NaN — pre-fix runs *completed*, with `nan` in every loss readout and
  a finite gradient pushing the norm **away** from the range (verified by
  execution; History §8, CHANGELOG); `get_mean_errors` hands `Regress.add_latent`
  the fitted latent (detached/CPU/flat), not the result dict — born broken in
  `2811d27` (Jul 2023), never worked, always crashed → no History entry.
  Ride-alongs the work exposed: the docs-reference checker's `Regress` exemption
  was dead and its comment false (the class is in this repo) — deleted; SCOPE
  §2.5's "worth pinning down" question answered (never-worked, not
  broken-after-the-paper). Suite 505 passed / 14 xfailed.
- **Slice B landed (2026-08-23):** merged to `main` in PR #72 —
  `read_meshes_get_sampled_pts` orchestrates three private helpers
  (`_register_to_mean`, `_compute_shared_frame`, `_draw_surface_samples`); #17 and
  #69 fixed en route, both closed by the merge (#17: `KNOWN_ISSUES` § History §7
  records the one affected run class — multi-object fits with `get_rand_pts=True`
  on `scale_jointly=False` models, never training data; #69: fixed by unification,
  the in-memory mutate-in-place branch deleted net −48, disk numerics unchanged).
  #3 got its structural precondition only: sigma's frame-dependence is stated in
  `_draw_surface_samples`' docstring, the single site where sigma is consumed.
- **Slice A landed (2026-08-22):** merged to `main` in PR #71 — `sdf_dataset.py`
  holds the classes + permanent re-import block, the 13 leaf helpers in
  `NSM/datasets/utils.py`, the two readers in `NSM/datasets/mesh_sampling.py`;
  characterization in `test_dataset_helpers.py` / `test_sampled_pts_readers.py` /
  `test_import_compat.py` (frozen name list on both import paths).
  Characterization surprise, pinned and docstring-corrected: `unpack_numpy_data`
  accepts a dict only with `list_additional_keys=[]` — the default reads
  `data_.files`, NpzFile-only — despite its (now fixed) "NpzFile or dict" claim.
- **Scale preservation is an idea, not a defect** (maintainer, 2026-08-22): similarity
  registration deliberately matching subjects to the reference's size is a valid
  design; keeping true size is an *additional mode* worth having. It therefore lives
  as `NSM_TRAINING_IDEAS.md` Idea 6 (with the measured evidence and design sketch),
  not in `KNOWN_ISSUES.md` and not on the tracker; the durable fact — similarity =
  rigid + uniform scale, size does not survive registration — is stated in the
  `reference_mesh` / `register_to_mean_first` docstrings.
- **Versioning (2026-08-23):** released state is `v0.2.0` (tag + `__version__`
  agree). The Unreleased CHANGELOG section holds this branch's Breaking set, so
  the release that ships it is **v0.3.0** per §10.1 (breaking bumps the minor).
  Cut timing is the maintainer's call — "soonish, or at the end of this cleanup";
  nothing gates it now that the `"xyz"` alias is gone rather than
  waiting-to-be-deleted. §10.1's setuptools-scm item (derive the version from the
  tag) is still open and would naturally ride with the v0.3.0 release PR.
- **Blocked on:** nothing for the Next. Standing exception, so it stops
  re-deferring silently: **0b** (the nsosim/fork consumer survey, §3) has needed
  maintainer input since Phase 0 — nsosim is not available locally — and gates
  the `train/deprecated/` quarantine (#18); every re-import-surface decision so
  far has substituted "forks assumed to use both paths" for its answer.
- **Context for whoever picks this up:** PR #68 carried one commit per concern, so
  `git log NSM/datasets/sdf_dataset.py` explains each fix. Decisions of record are in
  the PR body ("Of note"): the single-mesh clip was removed, not copied; the timing keys
  are optional diagnostics, not batch contract; `subsample=None` is refused, not
  resurrected; `joint_scale_buffer` deliberately stays out of the cache key until #19.
  The one results-affecting change is `docs/KNOWN_ISSUES.md` § History 6 (sampling cube),
  including the trap that pre-fix caches keep serving old points because the buffer is
  not in the cache key.
- **Formerly deferred, now executed (PR #85):** #19 (cache key) and #27 (checkpoint
  aliasing). Both force downstream regeneration or migration, so they were held to land
  together as one release rather than making consumers migrate twice — that release
  constraint still binds the v0.3.0 timing (see Next).
- **Done:**
  - Phase 0 scope — `docs/SCOPE.md`
  - Phase 1 map — `docs/ARCHITECTURE.md`, `docs/AUDIT_FINDINGS.md`
  - Phase A LR fix — PRs #9, #10, #11
  - Phase 3 §7.1 numerical regression harness — PRs #13, #14, merged to `main` in #30
  - Seeding fix + harness punch-list — `eaee68c`, `e432f9a`, `d2ba1c7`, `94b48f0`
  - **Everything above reached `main` for the first time in PR #30.** `main` had seen none
    of it before that.
  - Docs cite code by symbol, not line number, with a test — PR #31
  - flake8 to zero, CI gates it, tooling backlog closed, docs build set up — PR #32
  - Retired `worklist #N` numbering repointed at issues; the inverted `normalize_coordinates`
    test rewritten — PR #33
  - `CHANGELOG.md` and `v0.2.0` — PR #36
  - Phase 2, mechanical slice: docstrings that contradict their signatures — PR #37
  - #21 closed and #20's function-level half — PR #38
  - `main` protected: 1 review, four required status checks, admins exempt
  - Audit disposition approved and filed 2026-08-22: issues #40–#61 (mapping in
    `AUDIT_FINDINGS.md` § 0.3), #6 closed, folds commented onto #20/#22/#23/#35 —
    landed via PRs #62 (disposition) and #63 (quick wins)
  - Wave 1 — the four decided fixes (#47, #44, #53, #48-partial) plus the
    cyclic-anneal and Constant-schedule ride-alongs — PR #64
  - SCOPE rulings pass: the register's 13 § 4 rulings transcribed into SCOPE.md
    (every claim re-run against `main` first), the § 0.5 dead cluster deleted
    (`symmetric_chammfer`, `sdf_gradients`, `find_object_bounds_random_sampling`,
    `NSM/configs/deep_sdf_config`), and the stale §1 default-config bullet corrected
    post-#64 — PR #65
  - Phase-2 prose pass: all 62 § 3 corrections (two were already moot — sinkhorn left
    with #64's EMD deletion, the modulated-activations print with #63 — and the
    sdf_dataset unused-imports entry was a deliberate skip per its verdict), plus the
    `get_latent_vecs` variational docstring, plus ride-alongs the corrections exposed
    (Makefile `PDOC_ALLOW_EXEC` had lost its rationale to #64 and was removed after
    re-verifying the build; the never-pushed mesh-interp archive branch/tag is now
    recorded in the completed interpolation plan's Diverged) — and **the register
    `docs/AUDIT_FINDINGS.md` is deleted**, PR #66. Draft→issue mapping, preserved
    from its § 0.3: 1→#40, 2→#41, 3→#42, 4→#43, 5→#44, 6→#45, 7→#46, 8→#47, 9→#48,
    10→#49, 11→#50, 12→#51, 14→#52, 15→#53, 16→#54, 17→#55, 18→#56, 19→#57, 20→#58,
    21→#59, 22→#60, 23→#61 (13 withdrawn — the variational behaviour is deliberate)
  - `sdf_dataset.py` file:function fixes, maintainer-reviewed 2026-08-22: #40 (one
    buffered-cube helper, symmetric and unclipped — History §6), #41 (empty pos/neg
    raises by name; unused surfaces handled), #43 (`subsample` validated,
    `joint_scale_buffer` accepted), #61 (integer-reference registration path;
    `combine_meshes` keeps its Mesh contract), #22 (in-memory datasets train; timing
    keys optional), #23 (zero-count combos skipped), #24 (`LOC_SDF_CACHE` at
    construction) — one commit per concern, all closed by PR #68. Also filed and
    pinned strict-xfail: #67 (the None-surface path has never built end to end)
  - `sdf_dataset.py` semantic docstring pass — landed in PR #70, merged 2026-08-22:
    `mean` deleted from both samplers —
    the file's only accepted-and-never-read parameter left, by AST scan (#20 instance;
    CHANGELOG entry); every public function/method documented, silent conventions
    written down ("pts"/"xyz" key asymmetry, npz vs in-memory spelling, batch
    contracts, cache-key omissions), false text corrected; the in-memory
    joint-scaling defect pinned strict-xfail, filed as #69 with approved text and a
    `KNOWN_ISSUES` § Open entry; `_harness.py`'s stale import-time `LOC_SDF_CACHE`
    claim fixed (stale since #24); the scale-erasure fact measured (ICP scale factor
    exactly 1/1.3 on a 1.3-vs-1.0 sphere pair) and recorded as `NSM_TRAINING_IDEAS.md`
    Idea 6 plus docstring clauses — initially misfiled as a KNOWN_ISSUES entry, moved
    on the maintainer's call that it is a design alternative, not a defect
- **Surprises:**
  - **"Fixed seed" was not available.** NSM called no seeding function anywhere, and the
    near-surface sampler could not be seeded by any caller. Closed via pymskt 0.1.21 plus
    `derive_seed`; see `docs/KNOWN_ISSUES.md` § History.
  - **The seed had to be *derived*, per (subject, sampling pass, surface), keyed on mesh
    content.** Keying on list position and then on the cache hash each silently changed a
    subject's data when something unrelated moved.
  - **"Pin current behaviour, bugs included" reported the opposite of the truth.** Written
    that way, ~20 tests passed *because* something was broken. They now assert the
    behaviour NSM should have, marked `xfail(strict=True)`.
  - **Quarantinable code was 882 lines, not ~1,800.** Every module the plan expected to
    fall held a capability nothing else implements.
  - **Both inferred audit claims that were later tested were wrong**, in the direction of
    overstatement. This is why `AUDIT_FINDINGS.md` entries are hypotheses, not findings.
  - **The "obvious fix" has now been worse than the bug twice**, and the second was far
    worse than the first. `normalize_coordinates`' `padding` would have shifted the SDF by
    ~0.06 on a `tanh`-bounded output. `get_pts_center_and_scale`'s `center`/`scale` would
    have switched scaling **off** on every default run, because every caller passes
    `scale=norm_pts` and `norm_pts` defaults to `False` — a different coordinate frame for
    every dataset and checkpoint ever produced. Both were deleted rather than honoured.
    **Assume the next accepted-and-ignored argument is the same shape until measured.**
  - **`norm_pts: false` has never done what it says.** `center_pts` and `norm_pts` decide
    *whether* to normalize, not *which* operation runs, so the shipped config asks for
    centering without scaling and gets both. No NSM run has ever been centered-but-unscaled.
  - **Line numbers in prose do not survive editing the code they describe.** A seven-line
    portability fix invalidated ~15 citations in the same sitting, and a `black` pass would
    have moved them again. Citations are now symbols, checked by a test (#31). A checker for
    line *numbers* was considered and rejected: it converts silent rot into recurring
    transcription work.
  - **The 439 flake8 violations were 302 formatting, 110 prose, 27 real.** The backlog that
    justified `continue-on-error` for years was two commands and an afternoon.
  - **CI's lint step never ran `black` or `isort`** despite being named for them. That is
    how nine files drifted out of compliance unseen.
  - **`TestAFreshlyTrainedDecoder` passes with the training discarded** — verified by
    rebuilding the model after training and rerunning. An untrained decoder still yields a
    surface. Filed as #34.
  - **The `Target` requirement is already in `v0.1.0`**, so no released version predates it.
    A downstream on positional entries cannot dodge the migration by pinning an older tag.
  - **Something in the suite resets the locale to ASCII.** A bare `read_text()` passes in
    isolation and raises `UnicodeDecodeError` under the full suite; cost 27 failures to find.
  - **#41's None-surface trigger cannot reach `sdf_pos_neg_idx` end to end.** A `None`
    surface dies earlier, in `MultiSurfaceSDFSamples.get_sample_data_dict`: the buffer is
    preallocated `sum(n_pts_)` rows but the sampler returns only the non-None surfaces'
    points. The fdfe902 feature has therefore never worked through the dataset class;
    filed as #67, pinned strict-xfail, and the NaN-column handling is tested by direct
    method call until it is fixed.
  - **The shipped default config carries `dataset_uniform_pts_buffer: 0.2`**, so #40's
    asymmetric-cube bug affected real training data, not just a dormant parameter — it
    became History §6 rather than close-by-deletion. And because the buffer is absent
    from the cache key (#19), the fix alone does not resample: pre-fix `.npz` caches
    keep serving the old points until deleted.
  - **The docstring pass found a never-worked configuration the fix pass had missed.**
    Writing `norm_and_scale_all_meshes`' docstring forced running its in-memory branch:
    `KeyError` on both classes, plus a silently-dropped `joint_scale_buffer` behind it.
    Documenting a function honestly means executing it — the branch had survived #22's
    in-memory fixes and #43's `joint_scale_buffer` work untouched because neither had a
    reason to run that exact combination.
  - **#17's blast radius was narrower than the issue feared.** The executed determination
    (§8.0.B) showed only *one* configuration of `include_surf_in_pts` ever returned
    results — centering on, numeric sigmas; the production-shaped configuration
    (`scale_jointly=True` → centering off) always crashed with `UnboundLocalError`, and
    any `None` sigma always crashed with `ValueError`. Silent corruption was real but
    confined to multi-object fits on `scale_jointly=False` models; training data was
    never touched because the dataset classes never pass the flag.
  - **Test module basenames must be unique across `testing/`.** The tree has no
    `__init__.py` files, so pytest imports test modules flat: a second
    `test_import_compat.py` in another directory fails collection with "import file
    mismatch". Cost one amended commit in §8.0.C.
  - **`Regress` cannot take an honest-NaN latent row** — sklearn's
    `LinearRegression.fit` refuses NaN design matrices — so #29's "score the failed
    subject NaN" had to mean *skip the regression and report NaN r²*, not feed NaN
    rows through it. Safe only because degeneracy is decoder-level: every subject
    fails together, so there is no partial-alignment case to handle.
  - **#69's "two halves" were one defect, and the fix was net-negative.** The KeyError
    and the missing buffer both came from the in-memory branch reimplementing what the
    disk branch already did. Unifying on the disk branch's semantics (compute the frame,
    let `__getitem__` apply it) deleted the entire mutate-in-place branch — −48 lines —
    instead of repairing it.
  - **`unpack_numpy_data` makes key-presence checks dead.** It sets every requested
    group unconditionally — an absent group comes back as an *empty list* — so
    `"pos_idx" not in data` can never be True on a loaded cache. The single class's
    documented pre-index-layout upgrade had never fired (pre-layout caches crashed
    `__getitem__` with IndexError instead); found by §8.0.F's characterization,
    2026-08-24. A check against an unpacked cache must test length, not presence.

---

> Measurements: run `pytest testing/ --cov=NSM` rather than trusting any figure quoted
> below — the numbers in §1.3, §5 and §7 are as-measured-then and are not maintained.

---

**Motivation:** The LR-schedule bug (see §1) was a silent numerical error that ran
undetected for ~3 years and was found by an external collaborator, not by us. It is a
symptom, not an incident. This plan makes that class of bug findable and preventable.

> **Scope.** Code health only: documentation accuracy, structural mapping, test coverage,
> and decomposition of monoliths. Deliberately **out of scope**: new science
> (`NSM_TRAINING_IDEAS.md`, `NSM_RECTIFIED_FLOW_CORRESPONDENCE.md`) and the ICP
> registration-robustness work, which has no plan file in this repo. Those resume *after*
> Phase 2, on firmer ground.

---

## 1. Why — the evidence that motivated this

### 1.1 The triggering bug

`get_optimizer()` built optimizer param groups in the order `[latent, model...]` and
assigned `lr_schedules[1]` to latents, `lr_schedules[0]` to the model — correct. But
`adjust_learning_rate()` then reassigned **by position** every epoch:

```python
for i, param_group in enumerate(optimizer.param_groups):
    param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)
```

**Every Adam/AdamW training run from May 2023 to Aug 2026 trained latents under the model
schedule and vice versa — for 100% of the run.** `get_optimizer` does set the intended
rates at construction, but the epoch loop starts at 1 and `adjust_learning_rate` runs at
the top of `train_epoch`, so the correct values are overwritten before the first
`optimizer.step()`.

`schedule_free_*` runs skipped `adjust_learning_rate` and so were never mis-mapped — but
they were arguably hurt worse, since every config was *tuned* against the Adam path and
running one schedule-free applies the values inverted and undecayed. See
`docs/KNOWN_ISSUES.md` §1.

Reported 2026-07-10 by an external collaborator, credited in the ledger.

### 1.2 Why our tooling could not have caught it

- Nothing crashed. No exception, no NaN, no shape error. Only the numbers were wrong.
- The two functions live in different conceptual layers and were never tested together.
- `NSM/utils.py` — the file containing both — has **zero docstrings** across 22 functions
  and 5 classes, so the `[0]=model, [1]=latent` convention existed only in the author's head.
- No test asserted "what learning rate does each param group have at epoch *k*".

Unit tests on extracted helpers would **not** have caught this. Only an assertion on
observable numerical state across an epoch boundary would. That shapes Phase 2.

### 1.3 Measured state of the codebase (2026-08-14)

| Metric | Value |
|---|---|
| `NSM/` source | 11,565 lines |
| Docstring coverage | 48% (122/247 functions, 8/26 classes) |
| Test coverage | 32% (100 tests + 1 skip, 13s runtime) |

Coverage is inverted relative to risk — the newest code is best tested, the core that
everything depends on is barely tested at all:

| Module | Lines | Docstrings | Coverage | Note |
|---|---|---|---|---|
| `datasets/sdf_dataset.py` | 2,195 | 55% | **7%** | largest file in repo |
| `reconstruct/main.py` | 1,443 | 62% | **24%** | production path |
| `train/train_deep_sdf.py` | 618 | **0%** | **10%** | 618 lines in 2 functions |
| `utils.py` | 280 | **0%** | 25% | most-imported module; held the bug |
| `models/modulated_periodic_activations.py` | 252 | **7%** | 67% | |
| `mesh/correspondence_metrics.py` | 699 | 100% | 94% | recent work — the target state |
| `models/loader.py` | 387 | 100% | 84% | recent work |

### 1.4 The production surface is not yet established

NSM is a training library first. `NSM/train/`, `NSM/datasets/sdf_dataset.py`,
`NSM/losses.py` and the mesh path are the product, not internals — so most of the 11.5k
lines are in active use, not incidental.

Known consumers:

| Consumer | Surface used |
|---|---|
| Training (first-party) | `train/`, `datasets/`, `models/`, `losses`, `mesh/` |
| `kneepipeline` inference | `TriplanarDecoder`, `reconstruct_mesh` |
| `nsosim` | mesh interpolation; extent unverified |
| Published models, shared configs | `model_params_config.json` schema, checkpoint format |
| Downstream forks | unknown; they carry their own trainers |

Checkpoint and config **formats** are part of the contract even though nothing imports
them. The LR fix needed a migration path precisely because a config file is a public
interface.

None of the above is enumerated. Until it is, assume any refactor can break something
outside this repo. Enumerating it is the first Phase 0 task (§3), not an input to this plan.

### 1.5 Open issues are all symptoms of the same disease

| # | Title | Class |
|---|---|---|
| 1 | `norm_and_scale_all_meshes` shouldn't work if not loading in memory | dead/incorrect path |
| 2 | SDFSamples — slow loading | performance |
| 3 | Scale of sigma depends on `scale_jointly` | ambiguous API contract |
| 5 | Wandb not needed if just doing inference | unnecessary coupling |
| 6 | pyvista `polydata._faces` syntax update | upstream API rot |

Four of five are exactly what a systematic audit surfaces. Issue #3 is already written up
as `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` and stalled mid-Phase-1.

---

## 2. Guiding principles

1. **The map precedes the documentation.** Never write a docstring for code that is a
   deletion candidate.
2. **Quarantine, don't delete.** Downstream forks and `nsosim` may reach into anything. Moving
   to `NSM/deprecated/` with a `DeprecationWarning` is reversible; `git rm` is a support
   burden when someone's pipeline breaks silently.
3. **Behavioural tests before structural tests.** A golden-output regression harness that
   catches silent numerical drift is worth more than 500 unit tests on pure helpers.
4. **Every bug found gets a provenance entry.** For science code, "which results are
   affected by which bug" is a first-class deliverable (§7).
5. **Silent behaviour changes are forbidden.** Any change that alters training or
   reconstruction numerics must fail loudly on old inputs and offer an explicit path to
   reproduce historical behaviour. Phase A (§4) is the reference implementation.

---

## 3. Phase 0 — Decide what the library is for  ✅ (0a done; 0b open)

**Why first:** Phase 1's map cannot mark anything "deprecated" until these calls are made.
Blocking all later phases.

- [x] Write a one-page scope statement: what NSM is, what it supports, what it does not.
- [x] Rule on each ambiguous module — **supported**, **deprecated**, or **dead**.
      **All five proposed rulings were refuted.** See `docs/SCOPE.md` §2:
  - `train/train_deep_sdf_multi_head.py` (428 lines) — the "broken" half is confirmed
    (`:85` passes the `for model in models` loop variable to `get_optimizer`). The
    "superseded" half is **false**: it trains N *independent* decoders against one shared
    latent embedding, whereas `train_deep_sdf` with `objects_per_decoder > 1` is one
    network with N output channels. Different architecture, not a replacement. Ruling:
    **supported, fix it** — a two-identifier repair
  - `train/deprecated/` — split. `..._multi_surface_orig.py` (562) is a strict subset →
    dead. `..._orig.py` (318) holds the only live `sample_difficulty_lx` implementation →
    port ~12 lines into `train_deep_sdf.py` first
  - `mesh/refine_mesh.py` — zero importers confirmed, but it is the only cross-mesh
    subdivision with ID preservation, and the completed interpolation plan records in
    writing that it was kept deliberately. **Research, keep**
  - `reconstruct/reconstruct_latent_S3.py` — only Sim(3) joint pose+latent fit in the repo,
    and an active item in the ICP plan §5. **Deferred research, scheduled**
  - `reconstruct/cartilage_func.py` — **production**: wired into `DICT_VALIDATION_FUNCS` in
    the live trainer. `predictive_validation_class.py` — research, live caller
  - `configs/generate_sdf_default_config.py` — supported. Confirmed
- [x] Establish the public API contract — `docs/SCOPE.md` §3. Consumer surface is exactly
      two symbols; tiers proposed as 6 public-stable / 48 public-provisional / rest internal.
- [ ] **`__all__` in code — deferred, and the plan text needs amending.** `NSM/__init__.py`
      imports only `utils`, so a top-level `__all__` would either name unbound symbols or
      force eager subpackage imports — which would pull `wandb`, `vtk` and a root-logger
      reconfiguration into every `import NSM` and destroy the one property that makes the
      consumer's import cheap. Put `__all__` per subpackage instead. `docs/SCOPE.md` §3.3.
- [x] Include the checkpoint and `model_params_config.json` formats in that contract —
      `docs/SCOPE.md` §4. Six on-disk contracts, none versioned except the LR `Target` key.
- [ ] **0b — survey downstream consumers before quarantining anything.** `nsosim` is not
      available locally; this remains open and now gates *only* the physical quarantine
      move, not the map. See `docs/SCOPE.md` §5.

**Deliverable:** `docs/SCOPE.md` ✅ + an `__all__` per subpackage (deferred, see above).

---

## 4. Phase A (done) — LR fix as the migration template

Delivered 2026-08-14/15 across PRs **#9**, **#10** and **#11**, all merged. Recorded here
because it is the **pattern** every subsequent behaviour-changing fix should follow.

### What shipped

- **`Target` on every `LearningRateSchedule` entry** (`"model"` / `"latent"`). Entry order
  is ignored. A config missing it — including a half-annotated one — raises, printing a
  paste-ready annotated copy of the caller's own entries.
- **No positional indexing anywhere in the LR path.** `get_learning_rate_schedules`
  returns a `{target: schedule}` dict; param groups carry a matching `target`;
  `adjust_learning_rate` is one lookup. `name` survives as a human label only.
- **Pre-Aug-2026 checkpoints are refused at load time**, not left to a downstream
  `KeyError` — which is skipped for `schedule_free_*` and would have failed hours in.
- **Migration code isolated** in `NSM/_lr_migration.py`, with a delete-when condition in
  its header.
- `docs/KNOWN_ISSUES.md` seeded, with ShapeMedKnee_2024 as a worked example.

### Approaches tried and rejected

Each of these shipped in a first draft and was removed before the work finished. They are
listed because the next behaviour-changing fix will be tempted by the same three.

- **A config flag declaring the entry order** (`lr_schedule_convention: v2 |
  legacy_swapped`). It made the ambiguity explicit rather than removing it, and left the
  ordering itself in place. Replaced by a per-entry `Target`. A flag that describes how to
  read positions is a sign the positions should not carry meaning.
- **Storing param-group names in the checkpoint** (`optimizer_group_names`).
  `state_dict()` already retains custom group keys and `load_state_dict()` restores them,
  so it never did anything. It was carried over from a downstream fork along with a
  docstring justifying it; the justification was tested and proved false, and the code
  survived anyway on a replacement rationale.
- **Restoring names positionally for old checkpoints** (`rename_` /
  `restore_optimizer_param_group_names`). Both guessed identity from group order, which is
  the assumption the fix existed to remove. Pre-Aug-2026 checkpoints are now refused
  outright.

Final size: `NSM/utils.py` grew by 173 lines, down from 341 in the first draft, with no
loss of function. Everything removed came out in review rather than from any automated
check — which is what the `Making Changes` section of `CLAUDE.md` exists to change.

**The generalizable lessons:**

1. Internal correctness guards and user migration guards are different mechanisms, and a
   behaviour-changing fix needs both. A name-based `KeyError` protects the invariant but
   is invisible to someone holding an old config.
2. Fix the *class* of defect, not the reported instance. This bug was positional coupling;
   the first fix removed one of three instances in the same code path.
3. Migration scaffolding needs a lifespan declared at write time, or it becomes permanent
   API by default.

---

## 5. Phase 1 — Map the codebase  ✅ (quarantine step gated on 0b)

**Order note:** this was step 1 in the original sketch. It runs *after* Phase 0 and
*before* documentation, so we never document code we are about to quarantine.

- [x] Build a module dependency graph; identify cycles. `docs/ARCHITECTURE.md` §2.
      Built with `ast` rather than `pydeps` (not installed), so deferred imports are
      distinguished structurally. **Layering is strictly unidirectional** and there is
      **exactly one cycle**, `utils` ↔ `_lr_migration`, which is deliberate and documented
      at both ends. Not a refactor target.
- [x] For every public function/class: caller count, in-repo and in `kneepipeline`.
      Note for anyone repeating this: exclude `build/lib/`, a stale gitignored copy of the
      whole package that double-counts every naive `grep -r`.
- [x] Mark each module — ledger at `docs/ARCHITECTURE.md` §3, with lines, coverage,
      docstring count, *inaccurate*-docstring count, importers and status.
- [x] Flag duplicated logic. **Six traps, not one.** The two `adjust_learning_rate`
      implementations are worse than described: `reconstruct/utils.py`'s copy is leaked
      into `NSM.reconstruct`'s namespace by `from .main import *`, so
      `from NSM.reconstruct import adjust_learning_rate` silently binds the wrong one.
      `docs/ARCHITECTURE.md` §6.
- [ ] Quarantine everything ruled dead → gated on **0b**, and it is now one file.

**Deliverable:** `docs/ARCHITECTURE.md` ✅ + `docs/AUDIT_FINDINGS.md` (216 findings) ✅.
**Checkpoint (re-measured on `main`):** 11,861 lines, 34% coverage, 153 tests + 1 skip.
**Quarantinable: 882 lines, not ~1,800** — and 12 of them must be ported out first. Every
module the plan expected to fall turned out to hold a capability nothing else implements.
Treat the 1,800 as a prediction that was tested and failed, not a target to hit.

---

## 6. Phase 2 — Documentation accuracy pass

Only over modules that survived Phase 1.

- [ ] Docstrings on every surviving public function/class: purpose, args, returns, raises,
      **and any silent convention** (index orderings, coordinate spaces, units).
      `[0]=model, [1]=latent` is precisely the kind of thing that must be written down.
      *(Partial, through the per-module passes: PR #37 — docstrings contradicting their
      signatures; PR #66 — the 62 prose corrections plus the `train/`/`utils` conventions;
      PR #70 — the `sdf_dataset.py` semantic pass. Modules no pass has reached remain.)*
- [ ] Verify each existing docstring against the implementation — 48% coverage says
      nothing about whether those 48% are *true*. *(Same per-module status: the passes
      above verified what they touched, nothing else.)*
- [ ] Enforce mechanically so it cannot rot: add `flake8-docstrings` (or `pydocstyle`) to
      `make lint`, failing on missing docstrings in public API. *(Re-checked 2026-08-24:
      `make lint` runs flake8 with no docstring plugin — fully open, and the accuracy paid
      for above can rot silently until this lands.)*
- [ ] Rewrite `CLAUDE.md` §Architecture from the Phase 1 map; drop the stale
      "EIKONAL LOSS HAS NOT BEEN TESTED" shout-comment into a tracked issue instead.
      *(Still open. The shout's substance now lives in §8.2's gate — both entry points
      raise `NotImplementedError` — but the comment itself still stands in `CLAUDE.md`.)*

### 6.1 Warnings, not just docstrings

Phase 1 found capabilities that are real but unready. A docstring is the wrong instrument
for those — the user who needs the warning is the one who did not read the docstring. Each
of these is small and independent of the decomposition work:

- [x] **Rewrite the `train_deep_sdf_multi_head` deprecation text** (`:30`). It currently
      says "Use `NSM.train.train_deep_sdf` with `'objects_per_decoder' > 1` instead" —
      advice that silently hands the user a different architecture (`docs/SCOPE.md` §2.1).
      Say broken-and-unfixed; name no replacement. Keep it out of the documented surface:
      its hyperparameters have never been tuned. *(Done in PR #66: the module now warns
      broken-and-unfixed at call time, names no replacement, and says why
      `objects_per_decoder > 1` is not one.)*
- [x] **Warn on the Eikonal loss at the point of use.** Never run by its author, never
      executed by a test. See §3 below — the warning's wording depends on whether it works.
      *(Superseded 2026-08-15 by the §8.2 gate: `eikonal_weight > 0` raises
      `NotImplementedError` at both entry points — stronger than a warning.)*
- [ ] **Fix, then warn, then document `mesh/refine_mesh.py`,** in that order. It raises
      `UnboundLocalError` on its own defaults (`:399`), so documentation written today
      describes something nobody can run. `docs/SCOPE.md` §2.3 lists the three conditions.
- [ ] **Guard `sample_difficulty_lx` when porting it** out of `train/deprecated/`. Its
      off-state has never been exercised; a feature whose disabled path is untested turns
      itself on eventually. Document it at the config key, not only in code.
      *(= #18; rides with the 0b quarantine.)*
- [ ] **Find the config keys that silently do nothing** because their implementing branch is
      commented out. These read as working features and produce no error — the inverse of
      the hazard above, and harder to notice. *(One instance found and deleted by 8.0.C's
      sweep — `compute_recon_loss(n_samples_assd)`, its implementing call commented out.
      The systematic hunt is still open.)*

**Deliverable:** docstring coverage ≥90% on surviving public API, lint-enforced.

---

## 7. Phase 3 — Test to a known-good baseline

> **§7.1 is DONE** (PRs #13, #14, then `eaee68c`/`94b48f0`). `testing/NSM/regression/`,
> green in CI via `make test`; what it asserts and how to work on it is in that directory's
> `README.md`. Every checkbox in §7.1 below is ticked. The defects it surfaced are in
> `docs/KNOWN_ISSUES.md` § Open. See the **State** block at the top of this file for
> what to do next and for what §7.1 assumed that turned out to be false.
>
> §7.2 has NOT started. Phase 4 is now unblocked.


**The step that stalled before.** It stalls because "test everything" against ~3,000
uncovered statements has no end condition. Bound it by *risk*, not by coverage percentage.

### 7.1 First and most important: end-to-end numerical regression harness  ✅ DONE

Before any unit tests. A tiny synthetic dataset (2–3 analytic meshes, 8 epochs, CPU,
fixed seed), then assert against stored baselines:

- [x] Training: loss trajectory, final latent norms, **per-param-group LR at each epoch**
- [x] Reconstruction: fitted latent, output mesh vertex positions, surface metrics
- [x] Tolerances tight enough to catch a real change, loose enough to survive
      platform float noise. Store baselines as versioned artifacts.
- [x] Must run in CI in <2 minutes. (~20s; whole suite 33s)

**Scope widened 2026-08-15** after Phase 1. The four items above leave the layer with the
worst findings untouched, so add:

- [x] **The dataset cache round-trip.** `sdf_dataset.py` is 2,195 lines at 7% coverage and
      its worst findings are *silent wrong data*: `mesh_to_scale`, `uniform_pts_buffer` and
      `subsample` all change cached content and none is in `get_hash_params`, so a second
      run silently reuses the first's alignment. A harness that builds its data in memory
      never touches that path. Assert: build cache → reload → samples identical; and
      changing a hashed parameter changes the key.
- [x] **The consumer's actual entry point, not just `reconstruct_latent`.** `kneepipeline`
      calls `reconstruct_mesh` with a *list* of paths (multi-object). That function has
      exactly one executed line in the current suite — its `def`. Assert the returned
      `mesh` list **order** too: index 0 = bone, 1 = cartilage is a load-bearing contract
      that nothing in the signature, docstring or result dict names.
- [x] **Model save/load round-trip.** `testing/NSM/models/test_loader.py:232` loads a saved
      model and never compares its output to the original's, so a wrong-but-same-shaped
      forward passes every assertion. Train → save → load → assert bitwise-identical.
      *(Closed in §8.0.F commit 5, PR #85:
      `test_model_roundtrip.TestRoundTrip::test_a_bare_load_state_dict_round_trips_bitwise`
      covers the consumer's bare path; `test_a_trained_model_round_trips_bitwise` the
      `load_model` path — both bitwise against the in-memory original.)*
- [x] **Name the CPU/GPU gap rather than discovering it later.** A <2-minute CI harness is
      CPU; production is CUDA. Add a separate opt-in GPU test asserting the seed-ordering
      constraint `kneepipeline` depends on (`torch.manual_seed` *after* `.cuda()`, per
      `docs/KNOWN_ISSUES.md`), and state in the harness that CPU baselines do not
      bound GPU divergence. *(Both halves shipped in `94b48f0` —
      `testing/NSM/regression/test_gpu.py`, skipped without CUDA: its module docstring
      states the gap with measured numbers, and `TestSeedOrderingAroundCudaTransfer` pins
      the seed-ordering measurement rather than the belief. **This box was left unticked
      until 2026-08-26** and carried a "(Attach to the v0.3.0 release PR)" note for work
      that was already in the tree — found by re-measuring instead of reading. The release
      PR (§8.0.O) still owns the one part that is genuinely release-time: running a real
      shipped checkpoint through it.)*

Roughly 30–40% more than the original four items, still one bounded artifact, still under
two minutes.

This single harness is what protects every refactor in Phase 4, and it is the only thing
that would have caught the LR bug.

**It is also how the audit register gets settled.** `docs/AUDIT_FINDINGS.md` holds 216
findings of which **178 are read-only inference, not executed** — and both inferred claims
that have since been tested were wrong, in the direction of overstatement (the triplanar
"affine map" claim, and the eikonal "forward+backward runs" claim). There is deliberately
**no separate verification pass**: findings get confirmed or killed as a by-product of
building these tests, so the effort leaves permanent tests behind instead of throwaway
scripts. The register's job is to tell this section where to point. The 30 landmines that
are both inferred and on the production path are the highest-value targets.

### 7.2 Contract tests on the production API (§1.4)

- [ ] `TriplanarDecoder` — construction from config, forward shapes, checkpoint round-trip
- [ ] `reconstruct_mesh` — end-to-end on a synthetic model, asserted against baseline
- [ ] Checkpoint backward compatibility: load a real historical checkpoint and assert it
      still reconstructs identically. Non-negotiable — this is the promise to `nsosim`.
      *(In-suite proxy landed in PR #85:
      `test_model_roundtrip.TestPreFixCheckpointsStillLoad` — a both-alias state dict
      through `load_model` and bare `load_state_dict(strict=True)`, outputs
      bitwise-equal, plus the decoder-wins-on-disagreement rule; loading a real
      shipped checkpoint stays a release-time check, since the 275 MB downloads do
      not belong in CI.)*

### 7.3 Characterization tests, written just-in-time

For each monolith, **only immediately before decomposing it in Phase 4**. Pin observed
behaviour, bugs included — a characterization test documents what the code *does*, and
anything surprising becomes a §8 ledger entry rather than a silent fix.

Priority order (by lines × inverse coverage × production-reachability):
1. `datasets/sdf_dataset.py` (2,195 lines @ 7%)
2. `reconstruct/main.py` (1,443 lines @ 24%)
3. `train/train_deep_sdf.py` (618 lines @ 10%)
4. `utils.py` (280 lines @ 25%) — partly done in Phase A

### 7.4 Targets

- [ ] Overall coverage 32% → **≥70%**, with **≥90%** on the production API
- [ ] Suite stays under 2 minutes so nobody skips it
- [ ] CI runs it on every PR

---

## 8. Phase 4 — Decompose the monoliths

Standard extract-under-test: pull a coherent piece out,
have the monolith call the extracted function, keep §7.1 green, then unit-test the
extracted piece properly.

- [x] `train_deep_sdf.py`: split the 618-line/2-function structure into setup, epoch loop,
      validation, checkpointing. *(§8.0.D, PR #76 — the orchestrator's four concerns are
      private helpers. `train_epoch` itself stays whole and is §8.0.L.)*
- [x] `sdf_dataset.py`: separate mesh loading, normalization/scaling, registration, SDF
      sampling, caching. *(§8.0.A/B/F, PRs #71, #72, #85 — helpers, readers, reader
      internals and the cache shell. `NSM/datasets/utils.py` is no longer the 2-line stub
      this bullet described; it is the 13 leaf helpers.)*
- [x] `reconstruct/main.py`: separate latent optimization, mesh generation, evaluation.
      *(§8.0.C/E, PRs #74, #79 — `latent_fit.py`, `wandb_logging.py`, `recon_evaluation.py`.
      The **module** split is done; `reconstruct_mesh` and `reconstruct_latent` themselves
      were moved unopened and are §8.0.J and §8.0.K.)*
- [ ] Fold in the stalled API-cleanup plans, which are Phase-4-shaped and should not run
      separately: `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` +
      `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` (issue #3). *(= §8.0.Q, last.)*
- [ ] Close issues #1, #2, #5, #6 as the relevant modules are touched. *(#5 closed by #79,
      #6 by #63. **#1 looks already fixed** — its proposed remedy (scale `xyz` in
      `__getitem__`) is what #69 shipped in PR #72; verify and close, §8.0.G rides it.
      #2 is performance work → §8.3.)*

**A warning about these four checkboxes, 2026-08-26.** Three of them tick at *module*
level and that is what made the remaining work invisible for two slices: the largest
functions in the repo are functions a slice moved verbatim and never opened. Ticking
"`reconstruct/main.py`: separate latent optimization, mesh generation, evaluation" is
true and `reconstruct_mesh` is still 408 lines behind 61 parameters. The slice index
below exists so a decomposition bullet cannot be satisfied by relocation again.

### 8.0 Slice index — scheduled 2026-08-26

§8.0.A–I are executed and keep their statements below. J–Q are scheduled. **Each gets its
own §8.0-style statement as commit 1 of its own slice**, with every claim re-run against
`main` first — writing eleven statements up front is the "size docs to your uncertainty"
mistake `CLAUDE.md` names. What is fixed here is the *order* and each slice's *scope*.

| | Slice | Carries | Why it is where it is |
|---|---|---|---|
| **G** | Observability: logging, not `print` | #58, the root-logger `basicConfig`, 257 prints, verify-and-close #1 | Mechanical and independent, and it touches every file H–M touch. Doing it after them means opening all of them twice. Statement below. |
| **H** | `models/` package | #26 (**High**), #45, #46, #34, the two-`Sine` trap, the VAE missing activation | The largest untouched surface on the production path, and half the consumer's public contract (`TriplanarDecoder`). No pass of any kind has reached it — not Phase 2, not 3, not 4. |
| **I** | `mesh/main.py` | #60, #57 (five sites, one helper), #54's sites there, `create_mesh_adaptive` | Second-largest file in the repo, on the production path, and **never named anywhere in this plan** — §7.3's priority list has four files and this is not one of them. |
| **J** | `reconstruct_mesh` internals | the 61-parameter signature, the interleaved timing plumbing | Seams are already clean: coerce → reference → sample → fit → build → metrics → assemble. |
| **K** | `reconstruct_latent` internals | #75, the 185-line nested `compute_loss`, the hybrid Adam/LBFGS branch | The last unopened production monolith. #75 (cannot chunk its forward pass) is a defect the decomposition has to make expressible. |
| **L** | `train_epoch`'s loss pipeline | the ~270-line batch loop | The statement §8.0.D said this needs, deferred deliberately, now due. |
| **M** | `NSM/utils.py` | #50, the module's remaining undocumented surface | §1.2's exhibit: the file that held the founding bug. Phase A documented the LR path and nothing else. |
| **N** | Phase 2 close + lint gate | the §6 checkboxes, `flake8-docstrings` in `make lint`, `CLAUDE.md` §Architecture rewrite | Must follow G–M — that is where the missing docstrings are — and the lint gate is what stops G–M's accuracy rotting. |
| **O** | v0.3.0 release | the pending Breaking set, setuptools-scm (§10.1), §7.1's GPU note, **`NSM.configs` ships in no wheel** (SCOPE §5), and **two items §8.0.H deferred here by name**: (a) a **combined pre-v0.3.0 config message** — the release adds three required triplanar keys (`padding`, `conv_norm_type`, `conv_activation`), each refused separately, so an old config is fixed one round-trip at a time; one message naming every missing key at once is the `_lr_migration` pattern applied to the set. (b) **`TriplanarDecoder`/`VAEDecoder`'s signature defaults**, still `conv_norm_type="batch"` against the `"layer"` everything trained — unreachable from a config now that the loader requires the key, but reachable by direct construction, and changing a public-stable signature needs the version boundary. | Maintainer-gated timing. Nothing in G–N waits on it. |
| **P** | 0b quarantine + #18 | `train/deprecated/` (876 lines), the `sample_difficulty_lx` port | Maintainer-gated on the nsosim survey, unchanged since Phase 0. |
| **Q** | #3, sigma coordinate space | `BREAKING_CHANGE_PROPOSAL.md` + `SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Last, because it is the one remaining *behaviour* change: it needs a §4-style migration guard and a version boundary, so it wants a release on either side of it. |

Rows that are **not** refactor and are deliberately absent: see §8.3.

Residual items with no row of their own, to ride with the slice that opens the file:
`__getitem__` unification across the two dataset classes (§8.0.F deferred it in writing),
`SDFSamples.__init__` at 169 lines, the #54/#55/#56 class sweeps, and — added by §8.0.G —
`NSM/utils.py`'s `print_gpu_memory`, which now logs and whose name says otherwise. It has
no caller in `NSM/`, in the suite or in `docs/`, so the fix is a rename or a deletion,
both Breaking: **§8.0.M**.

### 8.0 `sdf_dataset.py` decomposition — plan statement (2026-08-22)

The file is 2,644 lines: ~900 of module-level functions, ~1,700 of the two Dataset
classes. Slice A below moves the functions out **verbatim**; restructuring anything
comes later, each slice with its own statement.

**Slice A — target layout (all permanent, nothing transitional):**

- `NSM/datasets/utils.py` — the 13 leaf helpers, no NSM-internal imports:
  `derive_seed`, `mesh_content_key`, `get_rand_uniform_pts`,
  `get_pts_center_and_scale`, `is_zipfile`, `meshfix`, `get_cube_mins_maxs`,
  `get_buffered_cube_mins_maxs`, `unpack_pts`, `unpack_numpy_data`,
  `check_probabilities`, `check_probabilities_sum`, `combine_meshes` (~330 lines).
- `NSM/datasets/mesh_sampling.py` — the two subject-level pipelines,
  `read_mesh_get_sampled_pts` and `read_meshes_get_sampled_pts` (~580 lines);
  imports only from `.utils`.
- `NSM/datasets/sdf_dataset.py` — the two Dataset classes plus a **permanent
  re-import block**: `NSM.datasets` *and* `NSM.datasets.sdf_dataset` are both live
  import paths (`reconstruct/main.py` uses both in adjacent lines; downstream forks
  assumed to as well). The re-import is public API, not a shim — no delete-when.
  The file's top import block stays verbatim: flake8 ignores F401 globally, and the
  unused imports (`pcu`, `numpy_to_vtk`, …) leak into `NSM.datasets` via the
  star-import (ARCHITECTURE.md §5), so removing them would change that namespace.
  `__all__` stays out of this slice — it is the deferred Phase 0 item and a separate
  namespace decision.

**Size budget:** moves are verbatim; allowed growth is two module headers + import
blocks + the re-import block, **~70 lines net**. Beyond that is scope creep.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization tests against the *current* file (§7.3);
3. the move, with the ARCHITECTURE.md §3 ledger rows corrected in the same commit;
4. State update.

**Characterization targets (commit 2)** — what moves and is unpinned today:
`unpack_numpy_data` key spellings and raises; `is_zipfile` unreadable-path fallback;
the probability validators; cube-helper `ValueError`s; reader branches — missing
path → `None`, `get_random=False` (the `"pts"`-not-`"xyz"` spelling), the no-norm
branch (`scale=1`, `center=0`), register-without-mean raise (single raises bare
`Exception`, multi raises `ValueError` — pinned as-is), the deprecated-kwarg print
contract, `include_surf_in_pts` (single asserts correct behaviour; multi asserts
the surface's own points and is `xfail(strict=True)` for #17), and the
`mesh_to_scale` / `scale_all_meshes` / `center_all_meshes` frame math on analytic
spheres.

**Verification per claim:**

| Claim | Verification |
|---|---|
| Verbatim move changes no behaviour | full suite + regression harness green before/after; `git diff --color-moved` shows pure moves |
| Both import paths keep every symbol | new test importing each moved name from both `NSM.datasets` and `NSM.datasets.sdf_dataset`, list frozen in the test |
| No import cycle introduced | `utils.py` imports nothing NSM-internal, `mesh_sampling.py` only `.utils`; ARCHITECTURE.md §2 ast pass re-run |
| Cache keys unaffected | existing `TestHashedParametersChangeTheKey` / `TestCacheRoundTrip` green |
| Pool pickling unaffected | classes stay in `sdf_dataset`; existing `test_multiprocessing_does_not_change_the_data` green |

**Deferred out of slice A, deliberately:** splitting the reader internals
(registration / frame computation / per-surface draws) is slice B — that is where
#69 (`norm_and_scale_all_meshes` in-memory) and #3 (sigma coordinate space) ride
along, and it gets its own statement first. Class-side cache/build decomposition
stays grouped with #19/#27 (see State § Deliberately deferred).

### 8.0.B `read_meshes_get_sampled_pts` internals split — plan statement (2026-08-23)

Slice A moved the readers verbatim; this slice restructures the multi reader's
inside. The single reader stays untouched — it is 80 flat, legible lines, and its
`include_surf_in_pts` block is the *correct* one (#17's trap: any "unification"
risks copying the broken variant over it).

**Target shape (all permanent, nothing transitional).** Three module-private
helpers inside `mesh_sampling.py`, named for the State's own decomposition —
private because they are internals, and so that the `test_import_compat` frozen
namespace list does not change:

- `_register_to_mean(orig_meshes, new_meshes, new_pts, paths, mesh_to_scale,
  mean_mesh, icp_transform)` → the transform used (caller-supplied or computed via
  `combine_meshes` for a list `mesh_to_scale`); applies it to every surface in place.
- `_compute_shared_frame(new_pts, mesh_to_scale, scale_all_meshes,
  center_all_meshes, scale_method)` → `(center, scale)`. Pure — the six-branch
  tangle becomes directly testable against the sphere arithmetic the slice-A
  characterization already asserts.
- `_draw_surface_samples(new_meshes, new_pts, sigma, n_pts, rand_function,
  include_surf_in_pts, uniform_pts_buffer, seed)` → `(rand_pts, pts_surface)`.
  The per-surface SDF loop and the `get_random=False` branch stay inline — coherent
  as they are, and the State does not name them.

**Fixes land *before* the split**, so the split commit is purely
behaviour-preserving (no xfail transitions inside it):

- **#17** — the leaked `new_pts_` in the draw loop. The issue's "determine which of
  the three applies" question is now settled by execution (2026-08-23, sphere pair):
  with `center_pts`/`norm_pts` both False and numeric sigmas — the shape
  `reconstruct_mesh` produces for `scale_jointly=True` models — it **always crashed**
  (`UnboundLocalError`); with any sigma `None` it **always crashed** (`ValueError`,
  `new_pts_` is a leaked list); with centering on and numeric sigmas it produced
  **silently wrong data** — 1580 points where 1874 is correct: the *last* surface's
  pre-normalization vertices appended once per surface, wrong surface *and* wrong
  frame. So the `KNOWN_ISSUES.md` § History entry covers exactly one run class:
  multi-surface calls with centering on — via `reconstruct_mesh`, that is
  `get_rand_pts=True` on a `scale_jointly=False` model. Neither shipped config
  reaches it (`get_rand_pts_recon: false`). Fix: append `new_pts[new_pts_idx]`; the
  slice-A strict-xfail passes unmarked. Extraction then removes the *class* of
  defect — a helper's scope cannot see a leaked binding from another section.
- **#69** — fix by unification, not by patching the in-memory branch:
  `norm_and_scale_all_meshes` computes `self.center`/`self.max_radius` in both
  storage modes (the only difference is reading `new_pts_{i}` npz keys vs the
  in-memory `new_pts` list) and the existing per-batch application in `__getitem__`
  — present in both classes (`SDFSamples.__getitem__`,
  `MultiSurfaceSDFSamples.__getitem__`) and conditioned only on the attributes —
  does the scaling. The mutate-in-place branch is deleted; the buffer is applied by
  construction. Disk-path numerics unchanged. Always crashed → no History entry.

**#3 rides along structurally, not as a fix.** After the split there is exactly one
site where sigma is consumed (`_draw_surface_samples`); its docstring states the
coordinate-space fact — draws happen in whatever frame the meshes are in at call
time: normalized when centering ran, original units otherwise. The breaking change
itself stays with `BREAKING_CHANGE_PROPOSAL.md` / `SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md`
and needs its §4-style migration guard; nothing in this slice moves it.

**Re-verified during scoping, already tracked:** `reconstruct_mesh` passes
`n_pts_random=` to readers whose parameter is `n_pts=`, so the kwarg is swallowed
and the 200,000-point default is used (asked 7, got 200,122 = 200,000 + vertices).
That is #16; its fix belongs to `reconstruct/main.py`, not this slice.

**Characterization added before any change** (commit 2), pinning what the split
touches and slice A left unpinned: the `icp_transform` reuse path (the dataset's
cross-combo contract — pass a transform back in, registration is skipped, the same
frame comes out), the joint uniform-cube draw (`None` in sigma, without
`include_surf_in_pts`), reader-level seed determinism (same seed → identical
draws; different seed → different), and a second #17 strict-xfail for the
uniform-cube + `include_surf_in_pts` combination.

**Size budget:** #17 ≤ +5 code lines; #69 net ≤ +10 (unification should land ≤ 0);
the split ≤ +80 net in `mesh_sampling.py` (three signatures + docstrings; bodies
are moves). Characterization tests are additive and outside the budget. Beyond
this is scope creep.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization additions; 3. #17 fix + History entry +
xfail unmarks; 4. #69 fix + xfail unmark + `KNOWN_ISSUES` § Open removal;
5. the split; 6. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #17 fix appends each surface's own points | slice-A strict-xfail passes unmarked; new uniform-cube xfail passes unmarked |
| #17's affected-run classes are as stated | the executed determination above, recorded in the History entry |
| #69 both halves land together | `TestScaleJointlyInMemory` passes unmarked (asserts the buffered domain; its `raises=KeyError` made a half-fix a plain failure) |
| #69 leaves disk-path numerics unchanged | regression harness green (`test_dataset_cache`, training regression) |
| Split changes no behaviour | full suite + harness green before/after; `git diff --color-moved`; no xfail transitions in the split commit |
| No namespace change | `test_import_compat` frozen list untouched |
| No new import or dependency edge | helpers live in `mesh_sampling.py`; module still imports only from `.utils` |

### 8.0.C `reconstruct/main.py` — fixes and first decomposition slice — plan statement (2026-08-23)

The file is 1,509 lines; the three §8 concerns (latent optimization, mesh
generation, evaluation) live in one module. This slice lands the three issues
filed against the file (#15, #16, #29) plus the in-package instances of #16's
class, then moves the latent-optimization stack and the wandb helpers out
verbatim. **Fixes land before the split**, 8.0.B's pattern: the split commit is
purely behaviour-preserving.

**Target shape (permanent unless marked):**

- `NSM/reconstruct/latent_fit.py` — the latent-optimization stack, moved
  verbatim: the four `reconstruct_latent_*` type-check/setup helpers,
  `reconstruct_latent_preprocess_sdf_gt`, `project_latent`,
  `latent_norm_penalty`, `reconstruct_latent` (~690 lines). Imports torch/wandb/
  `NSM.losses`/`.utils` only — leafward, no cycle. The log *format* has no
  `%(name)s`, so the logger renaming to `NSM.reconstruct.latent_fit` changes no
  output byte.
- `NSM/reconstruct/wandb_logging.py` — `_process_meshes_for_wandb`,
  `prepare_results_for_wandb` (~145 lines). Separate module because
  `reconstruct_mesh` calls it: parking it in an evaluation module would make
  `main` import evaluation and evaluation import `main`.
- `NSM/reconstruct/main.py` — keeps `reconstruct_mesh`, `tune_reconstruction`,
  `get_mean_errors`, `compute_correlation_coefficient`, the module-scope
  `logging.basicConfig` (verbatim — its removal is #58's business), and a
  **permanent re-import block**: `NSM.reconstruct` (star-import namespace) *and*
  `NSM.reconstruct.main` are both live import paths
  (`test_predictive_validation.py` monkeypatches `NSM.reconstruct.main`;
  kneepipeline and the trainers use the package path; forks assumed to use
  both). Every name currently importable from either path stays importable from
  both, `_process_meshes_for_wandb` included.

**Deferred out of this slice, deliberately:** moving the evaluation trio
(`get_mean_errors`, `tune_reconstruction`, `compute_correlation_coefficient`)
to its own module. It buys nothing now — `main` still imports wandb for
`reconstruct_mesh`'s own logging, so #5 is not advanced — and costs either an
import cycle (`main` ⇄ evaluation) or breaking `NSM.reconstruct.main`'s
namespace. It becomes worth doing as part of #5 (making wandb optional), which
needs its own statement. Also out: #58 (ungated prints), #56 (the
`sigma_rand_pts` default is 0.001 in `reconstruct_mesh` and 0.01 in
`get_mean_errors` — that divergence is #56's class), `reconstruct_latent_S3.py`
(deferred research, #35).

**The fixes, with the decisions of record:**

- **#15 — unify the readers on `"pts"`.** `read_mesh_get_sampled_pts` writes its
  draw to `results["xyz"]`; everything first-party reads `"pts"`
  (`reconstruct_mesh`, `reconstruct_latent_S3`, and both dataset classes via a
  two-branch probe). Fix in `mesh_sampling.py`: the single reader writes the
  draw to `"pts"` on every path, keeping `"xyz"` as an alias **bound to the same
  array** — *transitional*, delete when the 0b fork survey confirms no external
  `["xyz"]` readers, or at v0.3.0, whichever first. The two probe workarounds in
  `sdf_dataset.py` (`:608`, `:1331`) collapse to `result_["pts"]` — removing the
  existing workarounds, not adding a third. Single-object `get_rand_pts=True`
  always crashed → **no History entry**; cached content is unchanged (same
  array, new key) → cache tests prove it.
- **#16 — honour `n_pts_random`.** `reconstruct_mesh` forwards
  `n_pts_random=` to readers whose parameter is `n_pts=`; both readers'
  `**kwargs` swallow it and the 200,000-point default runs. Fix: pass
  `n_pts=n_pts_random` (the multi path already listifies it). This is the
  accepted-and-ignored shape the memory says to *delete* — but the two deleted
  precedents (`padding`, `center`/`scale`) would have changed every default run
  if honoured, and honouring this one is measured to touch only
  multi-object `get_rand_pts=True` runs (never shipped; single always crashed,
  #15), while the parameter's intent is unambiguous and config-plumbed
  (`n_pts_random_recon` → three layers). Honouring changes those runs' draw
  from 200,000/surface to the requested value → **History entry**. Ride-along:
  the regression-harness class docstring that documents the swallowed argument
  (and the measured 400,688) is rewritten, and the seeded tests get ~1000×
  smaller draws, which is the harness cost §7.1 blamed on this bug.
- **#16's class, enumerated** (AST scan, every function, parameter name absent
  from body — run 2026-08-23, both repos' consumers checked): in-package,
  `compute_recon_loss(n_samples_assd)` — its implementing call is commented
  out, no caller passes it → **deleted**, and the `batch_size_latent_recon`
  plumbing — `get_mean_errors` forwards it into `reconstruct_mesh`'s `**kwargs`,
  so **every validation pass prints the deprecation warning at itself** →
  parameter deleted from `get_mean_errors`/`tune_reconstruction` and the
  `train_deep_sdf.py:238` call site; the shim in `reconstruct_mesh` stays (it is
  the migration surface for external callers). Out of package, recorded not
  fixed: `models/` instances are #20's annotated parameter half;
  `train_epoch(return_loss, verbose)` ×4 belongs to the trainer slice;
  `interpolate.update_positions(verbose)` and S3's `epsilon` noted for their
  modules' passes; `get_learning_rate(epoch)` on Constant and `__exit__(args)`
  are interface conformity, not defects.
- **#29 — raise by name instead of the fake-success dict.** New
  `NoZeroLevelSetError(RuntimeError)` in `main.py`, raised where the mean mesh
  comes back `None`; message names the state (zero-latent SDF has no sign
  change at `n_pts_per_axis_mean_mesh` resolution) and the two causes (model
  not trained far enough; grid too coarse). `get_mean_errors` catches it
  per-path and records what it always recorded — nan metrics — plus a **NaN
  latent row** to `Regress` (today it hands over the zero vector, so the r²
  is computed against fabricated latents; NaN rows make it an honest nan and
  keep path↔latent alignment). Training validation therefore still survives a
  model that has not learned a sign change; the direct caller (kneepipeline)
  gets a named error instead of a `KeyError` at `result["center"]`. Pre-fix
  runs recorded zero latents and nan metrics as if fitted → **History entry**
  (detection: latent exactly all-zero *and* metrics nan).
  `TestDecoderWithNoZeroLevelSet` is rewritten around the raise; its strict
  xfail retires with this commit. House precedent: #48's barrier, #41.

**Size budget:** #15 ≤ +5 in `mesh_sampling.py`, net negative in
`sdf_dataset.py`; #16 is a 2-line call change; the class sweep is net negative;
#29 ≤ +45 (exception, raise, catch-and-record); the split ≤ +80 net (module
headers, import blocks, re-import block). Characterization tests are additive
and outside the budget.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization — freeze both namespaces
(`NSM.reconstruct`, `NSM.reconstruct.main`) as a list-pinned import test, pin
the unpinned movers (`prepare_results_for_wandb`'s filtering,
`project_latent`, `latent_norm_penalty`'s quadratic/huber branches, the
type-check raises, `reconstruct_latent_get_lr_update_freq` arithmetic), pin the
self-inflicted deprecation print (capsys), add strict-xfails for #15 (single
sampled path returns) and #16 (the request is honoured); 3. #15 fix + xfail
unmark; 4. #16 fix + History + harness-doc rewrite; 5. the in-package class
sweep; 6. #29 fix + History + test rewrite (namespace list gains the exception
— deliberate, in the same commit); 7. the split + ARCHITECTURE.md §3 ledger
rows; 8. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #15 unification changes no cached data | `test_dataset_cache` green untouched; alias asserted `result["pts"] is result["xyz"]` |
| #15 single sampled path works | the strict-xfail passes unmarked |
| #16 honoured | strict-xfail passes unmarked: request 200, receive 200 + appended surface points, not 200,000 |
| #16 touches only never-shipped runs | both shipped configs `get_rand_pts_recon: false` (re-checked); single path always crashed (#15's evidence) |
| deprecation print was self-inflicted | capsys pin in commit 2, deleted in commit 5 when the print stops |
| #29 keeps validation alive | new test: `get_mean_errors` over a no-zero-level-set decoder returns nan metrics and nan r², raises nothing |
| split changes no behaviour | full suite + harness green; `git diff --color-moved` pure moves; no xfail transitions in the split commit |
| both import paths keep every name | frozen namespace lists, asserted on both paths, written in commit 2 *before* anything moves |
| no new import cycle | `latent_fit`/`wandb_logging` import nothing from `.main`; ARCHITECTURE.md §2 ast pass re-run |

### 8.0.D `train/train_deep_sdf.py` — the five filed issues and the orchestrator split — plan statement (2026-08-23)

The file is 649 lines in two functions: `train_deep_sdf` (~223, the orchestrator) and
`train_epoch` (~373, the epoch loop). The five issues filed against it (#28, #42, #49,
#52, #59) land first, then the trainer-side instances of #16's class, then the
orchestrator's concerns split into module-private helpers. **Fixes land before the
split** (8.0.B's pattern): the split commit is purely behaviour-preserving.

**Target shape (all permanent, nothing transitional).** No new modules — module-private
helpers inside `train_deep_sdf.py`, 8.0.B's pattern, so the `NSM.train` namespace does
not change:

- `_resume_from_checkpoint(config, model, latent_vecs, optimizer)` — the resume block
  (both checkpoint loads plus the two refusal guards).
- `_schedule_free_eval_warmup(model, latent_vecs, data_loader, config, epoch)` — the
  eval-mode warm-up. #42's fix site; extraction gives its test a callable seam.
- `_run_validation(config, model)` — the `get_mean_errors` kwarg block.
- `_save_checkpoint(config, epoch, model, latent_vecs, optimizer, sdf_dataset)` — the
  three save calls.
- `train_deep_sdf` keeps setup (defaults, validation, wandb init, dataloader, latents,
  optimizer) and the epoch loop, becoming a legible orchestrator. **`train_epoch` stays
  whole** — it *is* the epoch loop §8 names; its internal loss-pipeline decomposition
  would need its own statement and is not this slice.

**The fixes, with the decisions of record:**

- **#42 — the schedule_free warm-up forwards a raw dataloader item.** `model(batch)`
  where `batch` is `(sdf_data, indices)`; every schedule_free run dies at its first
  checkpoint or validation epoch (verified 2026-08-21 with real schedulefree 1.4.1).
  Fix: unpack the way `train_epoch` does — xyz reshaped, indices repeated per sample,
  latents looked up (variational sampling included), chunked by `batch_split` (the
  knob exists because full batches do not fit; a warm-up that skips it OOMs on exactly
  the configs that need it), `model(inputs, epoch=epoch)` under the existing
  `no_grad`. Always crashed → **no History entry**. Test strategy: `schedulefree` is
  not installed in `nsm-dev`, and the crash is in the trainer's forward, not in
  schedulefree — the pin monkeypatches `NSM.utils.schedulefree` with a stub
  `AdamWScheduleFree` (AdamW plus `train()`/`eval()` no-ops) and trains through a
  checkpoint epoch. The stub exercises the trainer's code path, not schedulefree's
  numerics — KNOWN_ISSUES §1 already records those need retuning.
- **#49 — `resume_epoch == 1` silently skips epoch 1 without resuming.** The resume
  guard reads `> 1` while the loop starts at `resume_epoch + 1`. Fix: one boundary —
  the resume block loads for `resume_epoch >= 1`, so `1` means "epoch 1's checkpoint
  is complete; continue at 2" (the issue's load option, not the raise option:
  `resume_epoch` uniformly names the last completed epoch). `resume_epoch=0`
  unchanged. Pre-fix `resume_epoch=1` runs completed, training from scratch for
  `n_epochs − 1` epochs with nothing loaded → **History entry** (detection: the run
  was launched with `resume_epoch=1`).
- **#59 — logged latent-norm stats are the last batch's; the positional back door.**
  `step_mean_vec_length`/`step_std_vec_length` use `=` where every surrounding
  accumulator uses `+=`, then divide by `len(data_loader)` — logged values wrong by
  ~×n_batches on every run since inception (issue's verification: true mean 0.0107,
  logged 0.0053, ×n_batches matches). Fix: `+=`. Gradients and weights untouched;
  wandb-only → **History entry** scoped to logged metrics.
  `add_plain_lr_to_config` loses `idx_model`/`idx_latent` — the positional override
  the Aug-2026 fix exists to forbid, whose only caller is a test asserting
  deliberately swapped labels; parameters and test deleted together. CHANGELOG
  Breaking (the parameters).
- **#16's class, trainer instances.** `train_epoch(return_loss, verbose)` — both
  AST-confirmed unread (the body reads `config["verbose"]`; `log_dict` is returned
  unconditionally). Deleted from the live trainer's signature and its one call site.
  multi_head's and the deprecated trainers' copies **stay**: multi_head belongs to
  #51's repair, `deprecated/` to the 0b quarantine — deleting parameters from a
  module ruled dead is churn. Also dead: the `if "resume_epoch" not in config`
  re-default at `:86` (the `setdefault` at `:57` precedes it). CHANGELOG Breaking
  (`train_epoch` signature).
- **#28 — `train_deep_sdf` returns nothing.** Returns the per-epoch history: one
  entry per epoch, `{**log_dict` (the exact wandb payload, validation metrics
  included on val epochs)`, "epoch", "lrs": {group_name: lr}, "targets":
  {group_name: target}, "latent_norms": [per-subject]}`. LRs read *after*
  `train_epoch` (the rates the epoch actually ran with), keyed by group `name` with
  the name→target map alongside — several groups can share a target, so target alone
  cannot key the dict. The wandb payload stays byte-identical (extras go on the
  history entry, not into `log_dict`). The harness's `recording_train_epoch` wrapper
  is **deleted**; `run_training` maps the return value into the record shape the
  baselines already pin, so the strict-xfail passes unmarked and the regression suite
  becomes a consumer of the public contract. CHANGELOG (non-breaking: `None` → list).
  Ordered after #59 so the history's `mean_vec_length` is honest from birth.
- **#52 — `mesh_names` is persisted as ground truth and can be silently wrong.** The
  names move to where the ordering is defined: `MultiSurfaceSDFSamples(...,
  mesh_names=None)`, validated at construction against the per-subject surface count
  (`self.n_meshes`), stored on the instance. The trainer, at its existing
  `mesh_names` validation site — entry, not save time: the same check, hours
  earlier, and `save_model_params` then writes what was checked — **adopts** the
  dataset's names when the config has none and **raises** when both exist and
  disagree. Deliberately *not* in the cache key: names do not change sampled data,
  so unlike #19's omissions this one is correct. Single-surface `SDFSamples` is
  skipped: one surface has no ordering to mis-declare. Config-only names with a
  dataset that carries none keep today's behaviour — identity has to come from the
  user; the fix moves the declaration next to the ordering rather than inventing
  one. Root CLAUDE.md § Multi-Surface Config gains the dataset-side recommendation.
  CHANGELOG Added.

**Deferred out of this slice, deliberately:** `train_epoch`'s internal decomposition
(needs its own statement); everything multi_head (#51) and `train/deprecated/` (0b
quarantine, including the §6.1 `sample_difficulty_lx` port); wandb-optional (#5 —
`train_deep_sdf` still imports and calls wandb, and `log_latent` still builds
`wandb.Histogram`s); the two documented-not-fixed Open entries (`enforce_minmax`
prediction-clamp — SCOPE §2.2 config work; `grad_clip` latent-clipping experiment).

**Size budget:** #42 ≤ +15; #49 ≤ +5; #59 net negative in `NSM/`; the sweep net
negative; #28 ≤ +25 in the trainer and net negative across the repo (the harness
wrapper goes); #52 ≤ +45 across both files; the split ≤ +80 net (four signatures +
docstrings; bodies are moves — and the file's two public functions gain its first
docstrings). Characterization tests are additive and outside the budget.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization — freeze the `NSM.train.train_deep_sdf` public
namespace (frozen-list import test), pin `resume_epoch=0` and `>1` behaviour, add
strict-xfails for #42 (a stubbed schedule_free run survives its first checkpoint
epoch), #49 (`resume_epoch=1` resumes from epoch 1's checkpoint), #59 (logged
`mean_vec_length` matches the true epoch mean), #52 (dataset-carried names are
adopted/validated) — #28 is already pinned
(`TestTrainerContract::test_train_deep_sdf_returns_its_history`); 3. #42 fix +
unmark; 4. #49 fix + History + unmark; 5. #59 fix + History + CHANGELOG + the
override-test deletion; 6. the trainer sweep + CHANGELOG; 7. #28 fix + harness
wrapper deletion + unmark + CHANGELOG; 8. #52 fix + CLAUDE.md + CHANGELOG + unmark;
9. the split; 10. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| #42's crash is the trainer's, not schedulefree's | the strict-xfail reproduces the `TypeError` with the stub optimizer — no schedulefree import involved |
| #42 fixed means the warm-up runs the real forward | the test trains a stubbed schedule_free run through a checkpoint epoch to completion |
| #49's two boundaries agree | resume test: train, resume with `resume_epoch=1`, assert the epoch-1 checkpoint is loaded (latents match its file) and epochs [2..n] run |
| #59's `+=` restores the true mean | the logged value is asserted against a mean computed over `latent_vecs` directly — the issue's ×n_batches check, as a test |
| the sweep deletes only unread parameters | the AST scan (parameter name absent from body) re-run in-commit; full suite green |
| #28's history is what wandb would have seen | harness baselines (loss trajectory, per-group LR, latent norms) stay green with the wrapper deleted — the same numbers now flow through the return value |
| #52 catches a wrong declaration | dataset names ≠ config names raises at entry; dataset names + no config names adopts |
| split changes no behaviour | full suite + harness green; `git diff --color-moved` pure moves; no xfail transitions in the split commit |
| namespace unchanged | the frozen-list import test written in commit 2, untouched by the split |

### 8.0.E wandb-optional (#5) and the evaluation-module split — plan statement (2026-08-24)

Every claim below was re-run on `main` at `aae5979` (2026-08-24) before being written.

**The defect, reproduced.** `import wandb` sits at module top in six non-deprecated
modules — `reconstruct/main.py`, `reconstruct/latent_fit.py`,
`reconstruct/wandb_logging.py`, `reconstruct/reconstruct_latent_S3.py`,
`train/train_deep_sdf.py`, `train/train_deep_sdf_multi_head.py` — so with wandb
absent, `import NSM.reconstruct` **and** `import NSM.train` die with #5's exact
`ModuleNotFoundError` (subprocess with a blocked `wandb`). wandb appears nowhere in
`pyproject.toml` (`dependencies = []`), so the consumer's two-symbol surface
(`TriplanarDecoder`, `reconstruct_mesh` — SCOPE §3) depends at import time on a
package nothing declares. Every wandb *call* is behind an explicit request
(`log_wandb` / `use_wandb` / `config["log_latent"]`) except one: `get_mean_errors`'
metric tail builds `wandb.Histogram(item)` per key unconditionally (its only guard is
`except ValueError`), and the trainer's `_run_validation` never passes `log_wandb` —
so training validation needs wandb at **runtime** even with `use_wandb=False`.

**The #5 fix (permanent; schedulefree's pattern, `NSM/utils.py`):** per-module guarded
import — `try/except ModuleNotFoundError` → `wandb = None`. The module attribute
survives as a monkeypatch seam (as `NSM.utils.schedulefree` is for #42's stub). At
each explicit-request gate, raise `ImportError` by name when `wandb is None`:
`reconstruct_mesh`'s `log_wandb` block, `tune_reconstruction`'s login,
`get_mean_errors`' init, `reconstruct_latent`'s log block, S3's log block,
`train_deep_sdf`'s login, multi_head's login (the import line and that one site only —
the rest of the module is #51's), `log_latent`'s histogram block, and
`prepare_results_for_wandb`'s entry (for external callers who bypass the in-repo
gates). The one **skip-not-raise**: `get_mean_errors`' histogram tail sets
`hist = None` when wandb is `None` — nothing requested wandb there, and raising would
break exactly the run #5 protects (training validation without wandb). With wandb
installed nothing changes — payload and #28 history byte-identical — and wandb-absent
runs could never start before (import crash) → **no History entry**.

**The split (permanent).** 8.0.C deferred moving the evaluation trio because it cost
"an import cycle (`main` ⇄ evaluation) or breaking `NSM.reconstruct.main`'s
namespace". Re-scoped by execution, the cycle is smaller than feared:
`recon_evaluation.py` already exists, is leafward (`logging`/`numpy`/`.utils` only),
and the trio's only `main` symbols are `reconstruct_mesh` and `NoZeroLevelSetError`
(AST free-variable scan; the rest is `os`/`np`/`fnmatch`/`Regress`/`logger`/`wandb`).
So `tune_reconstruction`, `get_mean_errors` and `compute_correlation_coefficient`
move **verbatim** into the existing `recon_evaluation.py` — no new module — and the
cycle reduces to one **call-time** import inside `get_mean_errors`
(`from .main import reconstruct_mesh, NoZeroLevelSetError`), stated in place.
Call-time is load-bearing, not cosmetic: `test_predictive_validation.py` monkeypatches
`NSM.reconstruct.main.reconstruct_mesh`, and a call-time lookup keeps that seam alive.
`main.py`'s existing `from .recon_evaluation import compute_recon_loss` extends to
re-import the trio, so both frozen namespaces (`NSM.reconstruct`,
`NSM.reconstruct.main`) keep every name — `test_reconstruct_import_compat.py`'s lists
untouched — and the trainer's package-path import of `get_mean_errors` is unaffected.
The trio leaves `main.py`'s module `logger` for `recon_evaluation.py`'s; the log
format has no `%(name)s` (8.0.C's argument), so no output byte changes. After the
move, `main.py` holds `reconstruct_mesh` + `NoZeroLevelSetError` + the re-import
surface: §8's `reconstruct/main.py` bullet (latent optimization / mesh generation /
evaluation) is complete.

**Amendment (2026-08-24, caught before the split commit):** the docs-reference
checker turned the trio move red, and what it surfaced changes the split. SCOPE §2's
dead table (maintainer-approved disposition, 2026-08-22) rules `tune_reconstruction`
and `compute_correlation_coefficient` **dead**, deletion deferred to exactly this
pass ("Phase 4 decomposition of `reconstruct/main.py`"). Moving them would have made
dead code look load-bearing in a fresh module and left the ruling's venue pointing at
a completed event. So both are **deleted** per the standing ruling, as their own
commit between the #5 fix and the split: CHANGELOG Breaking; the frozen namespace
lists narrow by two names (the deliberate, changelogged decision the freeze exists to
force); SCOPE's two table rows retire with a pointer. Zero callers re-verified
2026-08-24 — the only references anywhere were the two defs, the frozen lists, this
branch's own `tune_reconstruction` raise-site test (deleted with it; raise sites nine
→ eight), and the SCOPE rows. The split then moves `get_mean_errors` **alone**, and
SCOPE's `main.get_mean_errors` citation moves with it.

**Deferred out of this slice, deliberately:** packaging (`dependencies = []` is
repo-wide, not wandb's; declaring extras is its own decision), #58
(`logging.basicConfig`), S3 beyond its import line (#35), multi_head beyond its
import line (#51), `train/deprecated/` (0b).

**Size budget:** #5 ≤ +55 across the six modules (six guarded imports, nine two-line
raises, one histogram guard); the split net ≤ +15 in `recon_evaluation.py`
(call-time import + cycle note + `Regress`/`fnmatch` imports), `main.py` net ≈ −270.
Characterization tests are additive and outside the budget.

**Sequence** (one commit each, suite green at every step):
1. this statement; 2. characterization — a single-subprocess wandb-blocked probe
(strict-xfail for #5: `import NSM.reconstruct` and `import NSM.train` both succeed
and the sentinels are `None`), and a wandb-present pin: `get_mean_errors`' `_hist`
values are `wandb.Histogram` (stays green across the fix); 3. #5 fix + per-site
raise tests + the no-wandb validation test + xfail unmark + CHANGELOG (closes #5 on
merge); 4. the split + ARCHITECTURE §3 ledger rows; 5. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| both packages import without wandb | the subprocess probe passes unmarked; asserts the sentinel is `None` on `main`, `latent_fit`, `train_deep_sdf` |
| nothing changes with wandb installed | full suite + regression harness green at every commit; the histogram pin green across the fix |
| every explicit request fails loudly | per-site sentinel-`None` tests, `pytest.raises(ImportError)` naming wandb |
| training validation survives without wandb | `get_mean_errors` over the #29 fixture with sentinel `None`: nan metrics, `_hist` `None`, no raise |
| the split changes no behaviour | full suite + harness green; `git diff --color-moved` pure moves; no xfail transitions in the split commit |
| both namespaces keep every name | `test_reconstruct_import_compat.py` frozen lists, untouched |
| the monkeypatch seam survives the move | `test_predictive_validation.py` green, untouched |
| no module-level cycle | `recon_evaluation.py`'s top imports stay `.utils`-only; the `main` import is call-time, stated in place; ARCHITECTURE §2 ast pass re-run |

### 8.0.F Class-side cache/build split + #19 + #27, one migration release — plan statement (2026-08-24)

Every claim below was re-run on `main` at `8ae0081` (2026-08-24) before being written.
Suite baseline: 664 passed, 1 skipped, 12 xfailed — six of the xfails are #19's
(`TestUnhashedParametersCollide` ×5, `TestReferenceMeshHashing` ×1), three are #27's
(`TestAliasedCheckpointEntries`).

**Why one slice.** Both fixes force a downstream migration — any change to the cache
key orphans every cached `.npz`, and #27 changes the checkpoint format in both
directions — so they land together and **ship in the same release** (the State's
"Deliberately deferred" argument). Whether that release is v0.3.0 or a later cut is
still the maintainer's timing call; the only constraint this adds is that no release
boundary falls between the two fixes. The class-side decomposition rides with them
because #19 rewrites exactly the code the decomposition reorganizes.

**The defects, re-verified.** #19: `get_hash_params` omits `mesh_to_scale` and
`uniform_pts_buffer` (and `subsample`'s only content effect is index padding, below);
the multi list carries an unexplained `False` literal and `create_hash` gives position
meaning (paths inserted in reverse); a `Mesh`-valued `reference_mesh` hashes by memory
address; mesh *content* is nowhere in the key. #27, probed today on a small
`VAEDecoder`: `state_dict()` emits 16,020 elements for 10,024 parameters,
`layers.0.weight` and `decoder.0.weight` share one storage, and — decisive for the fix
— `parameters()`/`named_parameters()` already deduplicate (12 tensors, Adam sees 12),
so removing the duplicate registration cannot touch optimizer state or resume.
`forward` runs through `self.decoder`; nothing else in `NSM/` reads `.layers`.

**The #19 fix (permanent).** Four decisions of record:

- **The key becomes a canonical named mapping.** `create_hash(loc_mesh) -> str` keeps
  its signature but hashes a `{name: value}` dict (canonically serialized, one shared
  implementation for both classes); `get_hash_params` returns that dict — no external
  callers (re-grepped: tests only). Position stops carrying meaning — the LR bug's
  shape — and the `False` literal and the reversed insertion die with it. A
  `cache_format` version entry makes the *next* content-affecting change one integer
  bump instead of a fresh #19; SCOPE §4's cache row turns "Versioned? Yes" (and drops
  its stale `.h5` mention — no h5 path exists in `datasets/`).
- **Coverage:** `mesh_to_scale` and `uniform_pts_buffer` enter the key.
  `joint_scale_buffer` stays out **permanently**, settled: post-#69 it never touches a
  cached byte (`norm_and_scale_all_meshes` stores the frame, `__getitem__` applies it)
  — same class as `mesh_names`. `subsample` also stays out, by the next bullet.
- **Index decoupling instead of keying on `subsample`.** The cache stores the *raw*
  per-sign index sets; the repeat-padding (`samples_per_sign // n + 1`) moves to the
  draw site (`__getitem__`, both classes, both storage modes), sized by the subsample
  in force. `subsample`'s only content effect is that padding (`save_data_to_cache`'s
  key list read against `sdf_pos_neg_idx`), so cached bytes stop depending on it and
  the measured stale-padding defect (4.4×/1.6× interior under-representation) dies
  with the coupling rather than being keyed around. Batch size is a serving parameter;
  forcing a full resample when it changes would be wrong in the other direction. For
  an unchanged subsample the padded array `randperm` sees is byte-identical to today's
  cached one, so batches are bit-identical; `sdf_pos_neg_idx`'s raise/empty semantics
  (#41, `TestEmptySignedSamples`) are unchanged. The issue's reload-guard item
  dissolves: the guard was only ever needed against stale padding.
- **Identity routing — fix the class, not the instance.** The class is "an identity in
  the key that is not content-stable". Every path in the key contributes
  `(path, st_size, st_mtime)`; every `Mesh` object contributes a digest of its
  geometry (point/face bytes); an int or list `reference_mesh` is resolved to the
  underlying path(s) *first* — today the raw int is hashed, so reordering
  `list_mesh_paths` re-aims the reference while the key stands still, the same defect
  one level up. Full-bytes hashing (`mesh_content_key`) is rejected for the key: it
  would read every mesh on every construction, cache hits included; stat is free and
  catches the in-place-edit case the issue names. The seed's bytes key stays what it
  is — build-time only, meshes already in hand.

Old caches never hit the new keys: one regeneration, no corruption, and regenerated
data is identical when seeded (History §3). No migration guard is possible or needed —
keys are opaque, a legacy file is indistinguishable from another config's, and serving
wrong data is exactly what stops happening. CHANGELOG tells users the one-time cost
and that stale cache directories are reclaimable disk. The three KNOWN_ISSUES Open
entries retire into **one History bundle entry** (issue #19's own instruction), and
the defect-describing docstrings (`get_hash_params`, `uniform_pts_buffer`, the class
Notes) are corrected in the same commits.

**The #27 fix (permanent, hook included).** `self.decoder = nn.Sequential(...)` stays
the single registration — it is what `forward` calls; the construction list stops
being a `ModuleList`. A load-time state-dict pre-hook registered on `VAEDecoder`
*itself* drops incoming `<prefix>layers.*` aliases — on the module, not in
`loader.py`, because the consumer's documented path is a bare `model.load_state_dict`
(SCOPE §4) and must strict-load old checkpoints too. Where the two aliases disagree
(checkpoint surgery, the issue's case), `decoder.*` wins — the same winner as today,
where registration order applies it last. **Deliberately permanent, not a
`_lr_migration`-style delete-when module:** all three shipped model releases are
pre-fix checkpoints, so a delete-when condition would be a promise to break them;
~15 lines in `triplanar.py`, reasoned in place. The reverse direction cannot be
shimmed — a post-fix checkpoint fails in pre-fix NSM with `Missing key(s)` — and
CHANGELOG says so. **No History entry:** the aliases share one storage, so results
were never wrong; this costs disk, not accuracy. Re-exporting the shipped checkpoints
(halving the 275 MB downloads) is follow-on coordination with the model releases, not
this slice. Commit 5 also closes §7.1's open **save/load round-trip** box — `main`'s
annotations (PR #84) name it the venue, since #27 changes the format being
round-tripped: save a post-fix model, reload via `load_model` *and* bare
`load_state_dict`, assert the forward bitwise-identical to the in-memory original
(`test_loader.py`'s existing round trip loads but never compares outputs). §7.2's
checkpoint-compat promise is proxied in-suite by the both-alias test; loading a real
shipped checkpoint stays a release-time check (also per #84's annotation).

**The split (permanent).** The two `get_sample_data_dict` bodies are the same
~150-line shell twice — hit path (bad-zip deletion, load+unpack, layout upgrade,
resave), miss path (build), store-mode coercion — with class-specific innards. Target:
the orchestrator lives once, in `SDFSamples`; the class-specific parts become private
hooks — `_build_subject(loc_mesh, ...)` (the combos loop) and
`_upgrade_cached_layout(data, ...)` (single: `pos_idx` backfill; multi: overlap pass +
index-range guard, keeping the delete-and-rebuild semantics). Private, so the frozen
namespace lists do not change (they are hasattr-based; additions don't trip them,
deletions do). `__getitem__` stays per-class — the per-surface draw is not this
slice's business beyond the pad-at-draw line. Commit 2 also *determines* whether a
subject passed as an in-memory `Mesh` (the `isinstance(..., (str, Mesh))` branches)
builds end to end at all: if yes, its key identity routes through the geometry
digest; if it never worked (a #67-shaped discovery), it is filed and pinned, and the
routing covers what runs.

**Deferred out of this slice, deliberately:** `find_hash`'s recursive first-match,
cross-date search (unchanged, now pinned); content-only keys that survive dataset
moves (paths stay in the key; a move regenerates, as today); `__getitem__`
unification; any cache-file layout change beyond unpadded indices (same npz keys,
same spellings); re-exporting shipped checkpoints; multi_head (#51), `deprecated/`
(0b).

**Size budget:** #19 key rewrite ≤ +60 net in `sdf_dataset.py` (identity routing +
named key; the literal, the reversed insertion, and the KNOWN-DEFECTS docstring
blocks all die); index decoupling ≤ +10 net; #27 ≤ +25 in `triplanar.py` (hook +
in-place reasoning, −2 registration); the split ≤ +20 net with the expectation of
negative (one duplicated shell deleted; hook signatures + docstrings added).
Characterization tests are additive and outside the budget.

**Sequence** (one commit each, suite green and lint clean at every step):
1. this statement; 2. characterization — pin the unpinned cache-hit machinery
(bad-zip deletion, single's `pos_idx` backfill + resave, multi's overlap-upgrade
resave, index-range delete-and-rebuild, store-mode coercion, started-loading log),
run the Mesh-subject determination, and add a strict-xfail: an in-place mesh edit
must change the key (#19 (b), the Open entry's "not currently pinned"); 3. #19 key
rewrite + unmarks (mesh_to_scale, uniform_pts_buffer, collision-reuse,
reference-mesh, in-place edit) + History bundle entry + Open retires + CHANGELOG +
SCOPE §4 row; 4. index decoupling + the equal-pos-neg unmark + the subsample-key
xfail rewritten to its true form (cached bytes byte-equal across subsamples — the
old premise inverted into a plain assertion); 5. #27 fix + hook + tests + the §7.1
save→load→bitwise round-trip + three unmarks + CHANGELOG + Open retires; 6. the
split + ARCHITECTURE §3 ledger rows; 7. State update (ticks §7.1's round-trip box).

**Verification per claim:**

| Claim | Verification |
|---|---|
| the key covers mesh_to_scale / uniform_pts_buffer | both strict-xfails pass unmarked; the colliding-runs xfail passes unmarked |
| an in-place mesh edit changes the key | the commit-2 strict-xfail passes unmarked |
| a Mesh reference hashes by geometry, stably | `TestReferenceMeshHashing` xfail passes unmarked; the path-string test stays green |
| the named key still covers what it covered | `TestHashedParametersChangeTheKey` (14 params) green untouched |
| cached bytes no longer depend on subsample | commit-4 assertion: builds differing only in subsample write byte-equal `.npz` |
| equal_pos_neg survives a subsample change | its strict-xfail passes unmarked |
| unchanged subsample ⇒ bit-identical batches | training regression baselines + `test_reloaded_items_match_the_freshly_built_ones` green untouched |
| pad-at-draw keeps #41's raise semantics | `TestEmptySignedSamples` green untouched |
| joint_scale_buffer touches no cached byte | `TestScaleJointlyInMemory` + cache round-trip green untouched |
| #27 single registration | the three strict-xfails pass unmarked |
| old checkpoints load, both entry paths | new test: both-alias state dict loads via `load_model` *and* bare `model.load_state_dict(strict=True)`; outputs bitwise-equal |
| an edit to a surviving key takes effect | `test_editing_a_checkpoint_by_key_must_take_effect` passes unmarked |
| a saved post-fix model reloads bitwise | the §7.1 round-trip: save → reload via both entry paths → forward bitwise-equal to the in-memory original |
| optimizer/resume untouched by #27 | probe (run 2026-08-24): `parameters()` yields each tensor once, aliases share storage — the optimizer never saw the duplicate; resume tests green |
| split changes no behaviour | full suite + harness green; no xfail transitions in the split commit |
| namespace unchanged | frozen lists on both import paths untouched |

### 8.0.G Observability: logging, not `print` — plan statement (2026-08-26)

Every count below was measured on `main` at `995a59d` (2026-08-26) by AST scan, not grep:
a `print` is "gated" when it sits under an `if` whose test mentions `verbose`/`log_`.

**The defect, measured.** `NSM/` outside `deprecated/` makes **257** `print` calls — 189
gated behind `verbose`, **68 ungated**. Three modules define
`logger = logging.getLogger(__name__)`; only two ever call it (21 calls between
`recon_evaluation.py` and `latent_fit.py`), and `reconstruct/main.py` defines a logger it
never uses. Meanwhile `reconstruct/main.py:44` calls `logging.basicConfig(...)` **at module
scope**, which reconfigures the **root logger of the host process** — and because
`reconstruct/__init__.py` star-imports `.main`, it fires on any `import NSM.reconstruct`,
invisibly, in the consumer's process. `ARCHITECTURE.md` §4 calls it the highest-value
single cleanup in the graph and it is still there.

So #58 is not "gate the prints" — most already are. It is that the library owns its
output stream and its host's logging configuration, and the caller owns neither.

**Why this is worth a slice rather than a ride-along.** It touches almost every file
H–M open. Sequenced after them, each of those files gets opened twice and each of their
diffs carries unrelated print churn.

**The consumer constraint, verified 2026-08-26 — and it cuts our way.**
`kneepipeline/steps/run_nsm.py:311-342` runs each NSM fit in a subprocess with
`capture_output=True`, then **parses the last line of stdout as JSON**
(`json.loads(stdout_lines[-1])`), and on failure surfaces `result.stderr[-1000:]`. Two
consequences: (1) anything NSM prints to stdout after that JSON line breaks the consumer,
so stdout is a contract surface, not a scratchpad; (2) the consumer passes
`verbose=True` (`:211`) and then **throws the output away** — it is captured, not shown —
while the stream it actually surfaces to a human is stderr. Today NSM's diagnostics go to
the discarded stream and are invisible at exactly the moment someone needs them. Routing
through logging to stderr is not a regression for this consumer; it is the fix.

**Target shape (all permanent).**

- **`logging.basicConfig` deleted** from `reconstruct/main.py`. A library never configures
  the root logger.
- **`NSM/__init__.py` adds `logging.getLogger("NSM").addHandler(logging.NullHandler())`** —
  the stdlib idiom, so NSM emits nothing until a host configures logging, and no
  "no handlers" noise is possible.
- **Every module that speaks gets `logger = logging.getLogger(__name__)`.** Module-scope,
  one line, no configuration. The hierarchy under `NSM.*` then lets a host silence or
  raise one subpackage without touching the others — which is the actual benefit of doing
  this properly rather than swapping `print` for a project-wide helper.
- **Level assignment, by what the line is for**, not by where it sits: progress and
  per-step chatter → `debug`; once-per-call facts a user would want in a log of a
  completed run (chosen frame, subject counts, cache hit/miss, fitted loss) → `info`;
  degraded-but-continuing (the deprecated-kwarg notices, the optional-dependency
  fallbacks, `meshfix` giving up) → `warning`. The 68 ungated prints are triaged
  individually; several are debug lines someone forgot to gate.
- **`%` -style lazy interpolation** (`logger.debug("center %s", c)`), not f-strings, so a
  suppressed line costs no formatting. This matters: the hot ones are inside the per-batch
  and per-step loops.
- **`train/deprecated/`'s 30 prints stay untouched** — that module belongs to §8.0.P.

**The `verbose` decision — settled 2026-08-26 (maintainer): deprecate it, logging is
the mechanism.** The problem it settles: any move from `print` to `logging` makes
previously-visible output invisible unless the host configures a handler
(`logging.lastResort` is level `WARNING`, verified), so `info` and `debug` records vanish
by default. Correct library behaviour, and a silent behaviour change for every caller
passing `verbose=True`.

**What ships:** logging is the only output mechanism from commit 5 on. `verbose=` on the
30 public functions that take it is **deprecated and still honoured for one release** —
it emits a `DeprecationWarning` naming the one-line replacement, *and* attaches a
`StreamHandler(sys.stderr)` at `INFO` to the `"NSM"` logger for the duration of the call,
**only if** that logger has no handler beyond the `NullHandler`, so a host that has
configured logging is never overridden. Delete-when: **v0.4.0**, in a transitional module
with the condition in its header (`NSM/_lr_migration.py` is the precedent). CHANGELOG:
Deprecated, plus the stdout → stderr stream change.

**Why honoured rather than a no-op, which is what "deprecate it" would normally mean.**
Measured 2026-08-26: **a `DeprecationWarning` is invisible by default outside
`__main__`** — the same call warns from a script and emits nothing when made from inside
another module (Python's default filter). `kneepipeline` calls `reconstruct_mesh` from
`steps/run_nsm.py`, a non-`__main__` module. So warn-and-no-op is, for the one consumer
we ship to, indistinguishable from deleting their output silently — Principle 5's exact
shape, wearing a deprecation label. The warning is the announcement; the release of
overlap is what makes it an announcement rather than a fait accompli.

**Safe to deprecate, checked rather than assumed.** `verbose` gates output only. Of the
30 public functions taking it, the five `if verbose` blocks containing a non-`print`
statement are all print *support* — a local computed for a print
(`mesh/main.py:722`), a `try/except` around one (`datasets/sdf_dataset.py:370`), loops
and nested `if`s of prints (`triplanar.py:417`, `latent_fit.py:40`, `:651`). Nothing
behavioural hides behind the flag (AST scan, 2026-08-26).

**The `config["verbose"]` key is deliberately NOT deprecated here.** It is a §4 on-disk
format contract — `NSM/configs/default_config.json:139` ships `"verbose": true`, so every
`model_params_config.json` ever written carries it — and there is no `log_level` key to
replace it with. Deprecating a config key with no replacement leaves the user unable to
express "log more" at all. The trainers keep reading it; it maps to the same stderr
bridge. Choosing its replacement is config-shape work and belongs with §8.1 (§8.3),
not here.

**Deliberately out of this slice:** *removing* the `verbose` parameters (that is the
v0.4.0 half of the deprecation, Breaking, and it needs the release this slice precedes);
the `config["verbose"]` key, above; `warnings.warn` → logging (different mechanism,
correctly used at the two sites that have it — `NSM/utils.py`'s schedulefree probe is
already a `UserWarning` on stderr, not the stdout print `ARCHITECTURE.md` §4 described);
tqdm/progress-bar behaviour; anything in `train/deprecated/` (§8.0.P); the `logger` name
that `recon_evaluation.py` and `latent_fit.py` already use (they keep it — this slice adds
peers, it does not rename).

**A retiring argument, recorded so it is not re-used.** §8.0.C and §8.0.E both leaned on
"the log format has no `%(name)s`, so renaming the logger changes no output byte" to prove
a module move was byte-clean. That argument dies here: after this slice NSM owns no
format at all, the host does. Any future move that wants the same guarantee needs a new
one.

**Size budget:** net **negative** in `NSM/` — 257 print statements become 257 logger
calls plus ~14 module-scope `logger = ...` lines, minus the `basicConfig` line and the
prints that triage into deletion. G-i's handler helper is ≤ +30 in one place. Anything
that grows the file is scope creep. Characterization tests are additive and outside the
budget.

**Sequence** (one commit each, suite green and lint clean at every step):
1. this statement; 2. characterization — pin what stdout currently carries at the two
   places it is a contract (a `reconstruct_mesh` run's stdout parses as the consumer
   parses it; the deprecated-kwarg notices, already capsys-pinned in §8.0.C, keep firing),
   and pin that `import NSM.reconstruct` does **not** mutate the root logger
   (strict-xfail — it does today); 3. delete `basicConfig`, add the `NullHandler`, unmark;
   4. the `verbose` bridge — deprecation warning + stderr handler + delete-when header —
   with its tests and CHANGELOG; 5. the per-module conversion,
   one commit per subpackage (`datasets/`, `mesh/`, `reconstruct/`, `train/`, `models/`
   + `utils.py`) so each stays reviewable; 6. verify-and-close #1 (its remedy shipped in
   #69; confirm against `main` and close, or say why not); 7. State update.

**Verification per claim:**

| Claim | Verification |
|---|---|
| importing NSM no longer touches the host's root logger | the commit-2 strict-xfail passes unmarked: a subprocess records root handlers/level before and after `import NSM.reconstruct` and asserts they are unchanged |
| NSM emits nothing unconfigured | subprocess: `import NSM` + a `reconstruct_mesh` run under no logging config produces empty stderr from NSM's own loggers |
| the consumer's stdout contract survives | the commit-2 pin: a fit's stdout still parses with `json.loads(lines[-1])` |
| `verbose=True` still shows the user something | test: `verbose=True` with no host config emits the same messages on **stderr**; asserted against the message set, not the stream formatting |
| a host that configures logging is not overridden | test: pre-attach a handler, call with `verbose=True`, assert NSM added none and the host's handler saw the records |
| `verbose=` announces its own deprecation | test: `pytest.warns(DeprecationWarning)` naming the replacement, on a representative call per subpackage |
| the warning cannot be the *only* notice | the measurement above, recorded in the decision: default filters hide it outside `__main__`, which is why the bridge honours the flag for a release rather than no-opping it |
| `config["verbose"]` still works | the trainer tests, green untouched — the key is out of scope and keeps its meaning |
| no message is lost in the conversion | per-subpackage: the message set before (capsys) equals the message set after (caplog), asserted as a set — this is what makes commit 5 mechanical rather than judgement |
| suppressed lines cost no formatting | the `%`-style form is checked by an AST test over `NSM/`: no `logger.*` call takes an f-string or a `%`/`+`-built first argument |
| numerics unchanged | full suite + regression harness green at every commit |

**Diverged from the statement, on execution (2026-08-26).** Five, each settled by
running it:

- **The bridge attaches at `DEBUG`, not `INFO`.** The statement's own level rule sends
  progress and per-step chatter to `debug`, which is where most of the 187
  `verbose`-gated prints landed — so an `INFO` bridge would have honoured the flag in
  name and dropped most of its output silently, the exact loss the bridge exists to
  prevent. A host that has set its own level still keeps it.
- **28 functions decorated, not 30.** `SDFSamples.load_mesh_step` and
  `_process_meshes_for_wandb` take `verbose` as a *required* parameter, so every call
  site is inside NSM, under an entry point that is bridged already. The rule that
  replaces the count: decorate every function whose `verbose` has a default.
- **The notice fires only when the flag was supplied**, so
  `mesh/interpolate.update_positions` — the one `verbose=True` default — keeps its
  output without announcing a parameter its caller never wrote.
- **The size budget said net negative in `NSM/`; it came out +272.** None of it is
  logic: 135 is the transitional bridge module, 66 is plumbing (28 decorator lines,
  26 import lines, 12 `logger = ...`), and the remaining +71 is black re-wrapping —
  a one-line `print(f"...")` becomes a three-line `logger.debug("...", arg)`. The
  statement wrote the budget as if the conversion were one-for-one on lines. The
  deletion arrives at v0.4.0 with the bridge.
- **Four messages lost a literal level prefix** (`"WARNING: "` ×3, `"Warning: "` ×2 on
  the two batch-size shims), which the record's level now carries; everything else is
  byte-identical. The statement called the conversion message-preserving without
  allowing for this.

Two judgement calls inside the level rule, recorded because they read as exceptions to
it: the pympler import-time fallback is `debug`, not the `warning` the
optional-dependency clause would give it — it fires in every consumer process for a
debug-only capability, and a warning there is what teaches people to filter NSM out —
and `configs/generate_sdf_default_config.py`'s one `print` stays a `print`, because a
script's own output on its own stdout is not the library speaking.

### 8.0.H The `models/` package — plan statement (2026-08-26)

Every claim below was re-run against `main` at `57ebfbe` before it was written. `models/`
is 1,522 lines across five files and has had no pass of any kind — not Phase 2, not 3,
not 4 — while holding half the consumer's public contract (`TriplanarDecoder`) and the
documented `load_model` entry point.

**What is actually wrong, measured.** Nine defects, and they are three shapes, not nine:

*Shape 1 — a config value reaches a constructor unchecked.* `padding` is not a learned
parameter, so a checkpoint trained at one value loads cleanly at another and samples the
feature planes at the wrong scale: 0.063 max SDF difference on a `tanh`-bounded output
(#26). `layer_split: false` — what `default_config.json` ships — is `False`, which
`Decoder` tests with `is not None`, so it means *split at layer 0*, not *do not split*:
verified, it moves every state-dict key from `layers.N.weight` to `layers.N.0.weight`.

*Shape 2 — an argument accepted and never read* (#20's class, and the memory of what
honouring one costs is why each is deleted rather than wired up).
`normalize_coordinates(padding=)`; `Decoder(xyz_in_all=)`, which `default_config.json`
ships and `loader` plumbs through four call sites; `Decoder(latent_noise_sigma=)`, stored
and never read; `VAEDecoder(activation=)`, built and discarded — the `LeakyReLU` never
enters the stack (ARCHITECTURE §7.1); `weight_norm_all`, defined and called nowhere.

*Shape 3 — a documented option that constructs and then does not work.*
`sum_sdf_features=False` sizes the VAE by `sdf_latent_size` while
`forward_with_plane_features` slices `sdf_latent_size` **per plane**, so the three planes
get (12, 0, 0) channels and the output is `torch.equal` to using the xz plane alone —
re-measured today, and every VAE parameter still receives gradient, so training converges
to a silently degraded model (#45). `Decoder(activation='linear')` gets `None` from
`get_activation` and calls it. `progressive_add_depth=True` returns `None` from
`forward_branch_` for every epoch below the last configured `start_epoch` (verified at
0/100/300/700; 1300 and 5000 work). `Decoder(norm_layers=(1,2), weight_norm=False)`
raises `IndexError` — it indexes `self.bn` by absolute layer index and appends only per
norm layer — while with `weight_norm=True`, the shipped value, the whole option is
silently inert. `TwoStageDecoder()` raises `TypeError` at any argument (`list` + `tuple`)
and mutates its module-level default dicts on the way out — verified: after one failed
construction `default_triplanar_params["latent_dim"]` is 32, process-wide.

And #34: `TestAFreshlyTrainedDecoder` asserts a surface comes back and the latent has the
configured shape. Both hold for an untrained decoder — measured today, trained
`assd_0/assd_1` = 0.224/0.172 against untrained 2.197/3.014, and the untrained run passes
every assertion in that class.

**Target shape (all permanent — this slice adds no transitional module).**

- **Each option works or refuses at construction**, which is #46's own closure criterion.
  `progressive_add_depth` works (a not-yet-started block is skipped, not turned into
  `None`); `activation='linear'` refuses (an affine SDF decoder is not a thing anyone
  wants silently); `norm_layers` is deleted; `TwoStageDecoder` builds and stops mutating
  its defaults; `layer_split=False` is normalized to `None` at the boundary, because
  `False == 0` makes the two indistinguishable by value and only one of them is what
  `default_config.json` means.
- **`sum_sdf_features=False` slices `sdf_latent_size // 3` per plane**, which is what its
  own `% 3` guard has always implied, and the guard becomes a `ValueError` so `-O` cannot
  strip it. `conv_pred_sdf=True` *with* concatenation refuses: the per-plane SDF channels
  have no defined combination rule under concatenation and never had one.
- **`load_model` refuses a triplanar config that omits `padding`**, with a message naming
  the one line to add. This is #26 option 1 and deliberately not option 3 — the plan's
  §8.1 note says taking the registry first is how that section swallows this slice.
- **Every dead argument is deleted, not honoured.** `Decoder` keeps its `**kwargs`, so a
  deleted name would go back to being silently ignored; the two that a config can still
  carry (`xyz_in_all`, `norm_layers`) raise from `**kwargs` when set to something truthy
  and stay silent when falsy, which is what every NSM-owned config ships.
- **One `Sine`.** `deep_sdf.Sine` (`w0` hardcoded 30, `__init__` misspelled `__init` so it
  is mangled to `_Sine__init` and never runs) is deleted; `deep_sdf` imports the
  `modulated_periodic_activations` one and `get_activation("sin")` returns `Sine(w0=30)`.
  `torch.equal` on the two outputs is `True`, so no run changes.

**Deliberately NOT in this slice, each for a stated reason.**

- **Adding the VAE's missing activation.** It is real, it is documented in ARCHITECTURE
  §7.1, and it is not fixed here: adding the activations **unconditionally** would shift
  every subsequent module's index inside `nn.Sequential`, so all three shipped checkpoints
  stop loading, and the weights were fitted without them anyway. An opt-in flag defaulting
  to off is compatible (verified bitwise) — what it needs is the retrain that says whether
  it helps, which is `NSM_TRAINING_IDEAS.md` Idea 13, not this slice. What this slice does is delete the dead
  `activation=` parameter and pin the structure so the next reader cannot "fix" it by
  accident. The issue text is drafted in the PR body for the maintainer to file.
- **Rejecting unknown `**kwargs`** on `Decoder`/`TriplanarDecoder`. It closes the same
  class and it is a behaviour change with unmeasurable external blast radius; it wants the
  release boundary §8.0.O owns.
- **A public "build the model this config describes" call** (#26 option 3, SCOPE §3.1's
  "single highest-value API change"). That is §8.1, deferred by the 2026-08-26 re-draw.
- **The `l2reg`/latent-gradient convention** (KNOWN_ISSUES `models/triplanar.py` §3). It
  rescales every run and is a research question, not a refactor.
- **Removing config keys other than `layers_with_norm`.** `layer_split`, `xyz_in_all` and
  `latent_dropout` stay in `default_config.json`; they are inert and their removal is
  config-shape work (§8.1/§8.3). `layers_with_norm` goes because the argument it feeds
  ceases to exist.

**Size budget.** Net negative in `NSM/models/`: five deleted arguments, one deleted class,
one deleted function and `self.bn` against roughly +25 lines of refusal and slicing. Past
+10 net in `NSM/` is scope creep. Tests are additive and outside the budget.

**Sequence** (one commit each; `make lint` clean and the full suite green at every step):

1. this statement;
2. characterization — a parameterised constructor-and-one-forward matrix over every
   documented option value of all four model types, plus structural pins for the things no
   fix here may silently change: the VAE's additivity table from §7.1 and the `Sine`
   resolution. Strict xfails for what is broken. **#34 gets no xfail** — the assertion it
   wants (trained error below an untrained control's) already holds, it is simply not
   asserted anywhere, so its measurement lands with its fix in commit 10;
3. #46(a) `TwoStageDecoder` builds, and stops mutating its module-level defaults;
4. #46(b) `activation='linear'`, `progressive_add_depth`, and the `layer_split=False`
   normalization;
5. #46(c) `norm_layers` deleted — decoder, loader, templates, shipped config;
6. #45 all three planes, with its § History entry;
7. #26 `load_model` refuses a config that omits `padding`;
8. the #20 dead-argument sweep, retiring both `models/` § Open entries;
9. one `Sine`;
10. #34 an assertion that goes red if training stops learning;
11. docs sweep (ARCHITECTURE §5/§6/§7, SCOPE §2.6's open question, CHANGELOG) and this
    plan's State.

**Verification per claim:**

| Claim | Verification |
|---|---|
| every documented option value works or refuses at construction | the commit-2 matrix, run over the four model types; its strict xfails all XPASS by commit 7 and are unmarked in the commit that fixes each |
| `sum_sdf_features=False` uses three planes | forward-shape test over both flag values, plus an assertion that the concat output is **not** `torch.equal` to the xz plane alone — the exact equality measured today |
| an old `sum_conv_output_features: false` checkpoint still loads | the VAE output width is `sdf_latent_size` before and after, so state-dict shapes are unchanged: asserted by loading a pre-fix checkpoint |
| no shipped model changes | both shipped configs set `sum_conv_output_features: true`; the round-trip test's bitwise assertion covers the rest |
| `load_model` refuses a config without `padding` | `TestPaddingIsNotInTheCheckpoint` is rewritten from "loads without error" to `pytest.raises`; its #26 strict xfail goes with it |
| the refusal names the fix | the raised message contains `"padding"` and a value, asserted on the message |
| `progressive_add_depth` is continuous across `start_epoch` | forward at `start-1`, `start`, `start+1` differs by less than the warmup step, rather than jumping to a full-weight layer — the ordering defect the `RuntimeError` branch hides |
| `layer_split=False` means no split | state-dict keys are `layers.N.weight`, the same list `layer_split=None` produces — the difference measured today |
| deleting an argument does not re-silence it | a truthy `xyz_in_all` / `norm_layers` raises `TypeError` through `**kwargs`; a falsy one does not |
| one `Sine` changes no arithmetic | `torch.equal(old_Sine()(x), Sine(w0=30)(x))` — `True` today, kept as a test |
| #34 is training-dependent | trained `assd` versus an untrained control built from the same config: measured 9.8× and 17.5×, asserted at a factor with its headroom in the docstring |
| the suite still passes | 704 passed / 1 skipped / 5 xfailed on `main` at `57ebfbe` is the baseline every commit is compared against |

### 8.0.I The `mesh/` package — plan statement (2026-08-26)

Every claim below was re-run against `main` at `4a16197` before it was written, and two of
them came back different from what the issue text says.

**The slice index row is wrong about where its own work lives, and that is the first
finding.** The row reads "`mesh/main.py`" and carries #57 and "#54's sites there". #57 has
**zero** sites in `mesh/main.py` — its five are in `correspondence_metrics.py` (2),
`interpolate.py` (2) and `refine_mesh.py` (1) — and #54's two mesh-side sites are in
`refine_mesh.py` and `correspondence_metrics.py`. So the slice is the `mesh/` **package**:
2,913 lines across six files, of which `main.py` is 930. That is also the honest framing,
because ARCHITECTURE §2.1 already records that the other four files are unreachable from
any other subpackage — nothing else is ever going to open them.

**What is actually wrong, measured.** Fourteen sites, five shapes.

*Shape 1 — a reshape used as a validator, and it is a modular coincidence* (#57, five
sites). Each site takes a VTK-style flat face array (`[n, i0, …, in, n, …]`) and calls
`reshape(-1, 4)` or `reshape(-1, 3)` with nothing checking the cell type. The reshape
succeeds exactly when the flat length happens to divide — which is a fact about the cell
count mod 3 or mod 4, not about the mesh being triangular:

| input | flat length | `reshape(-1, 4)` | `reshape(-1, 3)` |
|---|---|---|---|
| 3 quads | 15 | `ValueError` | **5 silent rows** |
| 4 quads | 20 | **5 silent rows** for 4 cells | `ValueError` |
| 96 triangles, VTK-style `.faces` | 384 | correct (96) | **128 silent rows** |
| 4 triangles + 4 quads | 36 | **9 silent rows** for 8 cells | `ValueError` |

The issue says pure-quad input "raises a bare reshape `ValueError` in two metrics". It does
for 3 quads and not for 4: measured today, `self_intersection_count` on a 4-quad strip
returns `0` and `foldover_count` returns `near_degenerate: 2`, both computed from five
fabricated triangles. On the `(-1, 3)` side, `build_mesh_laplacian(sphere.faces, …)` builds
a 373-nnz operator where the correct one is 288 and `compute_feature_mask` flags 50
vertices where the correct answer is 8 — so the interpolation output is wrong rather than
absent, which is the issue's own headline claim and it holds.

*Shape 2 — one boolean swaps two functions that are not interchangeable* (#60, first half).
`use_vtk` picks between `sdf_grid_to_mesh_vtk` and `sdf_grid_to_mesh`.
`sdf_grid_to_mesh(numpy_array, …)` raises `AttributeError: 'numpy.ndarray' object has no
attribute 'cpu'` on its first line while the VTK twin, which guards with `hasattr`, accepts
it. Their `narrow_band` defaults are `False` and `True`. **The `narrow_band` half is
behaviourally inert**: on a 48³ sphere both twins return the same vertex count either way
and differ by ≤ 8.9e-08, which is below float32 mesh precision — so aligning the defaults
is API hygiene and needs no § History entry. The `.cpu()` half always crashed, so it needs
none either.

*Shape 3 — 17 positional arguments across a call boundary* (#60, second half; ARCHITECTURE
§7 names this exact site as its example of the LR bug's shape). `create_mesh_adaptive`'s
no-surface fallback forwards its own `voxel_origin` — untouched at the `(-1, -1, -1)`
default — next to a `voxel_size` it derived from `search_bounds`. Measured with
`search_bounds=(0.0, 4.0)`, `n_pts_per_axis=17`: the fallback grid covers `[-1, 3]` on
every axis while the caller asked for `[0, 4]`. Wrong by construction, on a branch a caller
reaches by passing `search_bounds` and nothing else.

*Shape 4 — constructible but uncallable* (#54; ARCHITECTURE §7 calls this the surviving
instance of that class and assigns it here by name). `get_target_cells` reads
`np.zeros_like(max_length_binary)` where it means `max_lengths`, so it raises
`UnboundLocalError` on its own defaults — and on the `area_threshold`-only path, which is
the one SCOPE §2.3 gates its "keep" ruling on. `subdivide_large_triangles` inherits it.

*Shape 5 — a fabricated metric sitting next to a sibling that skips* (#54).
`score_correspondence(roundtrip_points=…, source_mesh=None)` substitutes the **warped**
mesh for the missing source: measured mean roundtrip distance `0.2500` against a true
`0.0017`, a factor of 144, returned in the same dict where `foldover_count` correctly says
`{"skipped": True, "reason": "source_mesh not provided"}`.

**Target shape (all permanent — this slice adds no transitional module).**

- **One validated face accessor**, `triangle_faces()`, in `triangle_metrics.py` — the leaf
  `correspondence_metrics` and `refine_mesh` already import, and the only file in `mesh/`
  that is *about* triangles. It accepts a mesh (requiring `is_all_triangles`, since
  pyvista's `regular_faces` returns a (M, 4) array for quads rather than refusing) or an
  already-(M, 3) array, and raises a named error otherwise. All five sites route through
  it. `refine_mesh.get_faces` becomes its forwarder — it *is* the accessor already, just
  unvalidated.
- **The twins guard alike and share defaults.** `sdf_grid_to_mesh` gets the `hasattr` guard
  and `narrow_band=True`, so `use_vtk` chooses an extraction backend and nothing else.
- **The fallback derives its origin from `search_bounds`**, and the 17 positional arguments
  become keywords, so the next parameter added to `create_mesh` cannot silently shift them.
- **`get_target_cells` works**, and SCOPE §2.3's conditions 2 and 3 land with it: the
  cross-mesh precondition warned at the entry points, and a module docstring saying what
  the module uniquely provides, why `pyvista.subdivide_adaptive` was rejected, and that
  `area_threshold` is a relative deviation rather than an area.
- **`score_correspondence` skips with a reason** when `source_mesh` is absent, which is
  what its own docstring already promises.
- **`create_mesh_adaptive`'s tail is shared with `create_mesh`.** The extract → scale →
  save loop is duplicated near-verbatim in both (the diff is comments plus one `verbose`
  forward); one private helper removes ~40 duplicated lines and gives the `verbose`
  divergence a single place to be decided.

**Deliberately NOT in this slice, each for a stated reason.**

- **The `triangle_metrics` merge decision** (SCOPE §2.6, ARCHITECTURE §6's "two edge-ratio
  implementations"). It is a scope ruling with an open question the maintainer owns, not a
  defect. This slice feeds it one fact rather than pre-empting it: putting the shared
  accessor there makes `triangle_metrics` the package's leaf, which argues for keeping it
  a separate file.
- **Making the mesh cluster reachable** (ARCHITECTURE §2.1). Adding submodule imports to
  `NSM/mesh/__init__.py` is a public-surface change and wants §8.0.O's release boundary.
- **Shrinking `create_mesh_adaptive`'s 26-parameter and `create_mesh`'s 17-parameter public
  signatures.** Same reason. What this slice does is stop the *internal* call between them
  being positional.
- **`correspondence_metrics`' blanket `except Exception` per metric.** It is documented
  behaviour ("one bad metric cannot sink the rest") and every arm is `# pragma: no cover`.
  Changing it is a design call, and after this slice the accessor's refusal is the thing it
  would swallow — which is the argument to make, at §8.0.N, with evidence.

**Size budget.** Roughly flat in `NSM/`: the accessor and the refusals (~+60) against the
shared tail (~−40) and five reshape lines that get shorter. Past **+30 net** in `NSM/` is
scope creep. Tests and docstrings are additive and outside the budget.

**Sequence** (one commit each; `make lint` clean and the full suite green at every step):

1. this statement;
2. characterization — a face-array matrix (triangle / quad / mixed / VTK-style-flat × the
   five sites), a twin-parity matrix over `sdf_grid_to_mesh` and `sdf_grid_to_mesh_vtk`,
   the fallback-grid bounds assertion, `get_target_cells()` on its own defaults, and
   `score_correspondence`'s skip contract. Strict xfails for what is broken;
3. #57 — `triangle_faces` and its five sites;
4. #60(a) — the twins guard alike and share defaults;
5. #60(b) — the fallback grid, and keywords across that call;
6. #54(a) — `get_target_cells` works, plus SCOPE §2.3's conditions 2 and 3;
7. #54(b) — `score_correspondence` skips instead of inventing;
8. `create_mesh_adaptive`: the shared tail extracted;
9. docs sweep (KNOWN_ISSUES § Open and § History, SCOPE §2.3, ARCHITECTURE §3/§6/§7,
   CHANGELOG) and this plan's State.

**Verification per claim:**

| Claim | Verification |
|---|---|
| the reshape sites accept non-triangular input | the commit-2 matrix, parameterised over the four cell layouts and the five sites; its strict xfails XPASS at commit 3 |
| the accessor changes nothing for triangle meshes | `regular_faces` equals `faces.reshape(-1, 4)[:, 1:]` element-wise on a triangulated sphere, asserted, so no run through a triangle mesh moves |
| a quad mesh is refused rather than reshaped | `pytest.raises` on each of the five sites with the 4-quad strip — the case that silently returned `0` and `near_degenerate: 2` today |
| the twins accept the same inputs | both called with numpy and with torch, same vertex count from each |
| aligning `narrow_band` changes no geometry | vertex counts equal and the max absolute vertex difference is < 1e-06 between `narrow_band` False and True, on both twins — measured 8.9e-08, and the headroom goes in the test docstring |
| the fallback grid covers `search_bounds` | `create_mesh` is spied on through the fallback with `search_bounds=(0.0, 4.0)`; the asserted grid span is `[0, 4]`, against the `[-1, 3]` measured today |
| `get_target_cells` runs on its own defaults | direct call, plus `subdivide_large_triangles` on its defaults; both are `UnboundLocalError` today |
| `score_correspondence` does not invent | with `source_mesh=None` and `roundtrip_points` given, both roundtrip keys are `{"skipped": True, …}`; and with `source_mesh` given the value is unchanged from today's |
| the shared tail is a refactor, not a change | `create_mesh` and `create_mesh_adaptive` outputs are compared vertex-for-vertex against the pre-refactor implementation on a fixed analytic decoder |
| the suite still passes | 812 passed / 1 skipped / 3 xfailed on `main` at `4a16197` is the baseline every commit is compared against |


### 8.0.J `reconstruct_mesh` internals — plan statement (2026-08-27)

Every number below was re-run against `main` at `e9ec5d0` before it was written. Two of
the slice-index row's own numbers came back different, and the first of them changes what
the slice can do.

**The row's headline is "the 61-parameter signature", and shrinking it is not this
slice's to do.** Measured: 58 named parameters plus `**kwargs`, in a 409-line body. But
`reconstruct_mesh` is frozen public API — `testing/NSM/reconstruct/test_reconstruct_import_compat.py`
pins it on both `NSM.reconstruct` and `NSM.reconstruct.main`, and `kneepipeline`'s
`steps/run_nsm.py:185` calls it with **27 keyword arguments**. That is the same
release-boundary constraint §8.0.I cited when it left `create_mesh` (17) and
`create_mesh_adaptive` (26) alone for §8.0.O, and the State block asked whether the two
want doing together. **They do not, and the reason is the answer to the question:** all
three are public signatures, so all three shrink at the same release, and §8.0.O should
take them as one set rather than §8.0.J taking a third of it early. What §8.0.J owns is
the half that does not need a version boundary — the internals behind the signature, and
the hole that makes a 58-parameter signature dangerous *today*.

**What is actually wrong, measured.** Five defects, one dead branch, one dead import set.

*Defect 1 — 58 named parameters and a `**kwargs` that accepts anything.* Nothing in
`reconstruct_mesh` inspects `kwargs` except a `batch_size_latent_recon` deprecation
notice; every other key is swallowed. Measured on the single-object sampled run:
`n_pts_per_axes`, `num_iteration`, `calc_assd_`, `latent_reg_wieght` and
`clamp_distance` — 5 of 5 — complete without an error, a warning or a log record, having
silently used the default for the parameter the caller meant. This is the widest surface
in the repo for the accepted-and-ignored trap and the one where a typo is most likely,
because the correct spellings are 58 near-synonyms.

*Defect 2 — `register_similarity` is read two ways in the same call.* Line 198 gates the
mean-mesh build on `register_similarity is True`; line 256 forwards
`register_to_mean_first=True if register_similarity else False`, which is truthy. A
truthy non-`True` value therefore skips the build and then asks the reader to register to
a mean mesh that was never made. Measured: `register_similarity=1` and
`register_similarity="similarity"` both raise
`Exception: Must provide mean mesh to register to` from `mesh_sampling.py:149` — a bare
`Exception`, from a file the caller never named, saying to supply something that is not a
parameter of the function they called.

*Defect 3 — the reference mesh is built on a path that discards it.* The build fires on
`scale_jointly or register_similarity is True`, but the mean mesh is only ever *used* by
`register_to_mean_first`, which is set from `register_similarity` alone (verified: it is
the only reader of `mean_mesh` in either sampler, `mesh_sampling.py:141` and `:552`). So
`scale_jointly=True, register_similarity=False` reconstructs a whole mean shape and throws
it away. Measured by counting decoder point-evaluations on the single-object run at the
`n_pts_per_axis_mean_mesh=128` default: **524,968 → 1,401,237**, an extra **876,269**
evaluations for nothing. The sharper half is not the cost: that path also raises
`NoZeroLevelSetError` when the discarded mean shape has no surface, so an under-trained
model aborts the run over a mesh it was never going to consult. `NSM/configs`'s shipped
default has `scale_jointly: True` and `get_mean_errors` defaults `register_similarity=False`,
so the combination is the one a config-driven caller reaches first. (`train_deep_sdf.py:341`
passes `register_similarity=True` explicitly, which is why the trainer has never hit it.)

*Defect 4 — six stage timings are measured and five are returned.* `time_calc_recon_loss`
is computed at `main.py:439` and never reaches the result dict. It is the timing of the
one stage that is *optional*, so `return_timing` is silent about exactly the stage a
caller would be trying to attribute cost to.

*Defect 5 — §8.0.G's conversion left the log records gated behind the flag it deprecated.*
Ten of the function's fifteen `logger` calls sit under `if verbose is True:`. That was
message-preserving at the time and correct for that slice; the consequence, now, is that a
host which configures logging — the entire point of the conversion — sees none of them
unless it *also* passes the parameter §8.0.G deprecated. One of the ten is a
`logger.warning` that a surface was skipped (`main.py:293`), which is a fact about the
result, not chatter.

*Dead branch.* `main.py:280` raises `ValueError("multi_object must be True or False")` on
a variable assigned exactly `False`, `True`, or not at all, three lines above, in the only
place it is written.

*Dead imports.* Ten `F401`s in the file, and they are two different things: `copy`, `os`,
`sys`, `numpy as np` and `pymskt as mskt` are unused, while `fnmatch`, `EIKONAL_UNSUPPORTED`,
`eikonal_loss`, `Regress` and `adjust_learning_rate` are unused *here* and frozen as
re-exports by the import-compat test. `KNOWN_ISSUES` § Open names `F401` being
project-ignored and says the blocker is that "several are deliberate re-exports"; this file
is the case that shows the two kinds are distinguishable by marking them. That entry's
count is also stale — it says 43, the measurement today is **54**.

**Target shape (all permanent — this slice adds no transitional module).**

- **Unknown keywords are refused by name.** `**kwargs` keeps
  `batch_size_latent_recon` (the consumer passes it, `run_nsm.py:200`) and raises
  `TypeError` naming every other key it was given. Refusing is the fix and honouring is
  not: NSM's own rule is that an accepted-and-ignored parameter gets deleted rather than
  implemented. Nothing beyond the name — the missing signal was the refusal, and a
  caller holding a `TypeError` that names their own typo does not need help past that.
- **`register_similarity` is decided once**, into a local the gate and the forward both
  read, so the two cannot disagree again. Truthiness is the reading kept — `is True`
  is the outlier of the two, and the function's other flags are all truthy-tested.
- **The reference mesh is built when it is used**, i.e. on `register_similarity` alone.
  `scale_jointly` stops implying it. Verified numerically inert for every path that
  *does* use it, and the removal cannot move a fitted latent because
  `create_mesh_adaptive` consumes no RNG (measured: torch and numpy states are
  bit-identical across a call).
- **`time_calc_recon_loss` is returned**, and the stage timings stop interleaving: one
  small recorder replaces eleven `tic`/`toc` assignments threaded through the body, so a
  stage can be moved without moving a clock.
- **Five stages become keyword-only private helpers** — `_normalize_call`,
  `_build_reference_mesh`, `_sample_subject`, `_decode_meshes`, `_assemble_result` —
  leaving `reconstruct_mesh` an orchestrator. Keyword-only (`*`) is not decoration: it is
  §8.0.I review round 2's lesson applied at the signature, so no later author can write a
  positional call across one of these boundaries. **Five, not the row's seven.** The fit
  stage is a 30-key dict literal and one call; wrapping it in a helper with 30 keyword-only
  parameters moves the list and improves nothing. The metrics stage is one `if` and one
  call. Extracting either would be structure invented to match a plan sentence.
- **The log records stop being gated on the deprecated flag.** `verbose=True` still shows
  all ten, because the bridge attaches at `DEBUG` (`_verbose_deprecation.py:82`); what
  changes is that a host configured at `DEBUG` sees them too, and the skipped-surface
  warning reaches a host configured at `WARNING`. This is one file, not the repo-wide
  sweep of the same shape — that is recorded as a residual for whichever slice opens each
  remaining file.
- **The dead branch goes, the five dead imports go, and the five frozen re-exports get
  `# noqa: F401` and one comment** saying they are the import-compat contract.

**Deliberately NOT in this slice, each for a stated reason.**

- **Shrinking the public signature.** Above: it is §8.0.O's, with `mesh/main.py`'s two, as
  one set at one boundary.
- **`reconstruct_latent`'s internals and #75.** That is §8.0.K, the next row. This slice
  stops at the `reconstruct_latent(**reconstruct_inputs)` call.
- **The repo-wide `if verbose is True:` gate removal.** Same shape, ~180 more sites across
  files this slice does not open; opening them here would make the diff unreviewable and
  duplicate work every later slice has to do anyway.
- **Removing `F401` from `.flake8`.** The § Open entry's judgement call, and it needs the
  other 44 sites triaged, not five.
- **Issue #3's `scale_jointly` sigma semantics.** §8.0.Q, and a behaviour change. This
  slice only stops `scale_jointly` from implying a mean-mesh build; what it *means* for
  sampling is untouched.

**Size budget, by part** — a single net-lines ceiling has been missed in §8.0.H and
§8.0.I both, so this one is named:

| part | budget |
|---|---|
| the keyword refusal, with its message | +25 |
| the timing recorder | +25 |
| five helper signatures and docstrings | +70 |
| removed: gates, `tic`/`toc`, dead branch, dead imports | −45 |
| **net in `NSM/`** | **+75** |

Past **+95** is scope creep. Tests are additive and outside the budget.

**Sequence** (one commit each; `make lint` clean and the full suite green at every step):

1. this statement;
2. characterization — the five-typo `**kwargs` matrix, the `register_similarity`
   truthiness matrix, a decoder-evaluation count across `scale_jointly`, the
   `return_timing` key set against the timings the body measures, the log-record set at
   `DEBUG` with and without `verbose`, and a stage-handoff pin per seam. Strict xfails for
   what is broken;
3. unknown keywords refused by name;
4. `register_similarity` decided once;
5. the reference mesh built only when it is used;
6. the log records ungated; the dead branch and the dead imports;
7. `time_calc_recon_loss` returned, and the stage-timing recorder;
8. the five stages extracted, keyword-only — behaviour-preserving;
9. docs sweep (`KNOWN_ISSUES` § History 20 and the § Open recount, CHANGELOG,
   ARCHITECTURE) and this plan's State.

**Verification per claim:**

| Claim | Verification |
|---|---|
| unknown keywords are accepted and ignored today | the commit-2 matrix: five misspellings of real parameters complete a run with no error, no warning and no log record; its strict xfails XPASS at commit 3 |
| the refusal does not break the one consumer | `batch_size_latent_recon` still warns and runs — the existing `test_the_shim_warns_and_stays` pin, green untouched — and a call with the consumer's own 27 keywords is asserted to raise nothing |
| `register_similarity` is read two ways | parameterised over `True`, `1`, `"similarity"`: today the last two raise `Exception: Must provide mean mesh to register to`; after commit 4 all three take the same path |
| the reference mesh is discarded under `scale_jointly` alone | a counting decoder: point-evaluations equal with and without `scale_jointly` after commit 5, against the measured 524,968 vs 1,401,237 today |
| the discarded build is numerically inert | the fitted latent is bit-identical across commit 5 on a fixed seed, and `create_mesh_adaptive` leaves `torch.get_rng_state()` and `np.random.get_state()` unchanged |
| a path that uses the mean mesh is unaffected | `register_similarity=True` output compared vertex-for-vertex against the pre-commit-5 implementation, and `NoZeroLevelSetError` still raised there — the existing regression pin, green untouched |
| `return_timing` drops a stage it measures | the key set is asserted against the `time_* = toc - tic` assignments the body makes, read from the source, so a stage added later without a key turns it red |
| ungating changes nothing for `verbose=True` | the record set under `verbose=True` is asserted equal before and after; the added case is a host at `DEBUG` with no `verbose`, empty today |
| the extraction is a refactor, not a change | the full result dict — meshes vertex-for-vertex, latent, every metric and registration key — compared against the pre-refactor implementation on a fixed analytic decoder and a fixed seed |
| the suite still passes | 864 passed / 1 skipped / 3 xfailed on `main` at `e9ec5d0` is the baseline every commit is compared against |


### 8.1 Make the library plural — added 2026-08-15

> **Deferred 2026-08-26 — this is an upgrade, not the refactor.** All three bullets are
> new capability, and they are indexed in §8.3. The one thread that does *not* wait is the
> defensive half of #26, which §8.0.H carries: refusing to load a checkpoint whose config
> omits `padding` is a silent-wrong-answer fix, where "a third party adds a model type and
> it works everywhere" is a feature. Keep them apart — #26's issue text lists them as
> options 1 and 3 of the same fix, and taking option 3 first is how this section swallows
> the slice.

The name is *neural shape model**s***. Two defects currently make it singular in practice,
and both are structural, so they belong here rather than in the documentation pass.

- [ ] **A common decoder interface and a registration pathway.** Today
      `reconstruct/main.py:588` hardcodes a keyword-only `(latent=, xyz=)` call with no
      fallback, while `mesh/main.py:855-867` inspects the signature and falls back — two
      conventions in one pipeline. Three of `load_model`'s four advertised model types
      cannot be reconstructed. The goal is that a third party adds a model and it works in
      train, reconstruct, mesh and interpolate without editing NSM internals. One calling
      convention wins; every shipped checkpoint keeps loading.
      `models/loader.py` is where this hangs — see the open question in `docs/SCOPE.md` §2.6
      about whether the consumer could switch to `load_model` today.
- [ ] **A default config per model type, derived from the ShapeMedKnee configs.** The
      shipped `default_config.json` has 61 DeepSDF-shaped keys; the real production configs
      have 131. A triplanar model built from the shipped default silently falls back to a
      different architecture. Generate them the way the current one is generated, and pin
      them the way `testing/NSM/configs/test_default_config_sync.py` already pins it.
- [ ] **Config naming, validation and documentation.** The maintainer's assessment: the
      options are poorly documented and many names do not describe what the code does with
      them. Open sub-question — whether this needs a restructured config format or only
      renames plus validation. Every rename breaks a file someone has on disk, so the
      migration cost gets measured before the shape is chosen. The LR `Target` key
      (`NSM/utils.py` + `NSM/_lr_migration.py`) is the in-repo reference for how a config
      change fails loudly and hands the user a corrected copy of their own file.

**Sequencing note.** Both items change public behaviour, so both need §4-style migration
guards and §9 ledger entries, and both need §7.1 green first. The registry is the one that
unlocks the others: until there is one supported way to build a model from a config,
"a default config per model type" has no single consumer to be correct for.

### 8.2 Eikonal loss — gated 2026-08-15, needs repair

> **Deferred 2026-08-26 — the refactor's part of this is done.** Gating a never-runnable
> path behind `NotImplementedError` *was* the code-health outcome; making the loss work is
> research, tracked as `NSM_TRAINING_IDEAS.md` Idea 3. The three failures below stay here
> because they are the executed evidence, and whoever picks the research up needs them.
> Indexed in §8.3.

`eikonal_weight > 0` now raises `NotImplementedError` at both entry points
(`train_deep_sdf`, `reconstruct_latent`), with the message in `NSM/losses.py`.
`testing/NSM/test_losses.py` pins it and is written to **fail once the loss works** —
deleting that file is part of fixing this. No results are affected — the path always crashed, so per `CLAUDE.md` it gets no
`KNOWN_ISSUES.md` entry. Neither ShapeMedKnee config contains the key and
`kneepipeline` never passes it; production has never touched this code.

Three independent failures, in the order they must be fixed:

- [ ] **It crashes on the first backward pass.** `losses.py:54` reads
      `retain_graph=True if surf_idx < n_surfaces - 1 else False` — on the last (or only)
      surface that frees the forward graph the double-backward graph still needs, so the
      caller's `.backward()` raises. Verified for 1, 2 and 4 surfaces. One-line fix; the
      same pattern repeats in `compute_sdf_gradients`.
- [ ] **Triplanar models cannot use it at all.** It needs a second derivative through
      `grid_sample`, which PyTorch does not implement — verified on CPU and on a T4 with
      torch 2.8.0. The first-order gradient computes fine; the backward *through* it does
      not. This is not ours to fix, so the guard must stay for triplanar regardless: any
      future support is MLP-architectures-only until upstream changes.
- [ ] **It opposes clamped training,** which is the regime NSM actually uses
      (`enforce_minmax: true`, `clamp_dist: 1` in both production configs). A clamped
      target is flat outside the band, so its true gradient norm is 0, not 1. Measured on
      an analytic sphere at the generator's own suggested `0.1`: gradient norm converges
      (mean `|‖∇f‖−1|` 0.928 → 0.0091) while the zero level set is destroyed (surface error
      0.162 → 1.050). Unclamped, it is mildly helpful (0.0194 → 0.0172). If adopted, either
      restrict eikonal sample points to inside the clamp band or require
      `enforce_minmax: false`.
- [ ] Secondary: the eikonal term is computed on the **unclamped** prediction
      (`train_deep_sdf.py:510`) while L1 uses the clamped one (`:398`) — a second full
      forward pass, costing ~4x step time and ~3x memory when enabled.
- [ ] Secondary: `reconstruct/main.py` calls it under `torch.no_grad()` at `:723-724`,
      which raises independently of the above.

**When it is fixed, the test is the deliverable, not the fix** — an analytic sphere where
the gradient norm is measurably closer to 1 with the loss on, and a backwardable-regression
guard parametrised over 1/2/4 surfaces. Without that, "does it help" is unanswerable again
in a year.

**Rule:** every commit keeps the §7.1 harness green. Any intended numerical change gets a
§9 ledger entry and a §4-style migration guard.

---

### 8.3 Deferred until the refactor closes — drawn 2026-08-26

The plan's own §Scope banner has always said new science is out of scope. What it did not
say is that §8 had grown three items that are not code health either, and one of them had
reached the **Next** line. This is the list, so that deferring them is a decision with a
venue rather than a thing that keeps not happening.

| Deferred | Where it lives | Why it is not refactor |
|---|---|---|
| §8.1 — decoder interface, per-type default configs, config renames | here, banner above | New capability. Every bullet changes public behaviour and needs its own migration guard; none of them makes existing code more correct. |
| §8.2 — repairing the Eikonal loss | here + `NSM_TRAINING_IDEAS.md` Idea 3 | The loss is gated, which is the code-health answer. Making it work is an experiment with an unknown result — measured to *oppose* the clamped regime NSM actually trains in. |
| Idea 4 — the latent norm bound, training and recon sides | `NSM_TRAINING_IDEAS.md` Idea 4/7/11 | A training experiment. It sat in this plan's **Next** on 2026-08-25 while §8 had eleven unstarted slices; that is the specific failure this table exists to prevent. |
| Ideas 6, 10, 11, 12 | `NSM_TRAINING_IDEAS.md` | Same. Each is independently executable and none of them is blocked by the refactor. |
| #2 — `SDFSamples` slow loading | issue #2 | Performance. Real, but it is an optimisation, and §8.0.F just rewrote the cache path it would target. |
| §9's fourth bullet — whether the LR bug moved published results | §9 | A research assessment of finished work, not a change to the library. |

**The test for this table.** An item belongs in §8 if a reader would call the outcome
"the code was wrong and now it is right". If the honest description is "we tried something
and measured what happened", it belongs in the ideas file. #3 (sigma) stays in §8 under
that test — the same number means two things and one of them is wrong — which is why it is
§8.0.Q and not a row here.

---

## 9. Deliverable: the bug provenance ledger

New file `docs/KNOWN_ISSUES.md`. For science code this is a first-class artifact —
it answers "which of my results are affected?", which a code comment cannot.

Each entry: what was wrong, exact date range affected, which configs/optimizers/code paths,
observable consequence, how to detect it in an existing run, how to reproduce old behaviour.

- [ ] Seed with the LR-schedule bug (May 2023 → Jul 2026, Adam/AdamW only, `schedule_free_*`
      unaffected). Prior to this it existed only as a docstring in a downstream fork — it
      must live somewhere durable and citable.
- [ ] Add the sigma coordinate-space ambiguity (issue #3).
- [ ] Add every subsequent finding from Phases 1–4.
- [ ] Assess whether the LR bug materially affected published/downstream results. Initial
      read: the hyperparameter search ran under the buggy mapping, so the chosen values were
      optimal *for that mapping*. The models are self-consistent; retuning under the fixed
      mapping is a separate exercise and is not a prerequisite for anything here.

---

## 10. Sequencing and risk

```
Phase 0 (scope)  ──►  Phase 1 (map)  ──►  Phase 2 (docs)
                            │                   │
                            └──────►  Phase 3 (tests) ──►  Phase 4 (decompose)
                                            ▲                      │
                                            └──── 7.3 just-in-time ┘
```

### 10.1 Versioning — decided 2026-08-15

The original text here said "before starting Phase 1, tag a release (`v2.x`)". That could
not be done as written: `pyproject.toml` derives the version from `NSM.__version__`, which
is the string literal `"0.0.1"` and has never been bumped, so there is no `v1` and `v2.x`
would invent history. Phase 1 ran without a tag; the tag is now a Phase 3 prerequisite
instead, since it is Phase 4 that breaks things.

- [x] **Tag `v0.1.0` — "the state before the refactor."** Have `kneepipeline` and `nsosim`
      pin it. This is the rollback point and it decouples their release cadence from this
      work. *(Tagged — the `Target` requirement is in it, see Surprises. The
      consumer-pinning half never happened: kneepipeline consumes a checked-out working
      tree (`DEPENDENCIES/nsm`), not a tag; nsosim is 0b's question.)*
- [x] **Not `1.0.0`.** That is a stability promise, and Phase 1 found 71 landmines with 30
      of them unverified on the production path. Claiming 1.0 and then breaking things in
      Phase 4 makes the number meaningless. `0.x` is honest and gives the same rollback
      guarantee. *(Holding: v0.1.0 and v0.2.0 both shipped under `0.x`.)*
- [x] **Bump on release, not on commit.** Under `0.x`: breaking changes bump the minor
      (`0.2.0` after Phase 4), additive changes bump the patch. *(Practiced: v0.2.0
      (PR #36); the pending Breaking set makes the next cut v0.3.0 — State § Versioning.)*
- [ ] **Move to `1.0.0` when there is something to promise** — when `__all__` exists (§3)
      and the §7.1 harness is green. That ties the version to a milestone rather than a date.
- [ ] **Derive the version from git tags.** `pyproject.toml` already lists `setuptools-scm`
      in `build-requires` with `[tool.setuptools_scm]` commented out. Uncommenting it makes
      the tag the single source of truth. A hand-edited literal is exactly why the version
      sat at `0.0.1` for years, and re-deciding the scheme without fixing that mechanism
      leaves it free to go stale again. *(Rides with the v0.3.0 release PR — State
      § Versioning.)*

**Coordinate with downstream forks throughout.** At least one active fork carries modules
that do not exist upstream, so every week of unmerged refactor makes its merge worse.
Phases 0 and 1 in particular should be shared before execution — fork module usage is an
input to the dead-code call.

**Biggest risk:** Phase 3 stalls again. Mitigation — §7.1 is a single bounded artifact with
a clear done condition, delivered before any broad coverage push. If only §7.1 and §7.2 ever
land, the library is still meaningfully safer than it is today.

---

## 11. Related documents

| Document | Status | Relationship |
|---|---|---|
| `.claude/plans/BREAKING_CHANGE_PROPOSAL.md` | Phase 1 partial | Fold into Phase 4 |
| `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md` | Not started | Fold into Phase 4 |
| `.claude/plans/HYBRID_OPTIMIZER_REPORT.md` | Findings, Aug 2025 | Reference for `reconstruct/main.py` |
| `.claude/plans/NSM_RECTIFIED_FLOW_CORRESPONDENCE.md` | Proposed | Blocked on stable interpolation API |
| `.claude/plans/NSM_TRAINING_IDEAS.md` | Open master list | Idea 3 (test Eikonal loss) belongs in Phase 3 |
| `.claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md` | Complete 2026-05-22 | Target-state example |
| `docs/MULTI_SURFACE_REGISTRATION.md` | Current | Feature doc, verify in Phase 2 |
