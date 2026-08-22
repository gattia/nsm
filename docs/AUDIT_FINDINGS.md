# Audit findings — triaged

> ## ⚠️ A staging document, not a register of facts.
>
> Every entry was **re-verified by execution** against `main` in Aug 2026 and sorted into
> what happens to it. The maintainer approved the § 0 disposition on 2026-08-22; the 22
> approved drafts are filed as **#40–#61**, #6 is closed as already-fixed, and the § 2
> folds are commented onto #20/#22/#23/#35. What remains before this file is deleted is
> listed in § 0.6.
>
> **Delete when:** the issues are filed, the prose corrections land, the `SCOPE` and
> `KNOWN_ISSUES` edits are made, and this file goes with the last of them, leaving a pointer
> to that PR. It is transitional debt by its own rule, and its line numbers have drifted once
> already — every anchor below is the ORIGINAL cited location, not a current one.

# 0. Final disposition — approved by the maintainer 2026-08-22, filed the same day

Three earlier analyses proposed three different issue sets (17 / ~8 / 6) with different
membership. This section supersedes all three. Where they disagreed, the disagreement was
settled **by execution**, not by averaging. The maintainer reviewed the drafts on
2026-08-22 (withdrawing draft 13 and settling D1–D4); the 22 surviving drafts were then
filed verbatim, extracted mechanically from § 0.3.

**Baseline at time of writing:** suite 356 passed / 1 skipped / 16 xfailed (64s);
`make lint` clean. Branch **`quick-wins`** (six commits off `main`) holds the ride-along
fixes — the `emd_{idx}` f-string key, the `(str,)` membership comma plus a tightened
test, four leftover debug prints, the bound-method latent-norm print, the
missing-schedulefree notice moved from stdout to `warnings`, and the `ARCHITECTURE.md`
§7 row correction — and stays green at the same counts.

## 0.1 Settled by execution this session

The consolidation brief's six contested items (its §3), plus its three warnings that
could be tested:

| Claim | Verdict | Key evidence |
|---|---|---|
| `remove_overlapping_points` correct only at exactly 2 surfaces | **CONFIRMED** | Every sign pattern enumerated for n=2..5 through the real method: n=2 exact; n=3/5 remove **nothing**; n=4 removes only inside-3-of-4. Nothing is ever wrongly removed. |
| `sum_sdf_features=False` trains one plane of three, silently | **CONFIRMED** | VAE emits (12,8,8); plane slices received xz=(12,8,8), yz=**(0,8,8)**, xy=**(0,8,8)**; forward raises nothing; output `torch.equal` to xz-alone. Reachable from `sum_conv_output_features` (`loader.get_model`). |
| EMD never worked from `compute_recon_loss` | **CONFIRMED** | pykeops `Vi()` rejects **any** numpy array (both dtypes tested; torch accepted); the oldest version in git history already passed numpy. Plus the `result["emd_{idx}"]` missing-`f` at the NaN fallback (now fixed on `quick-wins`). |
| Every `schedule_free_*` run dies at its first checkpoint | **CONFIRMED with the real package** (register had verified via stub only) | `schedulefree` 1.4.1 installed to an isolated dir; real CPU training run: epoch 1 trains, first checkpoint's eval warm-up hands `model(batch)` the raw dataloader list → `TypeError` in `TriplanarDecoder.forward`. |
| `get_optimizer` drops `weight_decay` on the Adam branch | **CONFIRMED** | Direct read: `torch.optim.Adam(list_params)` bare; AdamW and schedule_free both pass it. |
| `MultiSurfaceSDFSamples` declared defaults raise | **CONFIRMED** | Already execution-verified in group 3 below. |
| "Unassigned (62)" heading | **WRONG — 45 + 16** | Titles compared mechanically against the grouped entries: 45 verbatim duplicates, 16 genuine. Heading corrected in place below. |
| `ARCHITECTURE.md` §7 "each builds fine and raises on first use" | **WRONG for 2 of 5** | `TwoStageDecoder()` raises in `__init__`; `Decoder(norm_layers=...)` builds *and forwards* under the shipped contiguous default with weight-norm on. Corrected on `quick-wins`. |
| Issue #6 vs group 10 | **Zero overlap; #6 already fixed** | The one deprecated call site now uses `n_faces_strict`; the sole remaining `n_faces` hit is a wandb dict-key name. Close #6 on its own merits. |

Shipped-config values these dispositions rely on, re-verified against both
`647_nsm_femur_v0.0.1` and `551_nsm_femur_bone_v0.0.1`: `optimizer: "AdamW"`,
`weight_decay: 0.0001`, `l2reg_recon: false`, `get_rand_pts_recon: false`,
`convergence_type_recon: "recon_loss"`, `scale_jointly: true`,
`sum_conv_output_features: true`, `conv_pred_sdf: false`.

## 0.2 Corrections to this register's own instructions

1. **`weight_decay` and `get_latent_vecs` must not fold into #20** (§2's fold list says
   to). #20's closure criterion — "does each parameter name appear in the body" — passes
   green with both bugs intact, and #20's Trap 1 (delete the parameter) cannot apply to
   either. They become drafts 15 and 16.
2. **Do not close #6 against group 10** (group 10's header says to). See table above.
3. **Group 14 is demoted from issue to `ARCHITECTURE.md` trap.** Its production-impact
   claim was refuted (`l2reg_recon: false` in both shipped configs), and
   `UniqueConsecutive` and `FastUnique` amplify identically — it is a long-standing
   convention, and patching one path alone would desynchronise the two decoder
   interfaces. Trap committed on this branch.
4. **Group 5 files from its evidence, not its headline.** The wrong-ground-truth trigger
   is *decoder* count (two or more multi-output decoders), not surface count; the
   surface-count half is `remove_overlapping_points`. Draft 5 states both precisely.
5. **Two of the sixteen genuine Unassigned entries were headed for the discard pile and
   are instead the two best silent-wrong-results bugs of the audit** (drafts 6 and 8).
   The brief's warning that grouping loses singleton bugs was correct.
6. **Execution verifies mechanisms, not intent.** Maintainer review of the drafts
   caught one verdict that flips on design intent (draft 13: the variational doubling
   is the standard VAE mean+logvar parameterisation, and KLD replaces the norm bound —
   deliberate, undocumented). The drafts where intent could likewise change the fix —
   not the existence of the finding — now carry an explicit intent-check flag
   (drafts 11 and 18).

## 0.3 Issue drafts — file only after approval

Twenty-two live drafts (draft 13 was withdrawn in maintainer review — kept in place
below as the worked example of the intent-blind failure mode). On the tracker they get
no tiers — CLAUDE.md orders issues by `file:function`; the two-tier split below exists
only to make this review faster. Every draft is self-contained (this register is
deleted when the drafts land, so the evidence a closer needs is in the draft, not
here).

**Filed 2026-08-22**, draft → issue: 1→#40, 2→#41, 3→#42, 4→#43, 5→#44, 6→#45, 7→#46,
8→#47, 9→#48, 10→#49, 11→#50, 12→#51, 14→#52, 15→#53, 16→#54, 17→#55, 18→#56, 19→#57,
20→#58, 21→#59, 22→#60, 23→#61. Alongside: #6 closed as already-fixed (`0aee8ad`), and
the § 2 folds commented onto #20, #22, #23 and #35.

### Tier 1 — silent wrong results, or blocks a run someone would launch

---
**Draft 1 — `sdf_dataset.read_mesh_get_sampled_pts` / `read_meshes_get_sampled_pts`: the
single- and multi-mesh samplers have diverged (uniform box, clipping, return types)**

Three divergences between two nominally parallel functions. (a) In both copies, `mins` is
rebound before `maxs` reads it, so a nonzero `uniform_pts_buffer` grows the sampling box
more above than below and moves its centre — dormant only because the default is 0.0,
and commit `48c5f60` added the parameter precisely so it could be nonzero. (b) The
single-mesh function clips uniform samples (`np.clip`, one hit in the file); the
multi-mesh one does not: with `uniform_pts_buffer=0.5, norm_pts=True`, single spans
±1.2500, multi ±1.5626. (c) `pts_surface` is a `list` from the single-mesh function and
an `int64 ndarray` from the multi — no in-repo consumer takes `.shape` of the single
value today, so (c) is recorded here for the fix, not as its own defect.
**Reachability:** training-data path; production reconstruction is gated off it
(`get_rand_pts_recon: false` in both shipped configs).
**Fixed means:** the span is captured once before rebinding so the box is symmetric;
the two functions agree on clipping (decided, not accidental); return types match.
Changes cached dataset content for nonzero buffers → `KNOWN_ISSUES.md` § History entry
if any real run used one.

---
**Draft 2 — `sdf_dataset.sdf_pos_neg_idx`: divides by zero when a surface has no
positive or no negative samples**

`ZeroDivisionError` at both call sites (`SDFSamples`, `MultiSurfaceSDFSamples`), end to
end via two realistic triggers: a `None` surface (the `fdfe902` feature) and one surface
nested inside another. The regression harness's own fixtures are shaped to dodge it
(`testing/NSM/regression/_harness.py` documents this) — the codebase already works
around its own bug. **Fixed means:** an empty pos/neg set raises an error naming the
surface (or is handled), and the harness fixture comment points at the fix instead of
the dodge.

---
**Draft 3 — `train_deep_sdf`: every `schedule_free_*` run dies at its first checkpoint
or validation epoch**

The eval warm-up (`optimizer.eval()` then `for batch in itertools.islice(data_loader,
50): model(batch)`) hands the decoder the raw dataloader item. Verified 2026-08-21 with
the real `schedulefree` 1.4.1: epoch 1 trains, the first checkpoint epoch raises
`TypeError: list indices must be integers or slices, not tuple` in
`TriplanarDecoder.forward`. Every run reaches it: `checkpoint_epoch = epoch in
config["checkpoints"] or epoch % config["save_frequency"] == 0`. Invisible until now
because `schedulefree` is not installed in `nsm-dev` (`get_optimizer` raises
`ImportError` first). **Fixed means:** the warm-up unpacks the batch the way
`train_epoch` does, pinned by a test that trains a schedule_free run through one
checkpoint. Note for the fix: KNOWN_ISSUES §1 records that schedule_free configs were
tuned against the Adam path — this crash is why nobody noticed.

---
**Draft 4 — `MultiSurfaceSDFSamples`: the declared constructor surface does not work**

(a) The documented default `subsample=None` cannot construct — `TypeError` in
`get_samples_per_sign` (`float * NoneType`); on a warm cache it instead skips joint
normalization and returns 600 unnormalised points with a different key set (absmax
1.055 > 1). (b) `joint_scale_buffer` — which sets the normalization radius of every
shipped multi-surface dataset (0.1 in production) — is not accepted or forwarded by the
multi-surface constructor at all (`TypeError: unexpected keyword argument`).
**Reachability:** multi-surface (bone+cartilage) is the production training
configuration. **Fixed means:** `subsample` is required or validated at construction;
`joint_scale_buffer` is accepted, forwarded, and lands in the cache key (#19's half).

---
**Draft 5 — multi-surface above the shipped two-surface configuration silently computes
the wrong thing (`reconstruct.main` decoder indexing; `remove_overlapping_points`)**

Two mechanisms, one class, both measured. (a) With **two or more multi-output
decoders** (a configuration `reconstruct_mesh`'s own docstring advertises), the
single-output branch indexes the flat `sdf_gt` by decoder index: with two 2-surface
decoders and four ground-truth surfaces, setting surfaces 2 and 3 to all-NaN left the
loss **bit-identical** — decoder 1 never reads its own ground truth. (b)
`remove_overlapping_points` tests `total != -2`, a **sum**, not a count: correct at
exactly 2 surfaces, removes nothing at 3 or 5, and at 4 removes only inside-3-of-4
points (full n=2..5 enumeration, 2026-08-21; it never wrongly removes). CLAUDE.md
documents 4-surface models (`bone/cart/med_men/lat_men`) as supported.
**Fixed means:** (a) the flat `sdf_gt` is indexed by a running surface offset or the
configuration is rejected at entry; (b) "inside two or more surfaces" is expressed as a
count — `(sdf < 0).sum(1) >= 2`. **(b) changes what points a dataset keeps → maintainer
decision D2 + `KNOWN_ISSUES.md` § History entry before it lands.**

---
**Draft 6 — `TriplanarDecoder`: `sum_sdf_features=False` silently trains on one plane
of three**

`__init__` sizes the VAE output by `sdf_latent_size` when `sum_sdf_features is False`,
but `forward_with_plane_features` slices by `sdf_latent_size + conv_pred_sdf` per
plane: the xz plane receives all 12 channels, yz and xy receive **(0, 8, 8)** —
zero-channel slices. No error; forward output is `torch.equal` to using the full
feature map as the xz plane alone; all VAE parameters still receive gradient (through
xz geometry), so training proceeds and converges to a silently degraded model.
**Reachability:** `loader.get_model` maps the config key `sum_conv_output_features`
onto it; both shipped configs set `true`, so no shipped model is affected — this hits
anyone exploring the documented option. (Ride-along: the assert message above the
sizing branch says "if sum_sdf_features is True" while guarding the False branch.)
**Impact upgrade (maintainer, 2026-08-22):** the maintainer's own past experiments ran
this option and concluded concatenation never improved results — those runs silently
trained one plane of three, so that conclusion is untrustworthy and the comparison is
queued for a re-run after the fix (`NSM_TRAINING_IDEAS.md`). The fix therefore needs a
`KNOWN_ISSUES.md` § History entry ("affected: any run with
`sum_conv_output_features: false`").
**Fixed means:** the flag either produces three correctly-sized plane slices or is
rejected at construction; pinned by a forward-shape test over both flag values.

---
**Draft 7 — `models`: configurations that construct successfully and crash on first
forward (class issue)**

Four instances, verified: `Decoder(progressive_add_depth=True)` propagates `None`
through the layer stack in a window covering every realistic epoch range;
`Decoder(norm_layers=...)` indexes `self.bn` by absolute layer index but appends only
per norm layer (`IndexError` for any set not starting at 0, with weight-norm off;
`norm_layers` is marked DEPRECATED, so the fix may be deletion); `activation='linear'`
returns a bare `None` the forward then calls; `TwoStageDecoder()` raises in `__init__`
(tuple + list concat) at any argument. **Closure criterion (what makes this class
sweep able to fail):** a parameterised constructor-and-one-forward smoke test over the
documented option values — each option either works or refuses at construction.

---
**Draft 8 — `utils.get_optimizer`: `weight_decay` is silently dropped on the Adam
branch**

`torch.optim.Adam(list_params)` — no `weight_decay` — while `AdamW` and
`schedule_free_AdamW` both pass it. `train_deep_sdf` forwards
`config["weight_decay"]` unconditionally, so every Adam config that sets it (the
regression harness's own config: Adam + 1e-4) silently trains without decay.
**Reachability:** both shipped configs use AdamW → **no shipped run affected**.
**Fixed means:** the argument is passed (decision D3: honour vs reject) — honouring it
moves the committed regression baselines, which is the proof it changes numerics, so
the fix regenerates baselines and adds a `KNOWN_ISSUES.md` § History entry.

---
**Draft 9 — `train_deep_sdf` cannot be driven by the shipped `default_config.json`**

Five unconditionally-read keys are missing and fatal in sequence, starting with
`KeyError: 'prefetch_factor'` (verified by adding one key at a time on a real CPU run);
`assd` is also read but unreachable from the shipped default. Related option values
that cannot work, folded here rather than filed separately:
`add_plain_lr_to_config` raises `KeyError: 'Initial'` on a Constant schedule that
`get_learning_rate_schedules` itself accepts; `norm_penalty_type='barrier'` returns NaN
for any latent outside `[min,max]` — the state at initialisation — so the option is
broken for its whole intended use; `Regress.add_latent` is handed the whole result dict
(`TypeError` for every user who enables the latent-to-factor validator).
**Fixed means (decided by maintainer, 2026-08-22):** regenerate `default_config.json`
**from the ShapeMedKnee `647` config** — the values that actually trained the shipped
models — updated to the current contract: `Target`-annotated `LearningRateSchedule`
entries, the `emd` key dropped (decision D1), and every key the trainer reads
unconditionally present. Pinned by the existing generator-sync test plus a new test
that instantiates the trainer from the shipped file. (This is the plan's §8.1 item
"a default config per model type, derived from the ShapeMedKnee configs" — this draft
delivers the first one.) The three option values either work or raise with a named
message.

---
**Draft 10 — `train_deep_sdf`: `resume_epoch == 1` silently skips epoch 1 without
resuming anything**

Verified end to end on the CPU harness: `resume_epoch=0` runs epochs [1,2,3,4];
`resume_epoch=1` runs [2,3,4] with **no checkpoint loaded** — the resume guard and the
loop boundary disagree. **Fixed means:** `resume_epoch==1` either loads the epoch-1
checkpoint or raises; both guards share one boundary; pinned by a resume test.

---
**Draft 11 — `save_model_params` silently refuses to overwrite and silently drops
non-JSON config values**

Called on every checkpoint, no-ops after the first: a resumed or re-configured run's
`model_params_config.json` — the file every downstream consumer reads to rebuild the
model — keeps the first run's hyperparameters and mesh list (verified: second call with
`lr=0.9999` leaves `lr=0.001` on disk, no log). `filter_non_jsonable` drops keys with
no log. **Intent check for the maintainer:** write-once *could* be deliberate
provenance protection — if so, the fix is to log the refusal and the dropped keys
loudly rather than to overwrite. **Fixed means:** overwrite or epoch-stamp (or, if
write-once is intended, warn loudly on divergence), and log every key removed.

---
**Draft 12 — `train_deep_sdf_multi_head`: repair checklist (SCOPE §2.1: supported,
broken, fix it)**

KNOWN_ISSUES § History §2 owns the headline bug (only the last decoder trains). The
repair checklist, each verified: latents are never moved to the device (runs only
*because* they are left on CPU; `device="cpu"` and `"mps"` both crash); non-short-
circuit `&` raises `KeyError: 'surface_weighting'` on the shipped default config
(→ one-sentence addition to History §2, committed on this branch);
`torch.mps.empty_cache()` on the CPU branch; per-surface L1 appended to a fixed-size
list then discarded; a hardcoded 100-epoch warm-up ignores its config key. Fold in:
multi-decoder checkpoints are written to `model_N/` subdirectories **no loader in the
repo can read back** (`save_model` naming). **Fixed means:** one epoch trains on the
shipped default on cpu and cuda, all decoders receive gradients, and a saved
multi-head run can be loaded.

---
**Draft 13 — WITHDRAWN (maintainer review, 2026-08-22): the variational behaviour is
deliberate, not a defect**

The mechanism was verified but the intent was misread, in both halves. The doubling is
the standard VAE parameterisation — the embedding stores mean and log-variance, so its
width is `2 × latent_size` while the decoder's latent is still `latent_size` (the
recorded value is therefore *correct* for consumers). And the bound is not "ignored":
when `variational` is on, training swaps the regularizer to KLD
(`train_deep_sdf.train_epoch`), which replaces the hard `max_norm` — the hardcoded
1000 is "effectively unbounded" by design. What is actually missing is documentation:
`get_latent_vecs` has no docstring, and nothing states that `latent_bound` is
superseded under `variational`. **Routed to the Phase-2 prose pass**, not the tracker.

Kept in place, numbering preserved, because this is the register's one confirmed case
of the intent-blind failure mode: a mechanism verified by execution whose verdict flips
on design intent only the maintainer holds. It is why every draft here waits for this
review, and why drafts 11 and 18 now carry explicit intent-check flags.

---
**Draft 14 — `train_deep_sdf`: `mesh_names` is written to `model_params_config.json` as
ground truth and can be silently wrong**

The per-surface index ordering is a positional contract spanning four modules; training
is self-consistent under a swap (the decoder learns whichever column it is given), so
the harm is precisely that the persisted `mesh_names` — added to prevent downstream
misidentification — can disagree with what each output channel actually learned.
**Fixed means:** the dataset carries surface identity from mesh list to output channel,
and `mesh_names` is validated against it (or derived from it) at save time.

### Tier 2 — loud failures in supported paths, API hygiene, error quality

---
**Draft 15 — EMD is dead end to end; repair or delete (decision D1)**

`compute_recon_loss(calc_emd=True)` has never returned a number from any caller in the
function's history: the only caller passes numpy and pykeops rejects numpy at the
boundary for every dtype. Behind that: `sinkhorn`'s default uniform weights cannot sum
equal for unequal point counts (so real mesh pairs would fail next), `max_iters` is
type-checked as `p`, `w_y`'s length is never validated (`w_x` is, twice), and
`default_config.json` ships `emd: true` while the trainer reads the key
unconditionally. The file is a vendored copy of `fwilliams/scalable-pytorch-sinkhorn`.
**Repair** = numpy→torch conversion at the caller + weight normalisation + the two
validation lines + a test on unequal-size meshes. **Delete** = remove `calc_emd`, the
vendored file, and the config key, with a deprecation note. Either way the shipped
default stops advertising a dead option.
**DECIDED (maintainer, 2026-08-22): delete.** Two supporting facts: both ShapeMedKnee
configs set `emd: false`, so even validation never attempted it on a real run (only
`default_config.json` says `true`); and EMD appears nowhere in the training-loss path
in this repo's history — if it was ever used as a loss, it was not through this code.

---
**Draft 16 — unhandled inputs surface as `UnboundLocalError`/`NameError`/invented data
instead of a named error (class issue, five sites)**

All verified: `refine_mesh.get_target_cells` raises `UnboundLocalError` **on its own
default arguments** (makes the whole module unusable; gates SCOPE §2.3's other
conditions); `reconstruct_latent`'s `optimizer`/`loss_fn` binding has no else-raise
(`optimizer_name='AdamW'` → `UnboundLocalError`); `reconstruct_latent` can return an
unbound `latent_` — one trigger is NaN loss under `convergence="recon_loss"`, **the
mode both shipped configs select**, so a diverged production fit reports
`UnboundLocalError` instead of "the fit diverged", and the same block initialises
`loss`/`recon_loss` to the literal `100`; `score_correspondence` fabricates a
plausible roundtrip metric when `source_mesh` is missing where every sibling path
returns `{'skipped': True, ...}`. **Fixed means:** explicit else-that-raises naming
accepted values; `latent_` initialised before the loop; the `100` sentinel replaced
with `inf`/`None`; the missing-input path skips with a reason. One shape, one PR.

---
**Draft 17 — functions that mutate a caller's object and also return it (PR #38's
unswept siblings)**

The class PR #38 just fixed in `get_pts_center_and_scale`, three more instances,
verified: `reconstruct_latent` clamps and device-moves the caller's `sdf_gt` list in
place through two undocumented helpers; `compute_recon_loss` downcasts the caller's
meshes to float32 in place (aliases `meshes` and `result_['orig_mesh']`);
`interpolate_mesh`'s `is_mesh` path returned the caller's own 82-point mesh at 28,002
points. **Fixed means:** each site copies, or documents the mutation and stops
returning the object; docstrings say which.

---
**Draft 18 — the same knob has a different default at each layer; adjacent metrics take
their arguments in opposite order**

Verified: `chamfer_norm` defaults to 2 in `reconstruct_mesh` and 1 in the two layers
below it — it is a **power**, so the layers report chamfer in different units (0.3297
vs 0.2179 on identical geometry), documented at the two lower layers and absent from
`reconstruct_mesh`'s docstring; `sigma_rand_pts` differs 10× between `reconstruct_mesh`
(0.001) and `get_mean_errors` (0.01), result-changing whenever `get_rand_pts=True`
(both shipped configs: false); `conv_norm_type` defaults to `'batch'` at three of six
construction sites and `'layer'` at the others while both shipped models use
`'layer'`; `roundtrip_distance(A,B)` equals `roundtrip_distance(B,A)` exactly while
its adjacent metric is sign-flipped under the same swap — a swap is invisible.
**Intent check for the maintainer:** each default divergence *could* encode a
deliberate per-layer choice (e.g. squared chamfer at the top layer); the defect that
survives either way is that the divergence is undocumented and the knob shares one
name. Which value wins is the maintainer's call per knob. **Fixed means:** one default
per knob, sourced once (or the divergence documented as deliberate at both ends); the
metric pair keyword-only or signature-aligned so a swap is a `TypeError`.

---
**Draft 19 — face arrays are reshaped without validation; a quad or VTK-style array
silently builds garbage (five sites, one helper)**

Verified: pure-quad input raises a bare reshape `ValueError` in two metrics; a mixed
quad/triangle mesh whose flat length happens to divide evenly **silently corrupts**;
`build_mesh_laplacian` accepts pyvista's VTK-style `.faces` (384 % 3 == 0) and builds a
different smoothing operator (nnz 373 vs 288, dense matrices unequal) — the
interpolation output is wrong rather than absent. **Fixed means:** one shared accessor
validates (or uses `regular_faces` and raises on non-triangular input); all five sites
route through it. **Not** #6, which is closed separately as already-fixed.

---
**Draft 20 — library code prints to stdout ungated: route the reconstruction and
sampling paths through `verbose`/logger (post-quick-wins remainder)**

Measured: `get_sdfs` prints one line per batch with **no `verbose` parameter at all**
while every sibling has one — 64 lines per `create_mesh_adaptive` call at the caller's
own defaults; timing prints on the sampler path (five sites) and unconditional prints
in `reconstruct_mesh`/`get_mean_errors` survive `verbose=False`. The four worst
one-liners are already deleted on `quick-wins`; this issue is the systematic pass.
**Fixed means (direction decided by maintainer, 2026-08-22): loggers, not `verbose`
flags.** Every print on the reconstruction and sampling paths moves to a module logger
(`logging.getLogger(__name__)`) so output is controlled centrally; existing `verbose`
parameters become sugar over the log level or are removed. Part of the same fix:
`reconstruct/main.py` calls `logging.basicConfig` at import time — a library must not
configure the root logger, so that call goes (it is one of ARCHITECTURE §4's
import-time side effects). A capsys test pins stdout silence at defaults for the
production entry points.

---
**Draft 21 — `train_deep_sdf` logging: latent-norm stats are assigned, not accumulated;
the LR fix's positional back door survives in the logging helper**

Verified: `step_mean_vec_length`/`step_std_vec_length` use `=` where the surrounding
accumulators use `+=`, so the logged latent-norm metrics are the last chunk over
`n_batches` — wrong by a factor of `len(data_loader)` (real 2-epoch run: true mean
0.0107, logged 0.0053, × n_batches matches). Gradients unaffected; wandb only. And
`add_plain_lr_to_config` retains `idx_model`/`idx_latent` overrides whose only caller
is a test asserting deliberately swapped labels — in the very function whose Aug-2026
fix eliminated positional mapping. **Fixed means:** `+=`; delete the two parameters
and their test.

---
**Draft 22 — `mesh.main`: `sdf_grid_to_mesh` crashes on numpy input while its VTK twin
does not; the fallback grid origin can disagree with `search_bounds`**

Two functions swapped by one unrelated boolean accept different inputs (unguarded
`.cpu()` vs `hasattr` guard) and carry different `narrow_band` defaults; the
no-`voxel_origin` fallback hands `create_mesh` the default origin `(-1,-1,-1)` while
`voxel_size` is derived from `search_bounds` — a wrong-by-construction grid on a
reachable branch. Production always passes torch + `use_vtk=True`, so severity is API
hygiene. **Fixed means:** both twins guard the same way and share defaults; the
fallback derives its origin from `search_bounds`.

---
**Draft 23 — `sdf_dataset`: the multi-surface reference-mesh path cannot run, and
`combine_meshes` breaks its own return contract**

`reference_mesh=int` with a list-valued `mesh_to_scale` raises `UnboundLocalError` one
statement before the `AttributeError` the audit predicted; `combine_meshes` returns a
pyvista `PolyData` (no `save_mesh`) whenever it actually combines two or more meshes,
against its own "Returns: Mesh" docstring. `docs/MULTI_SURFACE_REGISTRATION.md`
advertises the path as working. **Fixed means:** the path returns a usable combined
pymskt Mesh, which requires `combine_meshes` to keep its declared type; pinned by a
test exercising `reference_mesh=int` with two meshes.

## 0.4 Decision items — resolved by the maintainer 2026-08-22 except D4

- **D1 (draft 15): EMD — DECIDED: delete.** Never worked from any caller, nothing
  downstream consumes it, the maintainer does not use it, and the vendored sinkhorn
  file is maintenance surface.
- **D2 (draft 5b): `remove_overlapping_points` — DECIDED: fix it**, together with the
  class-5 indexing fix, one `KNOWN_ISSUES.md` § History entry covering both. Changes
  what points a dataset keeps for 3+ surfaces; not a drive-by.
- **D3 (draft 8): `weight_decay` under Adam — DECIDED: honour it.** Pass the argument;
  regenerate the harness baselines (the moved baselines are the proof it changes
  numerics) and add the History entry. No shipped run affected (both use AdamW).
- **D4 (group 14): latent-gradient N-amplification — DECIDED (maintainer,
  2026-08-22): keep the behaviour, document it twice, revisit deliberately.** No
  tracker issue. The ARCHITECTURE §6 trap stands, and a `KNOWN_ISSUES.md` § Open
  entry (committed on this branch) frames the revisit: the maintainer reports latent
  regularization was historically a pain to tune and was abandoned — consistent with
  the effective weight being silently divided by N, recorded there as a hypothesis
  with the experiment that would test it.

## 0.5 Non-issue dispositions

- **§2 fold list stands**, minus the two pull-outs (0.2.1). #6 closes as fixed on its
  own merits. #16 is absorbed by #20 as already noted there. The S3 findings
  (`reconstruct_latent_S3` unguarded arithmetic, undefined name in its error path,
  wandb-without-import) fold into **#35** as "the S3 copy of the main-path arithmetic
  is unguarded".
- **§3's 62 prose corrections**: land as Phase-2 commits, no tracker entries.
- **§4's 13 SCOPE rulings**: accepted as written; land as one SCOPE.md PR. Add to it
  the dead-public-symbol cluster from the Unassigned pile (`symmetric_chammfer` — an
  empty stub returning `None` with a whitespace docstring; `sdf_gradients` — returns
  98.8% fabricated zeros; `find_object_bounds_random_sampling`): rule dead, delete.
- **§5's `grad_clip` entry**: committed to `KNOWN_ISSUES.md` § Open on this branch,
  with the multi_head History sentence and the `ARCHITECTURE.md` latent-gradient trap.
- **§6 deletions**: accepted. The refuted-entry evidence stays until this file goes.
- **The 45 duplicated Unassigned entries** resolve with their groups. The 16 genuine
  ones: drafts 3, 6, 10, 12 (×4 sites + `save_model` naming), 14, 22; the withdrawn
  draft 13 → Phase-2 prose (document `get_latent_vecs`'s variational contract:
  mean+logvar doubling, KLD supersedes `latent_bound`); #35 fold; SCOPE dead-symbol
  ruling (×2); `KNOWN_ISSUES` F401 entry (already on this branch, covers the
  unused-import instance).

## 0.6 Delete-when, updated

This file is deleted by the PR that lands the last of: the approved drafts filed, the
`quick-wins` branch merged, the SCOPE.md rulings PR, and the Phase-2 prose pass — that
PR leaves a pointer here → tracker.

---

## What the triage found

240 entries, nine agents, each required to run something rather than read.

| Verdict | Count | | Outcome | Entries |
|---|---|---|---|---|
| Still reproduces | 193 | | New issues (17 proposed) | 62 |
| Already fixed | 19 | | Prose fix in Phase 2 | 62 |
| Not a defect — overstated | 19 | | `SCOPE.md` ruling | 13 |
| Does not reproduce | 9 | | `KNOWN_ISSUES.md` | 1 |
| Could not verify | 0 | | Deleted | 102 |

**The register was more accurate than its own banner claimed.** It warned that inference had
been wrong "in the direction of overstatement" two times out of two. Measured: ~9% flatly
wrong, plus a tail of ~30 where the mechanism was exactly where described but the stated
consequence was not reachable. Every mechanism was found modulo line drift; nothing was
fabricated.

**Grouping is the result.** 58 entries that each looked like an issue collapse to 17, being
instances of shared classes. Filing per entry would have produced tickets that each look
trivial and none of which closes a class — the failure #19 and #20 exist to avoid.

---

# 1. Proposed new issues

File the group, not the entries.

> ### ⚠️ The mechanisms are verified. The *impact* claims are not.
>
> Every entry below was reproduced by execution, and in this triage no mechanism was
> fabricated. But three rounds of agent work on this repo have now shown the same pattern:
> **the mechanism is found exactly where described, and the stated consequence is
> overstated.** The reconciler put ~30 entries in that category itself.
>
> So a sentence like "enabled in production" or "both shipped configs set this" is a
> *hypothesis about blast radius*, separate from the defect, and has to be checked against
> `kneepipeline/config.json` and the shipped `model_params_config.json` files before it is
> repeated in an issue.
>
> **Two of seventeen have been checked so far.** They came out differently — one confirmed,
> one refuted. The other fifteen carry their agent's impact claim unverified; they are
> marked `IMPACT UNVERIFIED` below.

### 1. The single- and multi-mesh samplers have diverged: uniform box, clipping and return types

Arithmetic bug present in both copies (mins is rebound before maxs reads it, so the box grows
1+buffer/2 times more above than below and its centre moves), plus a clip that exists in only
one of the pair. Dormant only because uniform_pts_buffer defaults to 0.0, and 48c5f60 added it
to be nonzero.

<details><summary>Folds in 3 entries</summary>

**NSM/datasets/sdf_dataset.py:301 — uniform_pts_buffer expands the max side more than the min side**  
`REPRODUCES`

Straightforward arithmetic bug in both copies, dormant only because the default is 0.0 — and
commit 48c5f60 added the parameter precisely so it would be nonzero. "Fixed" means capturing
the span once before rebinding mins, so the box grows symmetrically. Not covered by any open
issue (#19 mentions uniform_pts_buffer only as a missing cache-key input).

*Evidence:* Now at :357-358 (single) and :704-705 (multi), identical copy-paste. I exec'd the
two lines pulled straight out of the module source rather than retyping them: mins = mins -
uniform_pts_buffer / 2 * (maxs - mins) maxs = maxs + uniform_pts_buffer / 2 * (maxs - mins) #
mins already rebound box …

**NSM/datasets/sdf_dataset.py:308 — Uniform-sample clipping exists only in the single-mesh function**  
`REPRODUCES`

Two nominally parallel samplers produce different sampling domains for the same request. Same
root area as :301 so it should be one issue with it. "Fixed" means the two functions agree on
the domain (either both clip or neither does, decided deliberately).

*Evidence:* grep 'np.clip' NSM/datasets/sdf_dataset.py -> one hit, :366, inside
read_mesh_get_sampled_pts. read_meshes_get_sampled_pts has no equivalent after :704-705. Ran
both with uniform_pts_buffer=0.5, norm_pts=True, sigma=None: SINGLE: min -1.2500 max 1.2500
(== clip_val 1 + buffer/2) MULTI : min -1.5626 …

**NSM/datasets/sdf_dataset.py:316 — pts_surface return type differs between the single- and multi-mesh functions**  
`REPRODUCES`

Reproduces, but no in-repo consumer takes .shape of the single-mesh value, so there is no
defect today — it is one more face of the sampler-divergence class (with :308 and :314). Fold
the observation into that class if it is fixed; do not file it.

*Evidence:* Ran both on real meshes: read_mesh_get_sampled_pts -> type(results['pts_surface'])
= list read_meshes_get_sampled_pts -> type = ndarray, dtype int64, shape (100,)
r1['pts_surface'].shape -> AttributeError: 'list' object has no attribute 'shape' Current
lines :367 / :372 (list) vs :738 (ndarray).

</details>

### 2. sdf_pos_neg_idx divides by zero when a surface has no positive or no negative samples

Two realistic triggers reproduced end to end: a None surface (the fdfe902 feature) and one
surface nested inside another. testing/NSM/regression/_harness.py:341-343 documents the second
and shapes its fixtures to dodge it — the codebase already works around this bug in its own
tests.

<details><summary>Folds in 1 entries</summary>

**NSM/datasets/sdf_dataset.py:1305 — sdf_pos_neg_idx divides by zero when a surface has no positive or no negative samples**  
`REPRODUCES`

Reproduces at both call sites and end-to-end via two realistic triggers (a None surface, and
one surface nested inside another). Not tracked by any open issue, and the codebase's own test
fixtures were shaped around it — that is evidence plus a fixable statement: guard the repeat
so an empty pos/neg set raises something that names the surface, or is handled.

*Evidence:* Code now at NSM/datasets/sdf_dataset.py:1385-1386 (SDFSamples) and :2124-2125
(MultiSurfaceSDFSamples). Unit probe (scratchpad/e1_zerodiv.py, e1b.py): single:
ZeroDivisionError integer division or modulo by zero multi: ZeroDivisionError integer division
or modulo by zero NaN column (None mesh): …

</details>

### 3. MultiSurfaceSDFSamples' declared constructor surface does not work

The documented default subsample=None cannot construct (TypeError in get_samples_per_sign); on
a warm cache subsample=None additionally skips joint normalization, returning 600 unnormalised
points with a different key set (absmax 1.055 > 1); and joint_scale_buffer — which sets the
normalization radius of every shipped multi-surface dataset — is not accepted or forwarded by
the multi-surface constructor at all. One constructor, one PR.

<details><summary>Folds in 3 entries</summary>

**NSM/datasets/sdf_dataset.py:1598 — MultiSurfaceSDFSamples default subsample=None is unusable**  
`REPRODUCES`

The documented default of a public constructor cannot construct. Reproduces both in isolation
and end-to-end, and is not tracked. Fixed = subsample is required (or validated) rather than
defaulting to a value nothing supports. Same class as the :1487 entry; file them together.

*Evidence:* Default still `subsample=None` at :1684. End-to-end with the regression harness
and real meshes (scratchpad/e10_real.py): CONSTRUCTION ERROR: TypeError unsupported operand
type(s) for *: 'float' and 'NoneType' Path: __init__ -> run_before_loading_data (:1786) ->
get_samples_per_sign (:1983), line …

**NSM/datasets/sdf_dataset.py:1487 — Joint (scale_jointly) normalization is skipped entirely when subsample is None**  
`REPRODUCES`

Reproduces, and the returned item differs from the subsampled case in shape, key set and
coordinate space — the trainer reads timing keys unconditionally, so this cannot train either.
Fixable statement: reject subsample=None at construction, or make it produce the same contract
as a subsampled item. Groups with the :1598 entry.

*Evidence:* Code now at :1573 (`if self.subsample is not None:`) with the joint block at
:1635, and :2176 / :2240 for the subclass. Cold construction with subsample=None raises before
it matters (scratchpad/e8_real.py): TypeError: unsupported operand type(s) for /: 'NoneType'
and 'int' at :1384 in …

**NSM/datasets/sdf_dataset.py:1595 — joint_scale_buffer cannot be set on MultiSurfaceSDFSamples**  
`REPRODUCES`

Reproduces exactly as written. Multi-surface is the production configuration (bone+cartilage),
so the 10% joint-scaling buffer that sets the normalization radius of every shipped multi-
surface dataset is unreachable from the public constructor. Fixed = the parameter is accepted
and forwarded (and lands in the cache key, which is #19's half).

*Evidence:* scratchpad/e9_jsb.py, against real meshes: SDFSamples has joint_scale_buffer: True
Multi has joint_scale_buffer: False passed joint_scale_buffer -> TypeError __init__() got an
unexpected keyword argument 'joint_scale_buffer' multi dataset joint_scale_buffer attr: 0.1
Current lines: declared …

</details>

### 4. The multi-surface reference-mesh path cannot run

reference_mesh=int with a list-valued mesh_to_scale raises UnboundLocalError at :1440, one
statement before the AttributeError the register predicted; combine_meshes also returns a
pyvista PolyData whenever it actually combines, contradicting its own 'Returns: Mesh'. Found
independently by two slices, one of them arriving via a documentation anchor. The doc
advertises it as working.

<details><summary>Folds in 1 entries</summary>

**NSM/datasets/sdf_dataset.py:2193 — combine_meshes returns a pyvista PolyData, not a pymskt Mesh, whenever it actually combines (verified)**  
`REPRODUCES`

The type inconsistency is real and the docstring contradicts it, but the entry's failure story
is wrong: the multi-surface reference-mesh path dies one statement earlier with
UnboundLocalError. Worth one issue covering the whole path — fixed = reference_mesh=int with a
list-valued mesh_to_scale returns a usable combined pymskt Mesh, which requires combine_meshes
to keep its declared return type. (docs/MULTI_SURFACE_REGISTRATION.md:75 advertises this as
working; that doc entry is in another slice — fold them.)

*Evidence:* Type claim, run against pymskt (scratchpad/e13_combine.py); combine_meshes is now
at :2266, docstring 'Returns: Mesh' at :2274: int index -> <class 'pymskt.mesh.meshes.Mesh'>
1-elem list -> <class 'pymskt.mesh.meshes.Mesh'> 2-elem list -> <class
'pyvista.core.pointset.PolyData'> has save_mesh: …

</details>

### 5. Multi-surface support silently computes the wrong thing above the shipped two-surface configuration

Measured: with two 2-surface decoders and four ground-truth surfaces, setting surfaces 2 and 3
to all-NaN left the loss bit-identical — decoder 1 never reads its own ground truth.
remove_overlapping_points is a silent no-op for 3+ surfaces. CLAUDE.md documents 4-surface
models as supported.

<details><summary>Folds in 4 entries</summary>

**NSM/reconstruct/main.py:605 — Single-object decoder branch indexes sdf_gt by decoder index**  
`REPRODUCES`

Hard demonstration, not inference: with two multi-output decoders the second decoder silently
scores against the first decoder's surfaces. reconstruct_mesh's own docstring advertises this
configuration ("path1_mesh = decoder0_mesh1 OR decoder1_mesh0"). Fixed = the flat sdf_gt is
indexed by a running surface offset, or the configuration is rejected.

*Evidence:* Now at :610 `sdf_gt_[decoder_idx].squeeze()` (single-output branch) and :616 `for
sdf_idx in range(pred_sdf.shape[1])` indexing the flat `sdf_gt_[sdf_idx]`. Ran
scratchpad/t_sdfidx.py: TWO 2-surface TriplanarDecoders, four ground-truth surfaces,
pts_surface labelled 0/1/2/3. Ground truth for …

**NSM/datasets/sdf_dataset.py:1928 — remove_overlapping_points hard-codes a two-surface assumption**  
`REPRODUCES`

Reproduces exactly. CLAUDE.md documents 4-surface models (bone/cart/med_men/lat_men) as a
supported configuration, and objects_per_decoder=2 already ships, so the function is silently
a no-op for the direction the library is heading. Fixed = 'inside two or more surfaces'
expressed as a count, not as a magic sum of -2.

*Evidence:* Function now at :1996, the test at :2015/:2021 (`in_in = torch.sum(total == -2)` /
`keep_mask = total != -2`). Ran it directly on hand-built sign patterns
(scratchpad/e11_overlap.py): 2 surfaces in_both/in_one/out_both : rows 3 -> 2, in_in=1 3
surfaces in3/in2/in1/out : rows 4 -> 4, in_in=0 4 …

**NSM/reconstruct/main.py:616 — In-code TODO admits the multi-surface truncation is a hack that assumes surface 0 is the bone**  
`REPRODUCES` · owned by Same class as docs/SCOPE.md §3.1's surface-ordering ruling

Reproduces exactly as the TODO admits. It is a deliberate, documented-in-code design
compromise, and it is the same positional-surface-identity problem SCOPE.md §3.1 already owns
— record it there alongside the ordering contract rather than filing it as a defect.

*Evidence:* TODO now at :621-624; the `break` at :629. Ran scratchpad/t_trunc.py — the
committed 2-surface decoder fit against a single ground-truth surface: ``` bone-only gt
against a 2-surface decoder: loss = 0.08817599713802338 verbose log: 'sdf_idx (1) >=
len(sdf_gt_) (1)... exiting' ``` No error, no …

****NSM/reconstruct/cartilage_func.py:50 — cartilage_func's mesh slicing is a hardcoded positional layout with no validation****  
`REPRODUCES`

The structure reproduces, but "silently produces wrong numbers" does not — the whole-joint
variant raises before returning anything. docs/SCOPE.md §2.5 already rules the module
Production-and-clunky. Not worth an issue on the strength of a crash that is already loud.

*Evidence:* Anchor moved to cartilage_func.py:50 (compare_cart_thickness_whole_joint, slices
[:2]/[2:4]/[4:6] at lines 54-69). Executed with the 4-surface layout CLAUDE.md gives as the
mesh_names example, spying on the inner compare_cart_thickness: whole_joint sub-call:
(['b','c'], ['b','c'], (11,12,13,14,15)) …

</details>

### 6. Library code prints to stdout ungated — route the reconstruction path through a logger

Measured 64 lines per create_mesh_adaptive call at n_pts_per_axis=128 and several hundred at
reconstruct_mesh's default; get_sdfs has no verbose parameter while every sibling does.
kneepipeline/steps/run_nsm.py:340-342 json-parses the last stdout line of the inner NSM-fit
subprocess, so this survives on ordering alone.

<details><summary>Folds in 6 entries</summary>

**NSM/mesh/main.py:836 — get_sdfs prints one unconditional line per batch in the production path**  
`REPRODUCES`

Reproduces with a measured line count, on the live path, at the caller's own defaults, from a
function whose siblings all gate on `verbose`. Fixed means get_sdfs takes `verbose` (or a
logger) and is silent by default. The audit's framing of the downstream harm is fragile-not-
broken; the noise itself is the finding.

*Evidence:* Print now at NSM/mesh/main.py:857, warning at :841-843. `get_sdfs` has no
`verbose` parameter at all (t_main.py: `has 'verbose' param? False`). Ran t_spam.py —
create_mesh_adaptive at reconstruct_mesh's own defaults (batch_size=32**3, n_pts_coarse=64),
verbose=False, n_pts_per_axis=128: stdout …

**NSM/datasets/sdf_dataset.py:685 — Unconditional debug prints on the SDF hot path**  
`REPRODUCES`

The three shape/dtype/type prints are unmistakable leftover debugging on a per-mesh, per-
sample path and are a one-line deletion; 'fixed' means they are removed or gated behind
verbose. BUT the entry's justification is stale and should not be carried over: it cites the
kneepipeline consumer parsing the JSON result from the last line of stdout, and that consumer
now reads _step_result.json from disk precisely because stdout was unreliable. File it as
library noise, not as an integration hazard.

*Evidence:* Now at :746-748. Ran read_meshes_get_sampled_pts with verbose=False and
n_pts=[10,10] and captured stdout — 16 lines for a 20-point request, including per-mesh: (20,
3) (842, 3) float64 float32 <class 'numpy.ndarray'> plus timing prints at :559, :621, :685,
:751, :808 and 'Fixed mesh...' at :182. …

**NSM/reconstruct/main.py:750 — Latent-norm progress print emits the bound method instead of the value**  
`REPRODUCES`

Reproduced by execution. One-line fix in two files. File it together with the unconditional-
print entry as a single print-hygiene sweep of this module rather than two issues.

*Evidence:* Now at :770 `print("\tLatent norm: ", latent.norm)`. Ran
scratchpad/t_norm_print.py with verbose=True, capturing stdout: ``` PRINTED: '\tLatent norm:
<bound method Tensor.norm of tensor([[-2.8552e-02, 7.2031e-03, -1.6689e-02, 2.1267e-05,
9.5957e-04,' ``` It prints the bound method AND dumps the …

**NSM/reconstruct/main.py:1153 — Unconditional debug prints on the production reconstruction path**  
`REPRODUCES`

Reproduced by execution, and the entry understates the scope — the noisiest offender is in
NSM/mesh, not reconstruct. The consumer runs the NSM fit as a subprocess whose stdout the
pipeline scans, so this is a real hygiene problem. Fixed = every print in the reconstruction
path gated on verbose or routed through the module logger. One issue covering this and the
:750 bound-method print.

*Evidence:* Now :1197, :1198, :1209, :1234, and :1452 in get_mean_errors. Ran
scratchpad/t_recon_full.py with verbose=False, capturing stdout: ``` '... length of meshes:
2\nlength of orig_mesh: 2\nfinished computing recon loss\n' ``` Ran scratchpad/t_gme.py —
get_mean_errors over 2 meshes, verbose unset: ``` …

**NSM/models/modulated_periodic_activations.py:244 — Debug print left on ImplicitDecoder's forward path**  
`REPRODUCES`

Reproduces. One-line deletion — code, not prose, but it belongs in the same Phase 2 cleanup
sweep rather than a GitHub issue. Nothing depends on the output; the passing suite merely
tolerates it.

*Evidence:* Now modulated_periodic_activations.py:245, inside `if self.mod_network is None:`
(modulation=False is the constructor default at :210). Captured with
contextlib.redirect_stdout: dec = ImplicitDecoder(latent_dim=4, out_dim=1, hidden_dim=8,
num_layers=3, block_factory=LinearBlockFactory()) …

**NSM/utils.py:9 — Importing NSM prints to stdout unconditionally when schedulefree is absent**  
`REPRODUCES`

Reproduces exactly on current main. One caveat: the entry's stated consequence is stale — it
says the consumer 'parses stdout (progress lines followed by a JSON result as the last line)',
but kneepipeline now reads `_step_result.json` and its CLAUDE.md explicitly says the result is
NOT the last line of stdout. The remaining, still-valid point is the plain one: a library must
not write to stdout at import. Three-line fix (warnings.warn, move `import warnings` above the
try).

*Evidence:* RAN: `/mnt/data/conda-envs/nsm-dev/bin/python -c "import schedulefree"` ->
ModuleNotFoundError: No module named 'schedulefree' (it is genuinely absent from nsm-dev).
RAN: `/mnt/data/conda-envs/nsm-dev/bin/python -c "import NSM" 2>/dev/null` -> stdout:
`schedulefree not found, skipping import`. Code …

</details>

### 7. Unhandled inputs surface as UnboundLocalError / NameError / invented data instead of a named error

Five sites, one fix shape (explicit else that raises, or initialise the sentinel before the
loop). refine_mesh:399 first — it fires on the module's own defaults and gates the module
SCOPE §2.3 says to keep. The NaN trigger on :794 is a plausible production failure that
reports itself as the wrong exception; the same block returns the literal sentinel 100 as its
loss under convergence='recon_loss', which is what both shipped production configs set.

> **IMPACT CHECKED — confirmed.** Both `647_nsm_femur_v0.0.1` and
> `551_nsm_femur_bone_v0.0.1` set `convergence_type_recon: "recon_loss"`, so this is the
> production reconstruction path, not a corner. `loss`/`recon_loss` are initialised to the
> literal `100` at `reconstruct/main.py:468-469`; what still needs establishing is the exact
> path on which that initial value is returned rather than overwritten.

<details><summary>Folds in 5 entries</summary>

**NSM/mesh/refine_mesh.py:399 — get_target_cells raises UnboundLocalError on its own default arguments**  
`REPRODUCES` · owned by Not on GitHub. Already ruled in docs/SCOPE.md §2.3 condition 1, indexed in docs/ARCHITECTURE.md:210 and :310, and queued in .claude/plans/NSM_CODE_HEALTH_REFACTOR.md:360 — but the repo's stated rule is that issues are the only work queue, and there is no issue for it.

Reproduces exactly as written, and it is the one entry in this slice that makes a whole module
unusable. Fixed means `np.zeros_like(max_lengths)` plus a test that calls both public entry
points with their own defaults. It gates SCOPE.md §2.3's other two conditions, which cannot be
written against code nobody can run.

*Evidence:* Bug now at NSM/mesh/refine_mesh.py:400: `max_length_binary =
np.zeros_like(max_length_binary)`. Ran t_refine.py sections A and B on a 240-cell pyvista
Sphere: {'area_threshold': 0.5} -> UnboundLocalError: local variable 'max_length_binary'
referenced before assignment {'length_threshold': 1.2} -> …

**NSM/reconstruct/main.py:445 — `optimizer` / `loss_fn` can be referenced before assignment for unrecognised names**  
`REPRODUCES`

Both reproduce by execution. The 'AdamW in the config' hazard the entry raises is one wiring
change away, not live. Fixed = an explicit else that raises ValueError naming the accepted
values. Same issue as :178.

*Evidence:* Bind sites now :444-452 (optimizer) and :455-465 (loss_fn). Ran
scratchpad/t_unbound.py against the committed regression decoder: ``` optimizer_name='AdamW'
-> UnboundLocalError: local variable 'optimizer' referenced before assignment
optimizer_name='adam' -> OK loss_type='huber' -> NameError: free …

**NSM/reconstruct/main.py:794 — reconstruct_latent can return an unbound local `latent_`**  
`REPRODUCES`

Reproduces on two independent triggers, one of which (NaN under a convergence mode production
selects) is a plausible production failure that surfaces as UnboundLocalError instead of "the
fit diverged". Fixed = latent_ initialised to the current latent before the loop, and the 100
sentinel replaced with inf/None so the returned loss is never a magic number.

*Evidence:* Now `return loss, latent_` at :824; sentinels `loss = 100` / `recon_loss = 100` at
:468-469. Ran scratchpad/t_latent_unbound.py: ``` num_iterations=0 -> UnboundLocalError: local
variable 'latent_' referenced before assignment nan sdf_gt + convergence='overall_loss',
patience=2 -> UnboundLocalError: …

**NSM/reconstruct/main.py:178 — The string branch of the sdf_gt type check is unreachable; it raises TypeError instead of its message**  
`REPRODUCES`

One missing comma; reproduced by execution. Fixed = `(str,)` and the test tightened to
`pytest.raises(Exception, match=...)`. File as one issue with the :445 UnboundLocalError entry
— both are "bad input reaches an unhelpful exception instead of the written one".

*Evidence:* `grep -n 'type(sdf_gt) in (str)'` -> :176 (still no trailing comma). Ran: ``` from
NSM.reconstruct.main import reconstruct_latent_sdf_gt_type_check as f f('some/path.vtk') ->
TypeError : argument of type 'type' is not iterable ``` The intended message at :177-180
("Must provided xyz/sdf from mesh …

**NSM/mesh/correspondence_metrics.py:676 — score_correspondence fabricates an 'original' when source_mesh is missing**  
`REPRODUCES`

Reproduces: a plausible-looking number where every other missing-input path in the same
function returns {'skipped': True, ...}. 'Fixed' = when roundtrip_points is given and
source_mesh is None, skip both metrics with a reason (or raise), and add the test case that
currently does not exist.

*Evidence:* RAN corr_check.py: `score_correspondence(warped, target, source_mesh=None,
roundtrip_points=warped.points+0.01)`. Result: `foldover_count -> {'skipped': True, 'reason':
'source_mesh not provided'}` but `roundtrip_distance -> {'min': 0.01732049, ..., 'mean':
0.01732050, 'max': 0.01732051}` and `did …

</details>

### 8. PR #38's unswept siblings: functions that mutate a caller's object and also return it

Executed: reconstruct_latent clamps and device-moves the caller's list in place through two
undocumented helpers; compute_recon_loss downcasts the caller's meshes to float32;
interpolate_common returned the caller's own 82-point mesh at 28,002 points. Same class PR #38
just closed in get_pts_center_and_scale, and CLAUDE.md says to fix the class.

<details><summary>Folds in 4 entries</summary>

**NSM/reconstruct/main.py:248 — sdf_gt is mutated in place through the type-check and preprocess helpers**  
`REPRODUCES`

Confirmed by execution: reconstruct_latent clamps and device-moves the caller's list in place,
with no docstring on either helper (reconstruct_latent has no docstring at all). Not live in
production because reconstruct_mesh builds sdf fresh each call, but it is the exact class PR
#38 just fixed in sdf_dataset.get_pts_center_and_scale — this is an unswept instance of it.

*Evidence:* Symbol now at NSM/reconstruct/main.py:171 (type check) and :236 (preprocess); the
write is at :246 `sdf_gt[sdf_idx] = sdf.to(device)`. Ran (scratchpad/t_mutate.py,
python=/mnt/data/conda-envs/nsm-dev/bin/python): ``` orig = [torch.full((5,1), 5.0),
torch.full((5,1), -5.0)] …

****NSM/reconstruct/recon_evaluation.py:95 — compute_recon_loss mutates the caller's meshes to float32 in place****  
`REPRODUCES`

Executed and reproduces on both the recon mesh and the caller's original. Fixed = cast a local
copy for the ASSD call, or document that calc_assd downcasts. Unlike the cartilage_func
sibling this one really does alias the caller's objects, because it assigns to `.point_coords`
rather than constructing a new mesh.

*Evidence:* Now at recon_evaluation.py:103-106. Ran compute_recon_loss on float64 pymskt
Meshes: ASSD before: float64 float64 ASSD after : float32 float32 {'assd_0':
0.04966860369175949} Same call with calc_assd=False leaves both at float64. These are `meshes`
and `result_['orig_mesh']`, the same objects …

**NSM/mesh/interpolate.py:519 — The is_mesh path mutates the caller's mesh in place and does not say so**  
`REPRODUCES` · owned by Same class as the get_pts_center_and_scale in-place mutation just fixed by PR #38 (issues #20/#21); that fix did not sweep this site.

Reproduces, and far more destructively than the entry conveys — a 341x point-count change to
an object the caller still holds, from a function whose `return data` reads as pure. The repo
has just fixed this exact class elsewhere (PR #38 on get_pts_center_and_scale), which is the
argument for sweeping rather than leaving it. 'Fixed' = copy the mesh (or document the
mutation and stop returning it).

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/interp_mesh_check.py: built a
`pymskt.mesh.Mesh` from an 82-point sphere and called `interpolate_mesh(model, z1, z2,
n_steps=2, mesh=m, adaptive=True, smooth=True)`. Result: `returned object IS the argument:
True`; **the caller's own mesh went from 82 to 28,002 …

**NSM/train/utils.py:87 — add_plain_lr_to_config mutates the caller's config in place while also returning it**  
`REPRODUCES`

Both halves reproduce. The behaviour is arguably fine — all four call sites use `config =
f(config)` and testing/NSM/test_lr_schedules.py:520 already deepcopies defensively — but the
docstring says nothing about mutation. Prose fix: say it mutates and returns the same object.
No issue.

*Evidence:* scratchpad/e7_lrutils.py: returned is the same object as the input: True new keys
written into caller's dict: ['latent_lr_initial','latent_lr_type','latent_lr_update_factor','l
atent_lr_update_interval','model_lr_initial','model_lr_type','model_lr_update_factor','model_l
r_update_interval'] Second …

</details>

### 9. The same knob has a different default at each layer, and adjacent metrics take their arguments in opposite order

chamfer_norm is a power, so the three layers report chamfer in different units (0.3297 vs
0.2179 on identical geometry); sigma_rand_pts differs 10x and is result-changing whenever
get_rand_pts=True. One default per knob, one docstring line, one PR.

<details><summary>Folds in 4 entries</summary>

**NSM/reconstruct/main.py:840 — `chamfer_norm` is a power, not a norm, and its default disagrees across the three layers**  
`REPRODUCES`

The default disagreement and the squared units are real and measured. The entry overstates on
documentation — the power semantics ARE documented at the two lower layers; what is missing is
any mention in reconstruct_mesh's 8-line docstring. Fixed = one default across the three
layers, plus a docstring line at reconstruct_mesh. File jointly with the sigma_rand_pts entry
as one class.

*Evidence:* Ran scratchpad/t_chamfer.py: ``` reconstruct_mesh chamfer_norm default: 2 (:871)
compute_recon_loss chamfer_norm default: 1 (recon_evaluation.py:25) compute_chamfer power
default: 1 (utils.py:83) chamfer power=1: 0.32892286960623585 chamfer power=2:
0.05514394868235056 ``` End-to-end through …

**NSM/reconstruct/main.py:834 — `sigma_rand_pts` default differs by 10x between reconstruct_mesh and get_mean_errors**  
`REPRODUCES`

The 10x default disagreement reproduces and is result-changing, but only when
get_rand_pts=True — both shipped configs set get_rand_pts_recon=false, so it is latent in
production. Fixed = one default, documented. File as one issue with the chamfer_norm entry;
drop the n_pts_per_axis sentence, which is not the same pattern.

*Evidence:* reconstruct_mesh `sigma_rand_pts=0.001` at :864; get_mean_errors
`sigma_rand_pts=0.01` at :1320, forwarded unconditionally at :1373. Ran scratchpad/t_sigma.py
(get_rand_pts=False, the shipped setting): ``` sigma 0.001 assd_0 0.21913301750054126 sigma
0.01 assd_0 0.21913301750054126 identical …

**NSM/models/loader.py:123 — conv_norm_type default differs depending on which code path builds the triplanar model**  
`REPRODUCES`

The divergence is real, undocumented, and duplicated six ways; both shipped models use 'layer'
while three of the six sites default to 'batch'. Low severity because the mismatch fails
loudly. 'Fixed' means one constant, sourced once, and a stated reason for whichever value
wins.

*Evidence:* Five sites on current main, executed via loader:
_get_triplanar_params({'latent_size':16})['conv_norm_type'] -> 'batch' (loader.py:124)
_get_two_stage_params({'latent_size':16})['triplanar_params']['conv_norm_type'] -> 'layer'
(loader.py:190) get_model_config_template('triplanar')['conv_norm_type'] …

**NSM/mesh/correspondence_metrics.py:537 — Adjacent metrics take the same two arrays in opposite positional order**  
`REPRODUCES` · owned by docs/ARCHITECTURE.md:304 already tracks this class with a count of ~12 instances, but names other sites, not this one.

Reproduces exactly as described, and both failure modes are silent. Mitigation: the module has
no non-test importer, and the only in-repo caller (score_correspondence) passes them
correctly. Fix is free and lossless — make the two signatures agree, or make both keyword-
only. 'Fixed' = a caller cannot swap them without a TypeError.

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/corr_check.py: `roundtrip_distance(A,B)` and
`roundtrip_distance(B,A)` return identical per_vertex arrays and identical mean
0.11065713875408247 — a swap is completely invisible. `forward_backward_disagreement(B,A)` vs
`(A,B)`: `field` is exactly sign-flipped …

</details>

### 10. Face arrays are reshaped without validation, so a quad or a VTK-style array silently builds garbage

Five to six sites, one shared validation helper. Supersedes existing issue #6, which states
the same class as a one-line note — close #6 with a pointer.

<details><summary>Folds in 2 entries</summary>

**NSM/mesh/correspondence_metrics.py:291 — faces.reshape(-1, 4) assumes an all-triangle mesh in three places**  
`REPRODUCES`

Reproduces, including the dangerous mixed-mesh case, which I had to construct deliberately
(the entry's own numbers did not include a worked example). Same class as the interpolate.py
`reshape(-1, 3)` entry, so one issue should cover all five sites. 'Fixed' = the face accessor
validates (or uses `regular_faces` and raises on non-triangular input) in one shared helper.

*Evidence:* RAN corr_check.py + corr_check2.py. Pure-quad PolyData (2 quads, flat length 10):
`self_intersection_count` and `foldover_count` both raise `ValueError: cannot reshape array of
size 10 into shape (4)`. MIXED mesh of 4 quads + 3 triangles (flat length 4*3+5*4 = 32, 32 % 4
== 0, mesh.n_cells == 7): …

**NSM/mesh/interpolate.py:307 — faces argument silently accepts pyvista's VTK-style face array and builds garbage**  
`REPRODUCES`

Reproduces, with numbers the entry did not have. This is the highest-consequence instance of
the face-array class: the wrong input silently changes the smoothing operator and the pinned-
vertex set, so the interpolation output is wrong rather than absent. One issue with the
correspondence_metrics/refine_mesh sites.

*Evidence:* RAN interp_check.py. `pv.Sphere(theta_resolution=8,
phi_resolution=8).triangulate()`: 96 triangles, flat `.faces` length 384, 384 % 3 == 0.
`build_mesh_laplacian(mesh.faces, ...)` **did not raise**: nnz 373 (flat) vs 288
(regular_faces), and `torch.equal(dense_bad, dense_good)` is False. …

</details>

### 11. Model configurations that construct successfully and then crash on the first forward

docs/ARCHITECTURE.md:310 names the class with five instances and no issue owns it. The close
condition is a parameterised constructor-and-one-forward smoke test over the documented option
values, which also stops the class reopening.

<details><summary>Folds in 4 entries</summary>

**NSM/models/deep_sdf.py:171 — progressive_add_depth path propagates None through the layer stack**  
`REPRODUCES`

Reproduces exactly, and the failure window covers every realistic training run.
ARCHITECTURE.md §7 names the class ('constructible-but-uncallable configuration', 5 instances)
but there is no GitHub issue for it. File one class issue covering this, deep_sdf:180,
deep_sdf:308 and two_stage:24; 'fixed' means each of those constructor options either works or
refuses at construction time.

*Evidence:* Code now at NSM/models/deep_sdf.py:172 (bare `return` in forward_branch_) and
:218-219 (`if x is None: continue`). Ran /tmp/.../scratchpad/v3.py: for ep in
(1,199,200,1010,1209,1500): Decoder(latent_size=8, dims=[16]*8,
progressive_add_depth=True)(torch.randn(4,11), epoch=ep) Output: epoch=1: …

**NSM/models/deep_sdf.py:180 — Decoder indexes self.bn by absolute layer index but appends only for norm layers**  
`REPRODUCES`

Reproduces. Same class as the two entries above; group them into one issue rather than filing
three. Note norm_layers is marked DEPRECATED at deep_sdf.py:44 and loader.py:324, so the fix
may be deletion rather than repair.

*Evidence:* Code now at NSM/models/deep_sdf.py:138 (`self.bn.append(...)` only inside the
`elif ... layer in self.norm_layers` branch) and :181 (`x = self.bn[layer_idx](x)`). Ran: d =
Decoder(latent_size=8, dims=[16,16,16], weight_norm=False, norm_layers=(2,)) print(len(d.bn),
len(d.layers)) -> 1 4 …

**NSM/models/deep_sdf.py:308 — activation='linear' builds a Decoder that crashes on first forward**  
`REPRODUCES`

The crash reproduces; the 'advertised' framing is mildly overstated. Still worth fixing as
part of the class issue — get_activation returning a bare None for a value the caller cannot
distinguish from a mistake is the defect.

*Evidence:* get_activation now at NSM/models/deep_sdf.py:288-310; `elif activation ==
"linear": return None` at :307-308. Ran: Decoder(latent_size=8, dims=[16,16],
activation='linear')(torch.randn(4,11)) -> TypeError: 'NoneType' object is not callable (from
:182, `x = self.activation(x)`) Control, showing the …

**NSM/models/two_stage.py:24 — TwoStageDecoder cannot be constructed with its own defaults**  
`REPRODUCES`

Reproduces — the class is unconstructible by any direct caller, at any argument. Same class as
the three deep_sdf entries; one issue, four instances. ARCHITECTURE.md §7 already names it in
the class table but no issue owns it.

*Evidence:* default_mlp_params['dims'] is still a tuple at two_stage.py:24; deep_sdf.py:84
does `self.dims = [latent_size + 3] + dims`. In a fresh process: TwoStageDecoder() ->
TypeError: can only concatenate list (not "tuple") to list TwoStageDecoder(latent_size=8,
n_objects=5) -> TypeError: can only …

</details>

### 12. compute_recon_loss(calc_emd=True) is dead on arrival and default_config.json ships emd: True

pykeops rejects numpy for every dtype, so the failure is at the boundary and no input reaches
the point-count check the register blamed. Verified with a 4-cell torch/numpy x f32/f64
matrix. train_deep_sdf.py:233 reads the key unconditionally, so the shipped default cannot be
used.

<details><summary>Folds in 3 entries</summary>

****NSM/dependencies/sinkhorn.py:107 — sinkhorn's default uniform weights cannot sum equal when the two point clouds differ in size, so it always raises****  
`REPRODUCES`

The sinkhorn defect is real and executed. But the register attributes the EMD failure to
unequal sizes when in fact `calc_emd=True` NEVER works from compute_recon_loss for any input,
because the only caller passes numpy. That is the bigger, unrecorded finding and it should be
the issue: fixed = compute_recon_loss(calc_emd=True) returns a number for meshes of unequal
point count, which needs both the numpy->torch conversion and the sinkhorn weight
normalisation.

*Evidence:* Ran: `sinkhorn(torch.rand(5,3), torch.rand(7,3))` -> `ValueError: Weights w_x and
w_y do not sum to the same value, got w_x.sum() = 1.0 and w_y.sum() = 0.7142857313156128`.
Code at sinkhorn.py:106-118 unchanged. BUT the stated caller impact is wrong.
recon_evaluation.py:122 calls …

****NSM/dependencies/sinkhorn.py:49 — Unreachable duplicated type check in sinkhorn; max_iters is never type-checked****  
`REPRODUCES`

Executed and reproduces. Fixed = line 48 tests max_iters, not p; the message it already
carries then becomes true. Group with the other three sinkhorn validation findings — one
issue, four sites. Note the file is a vendored copy of fwilliams/scalable-pytorch-sinkhorn
(cited in requirements.txt), so the fix is a local-fork decision.

*Evidence:* Lines unchanged: sinkhorn.py:40 `if not isinstance(p, int)` and :48-49 `if not
isinstance(p, int): raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")`.
Ran: `sinkhorn(torch.rand(5,3), torch.rand(5,3), max_iters="banana")` -> `TypeError: '<=' not
supported between instances of …

****NSM/dependencies/sinkhorn.py:92 — sinkhorn validates w_x's length twice and never validates w_y's****  
`REPRODUCES`

Executed and reproduces. Fixed = line 91 reads `w_y.shape[0] != y.shape[0]`. Same one-issue
group as the other sinkhorn validation defects.

*Evidence:* sinkhorn.py:76 and :91 both read `if w_x.shape[0] != x.shape[0]`; the message at
:93-96 talks about w_y and y. Ran: `sinkhorn(x(5,3), y(5,3), w_x=ones(5)/5, w_y=ones(9)/9)` ->
passed validation and died inside keops with `ValueError: Incompatible values for attribute
nj: 9 and 5.` — exactly the …

</details>

### 13. train_deep_sdf_multi_head repair checklist

SCOPE.md §2.1 already rules the module 'supported, broken, fix it' and KNOWN_ISSUES §2 owns
the optimizer bug (:85). What is missing is the checklist the repair works from: latents never
moved to the device, non-short-circuit & raising KeyError, torch.mps.empty_cache on the CPU
branch (unreachable today because the CPU path dies earlier at :256), per-surface L1 appended
to a fixed-size list then discarded, and a hardcoded 100-epoch warm-up ignoring its config
key. Five entries, one repair.

<details><summary>Folds in 1 entries</summary>

**NSM/train/train_deep_sdf_multi_head.py:83 — multi_head never moves latent vectors to the device and hardcodes .cuda()**  
`REPRODUCES`

Confirmed three ways: the module only runs at all because the latents are LEFT on CPU, it
cannot run on cpu/mps despite advertising config["device"], and the naive fix breaks it.
Should be one issue with the other multi_head divergences (:118, :123, :359, :382), since
SCOPE.md 2.1 already rules the module must be repaired.

*Evidence:* scratchpad/e9_multihead.py — ran multi_head on the T4 with device="cuda", spying
on get_optimizer: latent_device: 'cpu' (config["device"] was "cuda") scratchpad/e11_mh3.py —
same module with device="cpu": RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, …

</details>

### 14. FastUnique scales latent gradients by the number of query points, diluting latent_reg_weight

Measured exactly 10.00x at N=10 and 1000.00x at N=1000 on both paths. The amplification is
asymmetric inside reconstruct_latent: the reconstruction term reaches the latent through
FastUnique and is scaled by N, while latent_loss reaches the same leaf directly and is not, so
the configured latent_reg_weight is silently divided by the number of query points.
kneepipeline/steps/run_nsm.py:190-191 enables it in production. Not a regression, so the fix
needs a KNOWN_ISSUES.md § History entry.

> **IMPACT CHECKED — the production claim is wrong.** `run_nsm.py:190-191` passes
> `l2reg=model_config["l2reg_recon"]`, and both shipped configs set it to `false`. The code
> is `if l2reg is True: latent_loss = latent_reg_weight * ... else: latent_loss = 0`, so the
> term this dilutes is **zero in production**. The 1000x measurement stands; "enabled in
> production" does not. File it as a defect for anyone who turns `l2reg` on, not as a live
> production bug, and it needs no `History` entry because no shipped run was affected.

<details><summary>Folds in 1 entries</summary>

**NSM/models/triplanar.py:158 — Latent gradients are scaled by the number of query points**  
`REPRODUCES`

Reproduces exactly, and I sharpened it past the entry: the amplification is asymmetric between
the reconstruction term and the L2/norm-penalty terms, so the configured latent regularization
is silently diluted by the number of query points on the production reconstruction path. Not a
regression (the legacy path always did this), so any fix needs a KNOWN_ISSUES.md §History
entry per CLAUDE.md § Numerical-behaviour changes. 'Fixed' means the recon and regularization
gradients on the latent are on the same scale, with the chosen convention stated.

*Evidence:* FastUnique is now triplanar.py:153-171; `expanded_grad =
grad_output.repeat(ctx.num_points, 1)` at :170 returns (N,D) for a (D,) input and relies on
autograd's sum-to-size. Ran v5.py — fast path, legacy path, and a manual decode-once
reference, N=10: fast/ref ratio : [10.0, 10.0, 10.0, 10.0, 10.0] …

</details>

### 15. Sweep: declared config keys and option values that raise, no-op, or silently disable the feature

The #20 pattern applied to configuration: the first task is the enumeration — for every key
the code reads and every value a docstring advertises, set it and record what happens. Eleven
entries, and the shipped default_config.json cannot drive train_deep_sdf at all.

<details><summary>Folds in 7 entries</summary>

**NSM/train/train_deep_sdf.py:108 — The shipped default_config.json cannot drive train_deep_sdf — six unconditionally-read keys are missing**  
`REPRODUCES`

Five of the six named keys are genuinely fatal on the shipped default; `assd` is not reachable
from it, so the entry is right in substance and slightly overstated in count. Real, fixable,
untracked: "fixed" = default_config.json defines the keys the trainer reads unconditionally,
pinned by a test that instantiates the trainer from it.

*Evidence:* Ran the real trainer on the shipped default config (tiny triplanar architecture
substituted so a model exists; CPU; synthetic meshes from testing/NSM/regression/_harness.py),
adding one missing key at a time. Script: scratchpad/e1_defaultcfg.py + e1b.py.
KeyError('prefetch_factor') at …

****NSM/configs/default_config.json:1 — No shipped config can construct a triplanar model faithfully****  
`REPRODUCES`

Factually correct and executed, but it is already ruled on in docs/SCOPE.md §1 as a Phase 4
work item ("ship a default config per model type, derived from the ShapeMedKnee configs").
Keeping it here duplicates a doc that already owns it.

*Evidence:* Ran: python -c "import json; c=json.load(open('NSM/configs/default_config.json'));
..." Output: n keys: 61 / missing triplanar keys: …

****NSM/configs/default_config.json:1 — Nothing in the library ever loads default_config.json****  
`REPRODUCES`

Reproduces, and it is an observation rather than a defect — docs/SCOPE.md §2.6 already rules
generate_sdf_default_config.py "supported: it owns the shipped default_config.json and is
pinned by test_default_config_sync.py", which says the same thing.

*Evidence:* `grep -rn 'default_config' --include=*.py NSM/ testing/ examples/`: inside NSM/
the only hits are generate_sdf_default_config.py's own DEFAULT_CONFIG_PATH constant (:95) and
write path (:99-107), plus a prose comment in train_deep_sdf.py:407. Readers are
testing/NSM/test_lr_schedules.py:547 and …

**NSM/train/train_deep_sdf.py:422 — multi_object_overlap is a config key whose only implementation is an unconditional raise**  
`REPRODUCES`

Reproduces, but it is a status question, not a defect: the key names an unimplemented feature.
It belongs in SCOPE.md alongside the eikonal ruling — 'accepted by config, not implemented,
crashes mid-epoch if enabled' — rather than as an issue.

*Evidence:* scratchpad/e13_misc.py — set config["multi_object_overlap"]=True on an otherwise
working run: Exception: Not implemented yet | raised at line 435 (inside train_epoch, after
data loading and the first forward pass) Same construct at multi_head:259-266.

**NSM/train/utils.py:90 — add_plain_lr_to_config raises KeyError on a Constant learning-rate schedule**  
`REPRODUCES`

Reproduces, and the asymmetry (Constant supported by get_learning_rate_schedules, fatal in the
logging helper) is the whole point. "Fixed" = read Initial defensively (fall back to Value, or
skip absent keys) and add a Raises note; pinned by a test that trains one epoch on a Constant
schedule.

*Evidence:* scratchpad/e7_lrutils.py, with {"Type":"Constant","Value":...} entries carrying
proper Targets: get_learning_rate_schedules OK? yes -> {'model': 0.005, 'latent': 0.001}
add_plain_lr_to_config -> KeyError('Initial') at NSM/train/utils.py:87 mixed (one Constant,
one Step) -> KeyError('Initial') as …

**NSM/reconstruct/main.py:297 — latent_norm_penalty returns a Python float 0.0 inside the range, breaking .item() on the logged value**  
`NOT_A_DEFECT`

The entry as titled is disproved: the float return is harmless and the hasattr guards handle
it. Blunt version — this is the alarm-direction overstatement the banner warns about. But a
real defect sits in the same function: norm_penalty_type='barrier' returns NaN for any latent
outside [min,max], which is the state at initialisation, so the option is broken for its whole
intended use. Fixed = clamp the log arguments (or reject the option). File the barrier bug;
drop the float claim.

*Evidence:* Floats at :295 and :306; return at :330. Ran scratchpad (t_latent_norm): ``` in-
range penalty: 0.0 <class 'float'> has item: False out-of-range: tensor(1.,
grad_fn=<MulBackward0>) ``` The headline claim is FALSE: every consumer is guarded (:759,
:766, :783, :788 all `hasattr(x, 'item')`), and a run …

**NSM/reconstruct/main.py:1372 — Regress.add_latent is handed the whole result dict, not a latent vector**  
`REPRODUCES` · owned by Noted in docs/SCOPE.md §2.5 as "a live defect ... worth pinning down", but not on the GitHub tracker

Reproduced by execution. The latent-to-factor regression validator raises for every user who
enables it, which means it has been dead since whenever the reconstruct_mesh return shape
changed. Fixed = pass `result_['latent'].detach().cpu().numpy().ravel()`, plus a regression
test. SCOPE.md records the observation but nothing owns the fix.

*Evidence:* Now at :1416 `reg.add_latent(result_)`; `result_` is reconstruct_mesh's full
result dict (:1410). Ran scratchpad/t_regress.py — Regress subclassed only to stub
get_all_factors (which reads filenames), three dict latents added: ``` reg.calc_r2() ->
TypeError : float() argument must be a string or a …

</details>

### 16. save_model_params silently refuses to overwrite and silently drops non-JSON config values

Two slices found the same bug from opposite ends. model_params_config.json is the only record
of what a run was configured with; it is written from the first checkpoint and never
corrected, so a run whose config changes or carries a non-JSON value leaves a persisted file
that disagrees with it.

<details><summary>Folds in 2 entries</summary>

**NSM/train/train_deep_sdf.py:198 — save_model_params is called on every checkpoint but silently no-ops after the first**  
`REPRODUCES`

Reproduces, and the resume case is real: a resumed run's model_params_config.json keeps the
first run's hyperparameters and mesh list, and that file is what downstream consumers read.
Should be filed once, jointly with the NSM/utils.py:312 entry from the utils slice, not twice.

*Evidence:* scratchpad/e4_smp.py — two calls with different configs into one
experiment_directory: first : {'resume_epoch': 0, 'n_epochs': 10, 'lr': 0.001,
'list_mesh_paths': ['a.vtk']} second: {'resume_epoch': 0, 'n_epochs': 10, 'lr': 0.001,
'list_mesh_paths': ['a.vtk']} file changed on 2nd call: False The …

**NSM/utils.py:312 — save_model_params silently refuses to overwrite and silently drops non-JSON config values**  
`REPRODUCES`

Reproduces exactly and it is user-visible: model_params_config.json is what every downstream
consumer reads to rebuild a model, and on a resumed or re-run experiment it silently describes
different hyperparameters than the weights beside it. 'Fixed' = overwrite (or epoch-stamp) and
log the keys filter_non_jsonable removes.

*Evidence:* RAN utils_check.py: first call with `{lr:0.001, resume_epoch:0, bad:<object>}`
writes the file, and `'bad' key present? False` (filter_non_jsonable dropped it with no log).
Second call with `{lr:0.9999, resume_epoch:500}` and `list_mesh_paths=['b.vtk']` -> on disk
still `lr= 0.001 resume_epoch= 0 …

</details>

### 17. The LR fix's leftovers: positional indexing in the logging helper, and latent-norm stats assigned rather than accumulated

CLAUDE.md's LR section exists because positional coupling here swapped two schedules for three
years, and its rule is to enumerate every place the shape occurs. These are the survivors,
plus the latent-norm metric that is wrong by a factor of n_batches and a test that currently
pins mislabelled output.

<details><summary>Folds in 4 entries</summary>

**NSM/train/train_deep_sdf.py:569 — step_mean_vec_length / step_std_vec_length are assigned, not accumulated — logged latent-norm metrics are wrong by a factor of len(data_loader)**  
`REPRODUCES`

Confirmed numerically: the reported value is (last chunk of last batch) / n_batches, so it
shrinks as the dataset grows. "Fixed" = `+=` at 582-583, matching the surrounding
accumulators. Only affects wandb logging, not gradients — worth an issue, not a KNOWN_ISSUES
History entry.

*Evidence:* scratchpad/e5_run.py — real 2-epoch CPU run, reading the latent embedding directly
after the epoch: n_batches (len(data_loader)): 2 true mean latent norm over whole embedding:
0.01071708 logged mean_vec_length: 0.00526427 logged std_vec_length : 0.0 logged * n_batches :
0.01052855 <- the LAST …

**NSM/train/utils.py:76 — The positional indexing the LR fix removed still survives in the logging helper**  
`NOT_A_DEFECT`

The entry itself concedes "It is correct today". targets.index() is an internal implementation
detail of a lookup that is keyed by Target, not a surviving positional contract. The only
substantive half — the idx_model/idx_latent override — is the :63 entry. Duplicate.

*Evidence:* scratchpad/e7_lrutils.py — both entry orders through the target-keyed path:
('model','latent') -> model_lr_initial 0.005 latent_lr_initial 0.001 ('latent','model') ->
model_lr_initial 0.005 latent_lr_initial 0.001 Correct in both. Pinned by …

**NSM/train/utils.py:63 — add_plain_lr_to_config retains a positional back door that a test pins to mislabelled output**  
`REPRODUCES` · owned by #20 is the adjacent class sweep ("parameters accepted and never read"); these are read, but by no caller

The parameters exist solely so a test can assert deliberately swapped labels, in the very
function whose Aug-2026 fix was about eliminating positional mapping. "Fixed" = delete
idx_model/idx_latent and delete test_explicit_indices_still_override. Absorbs the :76 entry.

*Evidence:* scratchpad/e7_lrutils.py: add_plain_lr_to_config(cfg, idx_model=1, idx_latent=0)
-> model_lr_initial = 0.001 (the LATENT entry's Initial) -> latent_lr_initial = 0.005 Only
caller in the entire repo (grep -rn 'idx_model|idx_latent' NSM/ testing/, excluding the
definition): …

**NSM/train/train_deep_sdf.py:152 — The param-group target key is duplicated as a bare string literal in the train loop**  
`NOT_A_DEFECT`

Not a defect — a one-line style coupling with no observable failure. Filed under "Defects" it
is overstated. Fold into whatever edit next touches the resume block; nothing to track.

*Evidence:* scratchpad/e13_misc.py: constant imported into train_deep_sdf? False literal used:
['if any(group.get("target") is None for group in optimizer.param_groups):'] (line 152)
NSM.utils.PARAM_GROUP_TARGET_KEY = 'target' Behaviour is correct today: the literal and the
constant are the same string, and …

</details>


### Unassigned — 45 duplicates + 16 genuine (heading previously said 62)

> Corrected 2026-08-21, measured by comparing entry titles mechanically against the
> grouped entries above: 45 of the entries below are verbatim duplicates of entries
> already folded into groups 1–17 and resolve with them; 16 are genuine and are
> dispositioned in § 0.5. Anyone sizing the work from the old heading was off 3.6×.

<details><summary>Show</summary>

**NSM/datasets/sdf_dataset.py:301 — uniform_pts_buffer expands the max side more than the min side**  
`REPRODUCES`

Straightforward arithmetic bug in both copies, dormant only because the default is 0.0 — and
commit 48c5f60 added the parameter precisely so it would be nonzero. "Fixed" means capturing
the span once before rebinding mins, so the box grows symmetrically. Not covered by any open
issue (#19 mentions uniform_pts_buffer only as a missing cache-key input).

*Evidence:* Now at :357-358 (single) and :704-705 (multi), identical copy-paste. I exec'd the
two lines pulled straight out of the module source rather than retyping them: mins = mins -
uniform_pts_buffer / 2 * (maxs - mins) maxs = maxs + uniform_pts_buffer / 2 * (maxs - …

**NSM/datasets/sdf_dataset.py:308 — Uniform-sample clipping exists only in the single-mesh function**  
`REPRODUCES`

Two nominally parallel samplers produce different sampling domains for the same request. Same
root area as :301 so it should be one issue with it. "Fixed" means the two functions agree on
the domain (either both clip or neither does, decided deliberately).

*Evidence:* grep 'np.clip' NSM/datasets/sdf_dataset.py -> one hit, :366, inside
read_mesh_get_sampled_pts. read_meshes_get_sampled_pts has no equivalent after :704-705. Ran
both with uniform_pts_buffer=0.5, norm_pts=True, sigma=None: SINGLE: min -1.2500 max 1.2500
(== …

**NSM/datasets/sdf_dataset.py:685 — Unconditional debug prints on the SDF hot path**  
`REPRODUCES`

The three shape/dtype/type prints are unmistakable leftover debugging on a per-mesh, per-
sample path and are a one-line deletion; 'fixed' means they are removed or gated behind
verbose. BUT the entry's justification is stale and should not be carried over: it cites the
kneepipeline consumer parsing the JSON result from the last line of stdout, and that consumer
now reads _step_result.json from disk precisely because stdout was unreliable. File it as
library noise, not as an integration hazard.

*Evidence:* Now at :746-748. Ran read_meshes_get_sampled_pts with verbose=False and
n_pts=[10,10] and captured stdout — 16 lines for a 20-point request, including per-mesh: (20,
3) (842, 3) float64 float32 <class 'numpy.ndarray'> plus timing prints at :559, :621, :685, …

**NSM/datasets/sdf_dataset.py:1305 — sdf_pos_neg_idx divides by zero when a surface has no positive or no negative samples**  
`REPRODUCES`

Reproduces at both call sites and end-to-end via two realistic triggers (a None surface, and
one surface nested inside another). Not tracked by any open issue, and the codebase's own test
fixtures were shaped around it — that is evidence plus a fixable statement: guard the repeat
so an empty pos/neg set raises something that names the surface, or is handled.

*Evidence:* Code now at NSM/datasets/sdf_dataset.py:1385-1386 (SDFSamples) and :2124-2125
(MultiSurfaceSDFSamples). Unit probe (scratchpad/e1_zerodiv.py, e1b.py): single:
ZeroDivisionError integer division or modulo by zero multi: ZeroDivisionError integer division
or …

**NSM/datasets/sdf_dataset.py:1487 — Joint (scale_jointly) normalization is skipped entirely when subsample is None**  
`REPRODUCES`

Reproduces, and the returned item differs from the subsampled case in shape, key set and
coordinate space — the trainer reads timing keys unconditionally, so this cannot train either.
Fixable statement: reject subsample=None at construction, or make it produce the same contract
as a subsampled item. Groups with the :1598 entry.

*Evidence:* Code now at :1573 (`if self.subsample is not None:`) with the joint block at
:1635, and :2176 / :2240 for the subclass. Cold construction with subsample=None raises before
it matters (scratchpad/e8_real.py): TypeError: unsupported operand type(s) for /: …

**NSM/datasets/sdf_dataset.py:1595 — joint_scale_buffer cannot be set on MultiSurfaceSDFSamples**  
`REPRODUCES`

Reproduces exactly as written. Multi-surface is the production configuration (bone+cartilage),
so the 10% joint-scaling buffer that sets the normalization radius of every shipped multi-
surface dataset is unreachable from the public constructor. Fixed = the parameter is accepted
and forwarded (and lands in the cache key, which is #19's half).

*Evidence:* scratchpad/e9_jsb.py, against real meshes: SDFSamples has joint_scale_buffer: True
Multi has joint_scale_buffer: False passed joint_scale_buffer -> TypeError __init__() got an
unexpected keyword argument 'joint_scale_buffer' multi dataset joint_scale_buffer …

**NSM/datasets/sdf_dataset.py:1598 — MultiSurfaceSDFSamples default subsample=None is unusable**  
`REPRODUCES`

The documented default of a public constructor cannot construct. Reproduces both in isolation
and end-to-end, and is not tracked. Fixed = subsample is required (or validated) rather than
defaulting to a value nothing supports. Same class as the :1487 entry; file them together.

*Evidence:* Default still `subsample=None` at :1684. End-to-end with the regression harness
and real meshes (scratchpad/e10_real.py): CONSTRUCTION ERROR: TypeError unsupported operand
type(s) for *: 'float' and 'NoneType' Path: __init__ -> run_before_loading_data (:1786) …

**NSM/datasets/sdf_dataset.py:1928 — remove_overlapping_points hard-codes a two-surface assumption**  
`REPRODUCES`

Reproduces exactly. CLAUDE.md documents 4-surface models (bone/cart/med_men/lat_men) as a
supported configuration, and objects_per_decoder=2 already ships, so the function is silently
a no-op for the direction the library is heading. Fixed = 'inside two or more surfaces'
expressed as a count, not as a magic sum of -2.

*Evidence:* Function now at :1996, the test at :2015/:2021 (`in_in = torch.sum(total == -2)` /
`keep_mask = total != -2`). Ran it directly on hand-built sign patterns
(scratchpad/e11_overlap.py): 2 surfaces in_both/in_one/out_both : rows 3 -> 2, in_in=1 3
surfaces …

**NSM/datasets/sdf_dataset.py:2193 — combine_meshes returns a pyvista PolyData, not a pymskt Mesh, whenever it actually combines (verified)**  
`REPRODUCES`

The type inconsistency is real and the docstring contradicts it, but the entry's failure story
is wrong: the multi-surface reference-mesh path dies one statement earlier with
UnboundLocalError. Worth one issue covering the whole path — fixed = reference_mesh=int with a
list-valued mesh_to_scale returns a usable combined pymskt Mesh, which requires combine_meshes
to keep its declared return type. (docs/MULTI_SURFACE_REGISTRATION.md:75 advertises this as
working; that doc entry is in another slice — fold them.)

*Evidence:* Type claim, run against pymskt (scratchpad/e13_combine.py); combine_meshes is now
at :2266, docstring 'Returns: Mesh' at :2274: int index -> <class 'pymskt.mesh.meshes.Mesh'>
1-elem list -> <class 'pymskt.mesh.meshes.Mesh'> 2-elem list -> <class …

**NSM/reconstruct/main.py:248 — sdf_gt is mutated in place through the type-check and preprocess helpers**  
`REPRODUCES`

Confirmed by execution: reconstruct_latent clamps and device-moves the caller's list in place,
with no docstring on either helper (reconstruct_latent has no docstring at all). Not live in
production because reconstruct_mesh builds sdf fresh each call, but it is the exact class PR
#38 just fixed in sdf_dataset.get_pts_center_and_scale — this is an unswept instance of it.

*Evidence:* Symbol now at NSM/reconstruct/main.py:171 (type check) and :236 (preprocess); the
write is at :246 `sdf_gt[sdf_idx] = sdf.to(device)`. Ran (scratchpad/t_mutate.py,
python=/mnt/data/conda-envs/nsm-dev/bin/python): ``` orig = [torch.full((5,1), 5.0), …

**NSM/reconstruct/main.py:605 — Single-object decoder branch indexes sdf_gt by decoder index**  
`REPRODUCES`

Hard demonstration, not inference: with two multi-output decoders the second decoder silently
scores against the first decoder's surfaces. reconstruct_mesh's own docstring advertises this
configuration ("path1_mesh = decoder0_mesh1 OR decoder1_mesh0"). Fixed = the flat sdf_gt is
indexed by a running surface offset, or the configuration is rejected.

*Evidence:* Now at :610 `sdf_gt_[decoder_idx].squeeze()` (single-output branch) and :616 `for
sdf_idx in range(pred_sdf.shape[1])` indexing the flat `sdf_gt_[sdf_idx]`. Ran
scratchpad/t_sdfidx.py: TWO 2-surface TriplanarDecoders, four ground-truth surfaces,
pts_surface …

**NSM/reconstruct/main.py:794 — reconstruct_latent can return an unbound local `latent_`**  
`REPRODUCES`

Reproduces on two independent triggers, one of which (NaN under a convergence mode production
selects) is a plausible production failure that surfaces as UnboundLocalError instead of "the
fit diverged". Fixed = latent_ initialised to the current latent before the loop, and the 100
sentinel replaced with inf/None so the returned loss is never a magic number.

*Evidence:* Now `return loss, latent_` at :824; sentinels `loss = 100` / `recon_loss = 100` at
:468-469. Ran scratchpad/t_latent_unbound.py: ``` num_iterations=0 -> UnboundLocalError: local
variable 'latent_' referenced before assignment nan sdf_gt + …

**NSM/reconstruct/main.py:840 — `chamfer_norm` is a power, not a norm, and its default disagrees across the three layers**  
`REPRODUCES`

The default disagreement and the squared units are real and measured. The entry overstates on
documentation — the power semantics ARE documented at the two lower layers; what is missing is
any mention in reconstruct_mesh's 8-line docstring. Fixed = one default across the three
layers, plus a docstring line at reconstruct_mesh. File jointly with the sigma_rand_pts entry
as one class.

*Evidence:* Ran scratchpad/t_chamfer.py: ``` reconstruct_mesh chamfer_norm default: 2 (:871)
compute_recon_loss chamfer_norm default: 1 (recon_evaluation.py:25) compute_chamfer power
default: 1 (utils.py:83) chamfer power=1: 0.32892286960623585 chamfer power=2: …

**NSM/reconstruct/main.py:178 — The string branch of the sdf_gt type check is unreachable; it raises TypeError instead of its message**  
`REPRODUCES`

One missing comma; reproduced by execution. Fixed = `(str,)` and the test tightened to
`pytest.raises(Exception, match=...)`. File as one issue with the :445 UnboundLocalError entry
— both are "bad input reaches an unhelpful exception instead of the written one".

*Evidence:* `grep -n 'type(sdf_gt) in (str)'` -> :176 (still no trailing comma). Ran: ``` from
NSM.reconstruct.main import reconstruct_latent_sdf_gt_type_check as f f('some/path.vtk') ->
TypeError : argument of type 'type' is not iterable ``` The intended message at …

**NSM/reconstruct/main.py:297 — latent_norm_penalty returns a Python float 0.0 inside the range, breaking .item() on the logged value**  
`NOT_A_DEFECT` · folded into the declared-config-options sweep, barrier half only

The entry as titled is disproved: the float return is harmless and the hasattr guards handle
it. Blunt version — this is the alarm-direction overstatement the banner warns about. But a
real defect sits in the same function: norm_penalty_type='barrier' returns NaN for any latent
outside [min,max], which is the state at initialisation, so the option is broken for its whole
intended use. Fixed = clamp the log arguments (or reject the option). File the barrier bug;
drop the float claim.

*Evidence:* Floats at :295 and :306; return at :330. Ran scratchpad (t_latent_norm): ``` in-
range penalty: 0.0 <class 'float'> has item: False out-of-range: tensor(1.,
grad_fn=<MulBackward0>) ``` The headline claim is FALSE: every consumer is guarded (:759,
:766, :783, …

**NSM/reconstruct/main.py:445 — `optimizer` / `loss_fn` can be referenced before assignment for unrecognised names**  
`REPRODUCES`

Both reproduce by execution. The 'AdamW in the config' hazard the entry raises is one wiring
change away, not live. Fixed = an explicit else that raises ValueError naming the accepted
values. Same issue as :178.

*Evidence:* Bind sites now :444-452 (optimizer) and :455-465 (loss_fn). Ran
scratchpad/t_unbound.py against the committed regression decoder: ``` optimizer_name='AdamW'
-> UnboundLocalError: local variable 'optimizer' referenced before assignment
optimizer_name='adam' -> …

**NSM/reconstruct/main.py:1372 — Regress.add_latent is handed the whole result dict, not a latent vector**  
`REPRODUCES` · owned by Noted in docs/SCOPE.md §2.5 as "a live defect ... worth pinning down", but not on the GitHub tracker

Reproduced by execution. The latent-to-factor regression validator raises for every user who
enables it, which means it has been dead since whenever the reconstruct_mesh return shape
changed. Fixed = pass `result_['latent'].detach().cpu().numpy().ravel()`, plus a regression
test. SCOPE.md records the observation but nothing owns the fix.

*Evidence:* Now at :1416 `reg.add_latent(result_)`; `result_` is reconstruct_mesh's full
result dict (:1410). Ran scratchpad/t_regress.py — Regress subclassed only to stub
get_all_factors (which reads filenames), three dict latents added: ``` reg.calc_r2() ->
TypeError : …

**NSM/reconstruct/main.py:750 — Latent-norm progress print emits the bound method instead of the value**  
`REPRODUCES`

Reproduced by execution. One-line fix in two files. File it together with the unconditional-
print entry as a single print-hygiene sweep of this module rather than two issues.

*Evidence:* Now at :770 `print("\tLatent norm: ", latent.norm)`. Ran
scratchpad/t_norm_print.py with verbose=True, capturing stdout: ``` PRINTED: '\tLatent norm:
<bound method Tensor.norm of tensor([[-2.8552e-02, 7.2031e-03, -1.6689e-02, 2.1267e-05,
9.5957e-04,' ``` It …

**NSM/reconstruct/main.py:834 — `sigma_rand_pts` default differs by 10x between reconstruct_mesh and get_mean_errors**  
`REPRODUCES`

The 10x default disagreement reproduces and is result-changing, but only when
get_rand_pts=True — both shipped configs set get_rand_pts_recon=false, so it is latent in
production. Fixed = one default, documented. File as one issue with the chamfer_norm entry;
drop the n_pts_per_axis sentence, which is not the same pattern.

*Evidence:* reconstruct_mesh `sigma_rand_pts=0.001` at :864; get_mean_errors
`sigma_rand_pts=0.01` at :1320, forwarded unconditionally at :1373. Ran scratchpad/t_sigma.py
(get_rand_pts=False, the shipped setting): ``` sigma 0.001 assd_0 0.21913301750054126 sigma
0.01 …

**NSM/reconstruct/main.py:1153 — Unconditional debug prints on the production reconstruction path**  
`REPRODUCES`

Reproduced by execution, and the entry understates the scope — the noisiest offender is in
NSM/mesh, not reconstruct. The consumer runs the NSM fit as a subprocess whose stdout the
pipeline scans, so this is a real hygiene problem. Fixed = every print in the reconstruction
path gated on verbose or routed through the module logger. One issue covering this and the
:750 bound-method print.

*Evidence:* Now :1197, :1198, :1209, :1234, and :1452 in get_mean_errors. Ran
scratchpad/t_recon_full.py with verbose=False, capturing stdout: ``` '... length of meshes:
2\nlength of orig_mesh: 2\nfinished computing recon loss\n' ``` Ran scratchpad/t_gme.py — …

**NSM/mesh/main.py:836 — get_sdfs prints one unconditional line per batch in the production path**  
`REPRODUCES`

Reproduces with a measured line count, on the live path, at the caller's own defaults, from a
function whose siblings all gate on `verbose`. Fixed means get_sdfs takes `verbose` (or a
logger) and is silent by default. The audit's framing of the downstream harm is fragile-not-
broken; the noise itself is the finding.

*Evidence:* Print now at NSM/mesh/main.py:857, warning at :841-843. `get_sdfs` has no
`verbose` parameter at all (t_main.py: `has 'verbose' param? False`). Ran t_spam.py —
create_mesh_adaptive at reconstruct_mesh's own defaults (batch_size=32**3, n_pts_coarse=64), …

**NSM/mesh/refine_mesh.py:399 — get_target_cells raises UnboundLocalError on its own default arguments**  
`REPRODUCES` · owned by Not on GitHub. Already ruled in docs/SCOPE.md §2.3 condition 1, indexed in docs/ARCHITECTURE.md:210 and :310, and queued in .claude/plans/NSM_CODE_HEALTH_REFACTOR.md:360 — but the repo's stated rule is that issues are the only work queue, and there is no issue for it. · folded, listed first

Reproduces exactly as written, and it is the one entry in this slice that makes a whole module
unusable. Fixed means `np.zeros_like(max_lengths)` plus a test that calls both public entry
points with their own defaults. It gates SCOPE.md §2.3's other two conditions, which cannot be
written against code nobody can run.

*Evidence:* Bug now at NSM/mesh/refine_mesh.py:400: `max_length_binary =
np.zeros_like(max_length_binary)`. Ran t_refine.py sections A and B on a 240-cell pyvista
Sphere: {'area_threshold': 0.5} -> UnboundLocalError: local variable 'max_length_binary'
referenced before …

**NSM/mesh/main.py:280 — sdf_grid_to_mesh crashes on numpy input while its VTK twin does not**  
`REPRODUCES`

Two functions swapped by a single unrelated boolean must accept the same inputs. Fixed means
both guard with `hasattr(sdf_values, 'cpu')` and both carry the same narrow_band default —
file it together with the :277 asymmetry as one small issue, not two. Production always feeds
a torch tensor with use_vtk=True, so the severity is API hygiene, not a live crash.

*Evidence:* Unguarded conversion now at NSM/mesh/main.py:282; the guarded twin at :402-403.
Ran t_main.py section C with a float32 numpy sphere SDF: numpy into sdf_grid_to_mesh RAISED:
AttributeError 'numpy.ndarray' object has no attribute 'cpu' numpy into …

**NSM/mesh/main.py:638 — Fallback grid origin can disagree with search_bounds**  
`REPRODUCES`

A reproducible, wrong-by-construction grid on a reachable branch, with a one-line fix (derive
voxel_origin from search_bounds when the caller did not supply one). Fixed means the fallback
grid covers exactly search_bounds. Unreachable at today's defaults, so no KNOWN_ISSUES History
entry is owed — nobody has affected runs.

*Evidence:* voxel_size derived from search_bounds at NSM/mesh/main.py:644-645; the fallback
hands create_mesh the untouched voxel_origin at :693-696 (default (-1,-1,-1) at :561). Ran
t_aabb.py, calling create_grid_samples exactly as the fallback does: …

**NSM/train/train_deep_sdf.py:108 — The shipped default_config.json cannot drive train_deep_sdf — six unconditionally-read keys are missing**  
`REPRODUCES`

Five of the six named keys are genuinely fatal on the shipped default; `assd` is not reachable
from it, so the entry is right in substance and slightly overstated in count. Real, fixable,
untracked: "fixed" = default_config.json defines the keys the trainer reads unconditionally,
pinned by a test that instantiates the trainer from it.

*Evidence:* Ran the real trainer on the shipped default config (tiny triplanar architecture
substituted so a model exists; CPU; synthetic meshes from testing/NSM/regression/_harness.py),
adding one missing key at a time. Script: scratchpad/e1_defaultcfg.py + e1b.py. …

**NSM/train/train_deep_sdf.py:121 — resume_epoch == 1 silently skips epoch 1 without resuming anything**  
`REPRODUCES`

Exactly as described, demonstrated end to end. "Fixed" = resume_epoch==1 either loads the
epoch-1 checkpoint or raises; the two guards must use the same boundary.

*Evidence:* Ran train_deep_sdf three times on the CPU harness with n_epochs=4, recording which
epochs train_epoch was actually called for (scratchpad/e2_resume.py): resume_epoch=0: epochs
run [1, 2, 3, 4] resume_epoch=1: epochs run [2, 3, 4] <- no checkpoint loaded, no …

**NSM/train/train_deep_sdf.py:195 — schedule_free eval warm-up passes a (dict, tensor) dataloader tuple straight into the decoder**  
`REPRODUCES`

Crashes at the first checkpoint or validation epoch for every schedule_free_* run, so that
whole optimizer family is unusable. "Fixed" = the warm-up unpacks the batch and builds decoder
inputs the way train_epoch does, or the unfinished block is deleted.

*Evidence:* `schedulefree` is not installed in nsm-dev, so I stubbed NSM.utils.schedulefree
with an AdamWScheduleFree that subclasses torch.optim.Adam and adds no-op train()/eval() — the
only methods train_deep_sdf's schedule_free branch touches — then ran the real …

**NSM/train/train_deep_sdf.py:198 — save_model_params is called on every checkpoint but silently no-ops after the first**  
`REPRODUCES`

Reproduces, and the resume case is real: a resumed run's model_params_config.json keeps the
first run's hyperparameters and mesh list, and that file is what downstream consumers read.
Should be filed once, jointly with the NSM/utils.py:312 entry from the utils slice, not twice.

*Evidence:* scratchpad/e4_smp.py — two calls with different configs into one
experiment_directory: first : {'resume_epoch': 0, 'n_epochs': 10, 'lr': 0.001,
'list_mesh_paths': ['a.vtk']} second: {'resume_epoch': 0, 'n_epochs': 10, 'lr': 0.001,
'list_mesh_paths': …

**NSM/train/train_deep_sdf.py:333 — The per-surface index ordering is a fully undocumented positional contract spanning four modules**  
`REPRODUCES`

The mechanism reproduces but the entry OVERSTATES the consequence: swapping meshes does not
"train bone weights against cartilage targets" — training is self-consistent, the decoder
simply learns whichever column it is given. The real harm is that mesh_names is written into
model_params_config.json as ground truth for downstream consumers and can be silently wrong,
which is precisely what CLAUDE.md added mesh_names to prevent. File that, not the broader
four-module framing. Absorbs the :620 entry.

*Evidence:* scratchpad/e16_surfidx.py — built the dataset from the SAME meshes with the pair
order reversed, kept mesh_names=['bone','cart'], ran a full epoch: normal order :
['subject0_bone.vtk', 'subject0_cart.vtk'] swapped order : ['subject0_cart.vtk', …

**NSM/train/utils.py:63 — add_plain_lr_to_config retains a positional back door that a test pins to mislabelled output**  
`REPRODUCES` · owned by #20 is the adjacent class sweep ("parameters accepted and never read"); these are read, but by no caller

The parameters exist solely so a test can assert deliberately swapped labels, in the very
function whose Aug-2026 fix was about eliminating positional mapping. "Fixed" = delete
idx_model/idx_latent and delete test_explicit_indices_still_override. Absorbs the :76 entry.

*Evidence:* scratchpad/e7_lrutils.py: add_plain_lr_to_config(cfg, idx_model=1, idx_latent=0)
-> model_lr_initial = 0.001 (the LATENT entry's Initial) -> latent_lr_initial = 0.005 Only
caller in the entire repo (grep -rn 'idx_model|idx_latent' NSM/ testing/, excluding the …

**NSM/train/train_deep_sdf.py:569 — step_mean_vec_length / step_std_vec_length are assigned, not accumulated — logged latent-norm metrics are wrong by a factor of len(data_loader)**  
`REPRODUCES`

Confirmed numerically: the reported value is (last chunk of last batch) / n_batches, so it
shrinks as the dataset grows. "Fixed" = `+=` at 582-583, matching the surrounding
accumulators. Only affects wandb logging, not gradients — worth an issue, not a KNOWN_ISSUES
History entry.

*Evidence:* scratchpad/e5_run.py — real 2-epoch CPU run, reading the latent embedding directly
after the epoch: n_batches (len(data_loader)): 2 true mean latent norm over whole embedding:
0.01071708 logged mean_vec_length: 0.00526427 logged std_vec_length : 0.0 logged * …

**NSM/train/train_deep_sdf_multi_head.py:83 — multi_head never moves latent vectors to the device and hardcodes .cuda()**  
`REPRODUCES`

Confirmed three ways: the module only runs at all because the latents are LEFT on CPU, it
cannot run on cpu/mps despite advertising config["device"], and the naive fix breaks it.
Should be one issue with the other multi_head divergences (:118, :123, :359, :382), since
SCOPE.md 2.1 already rules the module must be repaired.

*Evidence:* scratchpad/e9_multihead.py — ran multi_head on the T4 with device="cuda", spying
on get_optimizer: latent_device: 'cpu' (config["device"] was "cuda") scratchpad/e11_mh3.py —
same module with device="cpu": RuntimeError: Expected all tensors to be on the same …

**NSM/train/train_deep_sdf_multi_head.py:118 — Non-short-circuit `&` on membership tests raises KeyError instead of skipping**  
`REPRODUCES`

Executed, not inferred, at both sites. "Fixed" = `and` / `config.get`, matching
train_deep_sdf. Group with the other multi_head divergences.

*Evidence:* Both sites hit at runtime while walking a real multi_head run forward
(scratchpad/e9_multihead.py, successive runs): KeyError : 'surface_weighting' | multi_head
line 330: isinstance(config["surface_weighting"], (list, tuple)) KeyError : 'val_paths' | …

**NSM/train/train_deep_sdf_multi_head.py:123 — torch.mps.empty_cache() is called on the CPU branch**  
`REPRODUCES`

The code is wrong as written and the shipped "cuda:0" never matches "cuda", both confirmed.
But the entry implies line 123 fires in practice; it cannot, because the cpu path dies
earlier. Mildly overstated. Real, trivial, and belongs in the same multi_head repair (use
NSM.utils.clear_gpu_cache, as train_deep_sdf does).

*Evidence:* scratchpad/e10_mh2.py: 120: if config["device"] == "cuda": 121:
torch.cuda.empty_cache() 122: elif config["device"] == "cpu": 123: torch.mps.empty_cache()
shipped default device string: cuda:0 -> == 'cuda' ? False torch.mps.empty_cache() ->
RuntimeError: …

**NSM/train/train_deep_sdf_multi_head.py:359 — multi_head accumulates per-surface L1 with .append() into a fixed-size list, then discards it**  
`REPRODUCES`

Both halves confirmed, with train_deep_sdf as the controlled contrast. "Fixed" = index-assign
into batch_l1_losses inside the split loop and accumulate that, as
train_deep_sdf.py:513-514/579-580 does. Group with the other multi_head divergences.

*Evidence:* AST scan of MH.train_epoch (scratchpad/e10_mh2.py): batch_l1_losses appears
exactly twice — batch_l1_losses = [0.0 for _ in range(n_surfaces)]
batch_l1_losses.append(l1_loss_.sum().item()) never read. Second half — per-surface losses
come from the last split …

**NSM/train/train_deep_sdf_multi_head.py:382 — multi_head hardcodes a 100-epoch code-regularization warm-up instead of reading the config key**  
`REPRODUCES`

Confirmed by execution and by whole-file search: the config key literally does not appear in
the module, so setting it through this entry point does nothing. "Fixed" = read the config
key. Group with the other multi_head divergences.

*Evidence:* scratchpad/e10_mh2.py: 382: config["code_regularization_weight"] * min(1, epoch /
100) * l2_size_loss code_regularization_warmup present in multi_head source: False
train_deep_sdf.py:557 uses min(1, epoch / config["code_regularization_warmup"]).

**NSM/train/utils.py:90 — add_plain_lr_to_config raises KeyError on a Constant learning-rate schedule**  
`REPRODUCES`

Reproduces, and the asymmetry (Constant supported by get_learning_rate_schedules, fatal in the
logging helper) is the whole point. "Fixed" = read Initial defensively (fall back to Value, or
skip absent keys) and add a Raises note; pinned by a test that trains one epoch on a Constant
schedule.

*Evidence:* scratchpad/e7_lrutils.py, with {"Type":"Constant","Value":...} entries carrying
proper Targets: get_learning_rate_schedules OK? yes -> {'model': 0.005, 'latent': 0.001}
add_plain_lr_to_config -> KeyError('Initial') at NSM/train/utils.py:87 mixed (one …

**NSM/models/deep_sdf.py:171 — progressive_add_depth path propagates None through the layer stack**  
`REPRODUCES`

Reproduces exactly, and the failure window covers every realistic training run.
ARCHITECTURE.md §7 names the class ('constructible-but-uncallable configuration', 5 instances)
but there is no GitHub issue for it. File one class issue covering this, deep_sdf:180,
deep_sdf:308 and two_stage:24; 'fixed' means each of those constructor options either works or
refuses at construction time.

*Evidence:* Code now at NSM/models/deep_sdf.py:172 (bare `return` in forward_branch_) and
:218-219 (`if x is None: continue`). Ran /tmp/.../scratchpad/v3.py: for ep in
(1,199,200,1010,1209,1500): Decoder(latent_size=8, dims=[16]*8, …

**NSM/models/deep_sdf.py:180 — Decoder indexes self.bn by absolute layer index but appends only for norm layers**  
`REPRODUCES`

Reproduces. Same class as the two entries above; group them into one issue rather than filing
three. Note norm_layers is marked DEPRECATED at deep_sdf.py:44 and loader.py:324, so the fix
may be deletion rather than repair.

*Evidence:* Code now at NSM/models/deep_sdf.py:138 (`self.bn.append(...)` only inside the
`elif ... layer in self.norm_layers` branch) and :181 (`x = self.bn[layer_idx](x)`). Ran: d =
Decoder(latent_size=8, dims=[16,16,16], weight_norm=False, norm_layers=(2,)) …

**NSM/models/deep_sdf.py:308 — activation='linear' builds a Decoder that crashes on first forward**  
`REPRODUCES`

The crash reproduces; the 'advertised' framing is mildly overstated. Still worth fixing as
part of the class issue — get_activation returning a bare None for a value the caller cannot
distinguish from a mistake is the defect.

*Evidence:* get_activation now at NSM/models/deep_sdf.py:288-310; `elif activation ==
"linear": return None` at :307-308. Ran: Decoder(latent_size=8, dims=[16,16],
activation='linear')(torch.randn(4,11)) -> TypeError: 'NoneType' object is not callable (from
:182, `x = …

**NSM/models/triplanar.py:158 — Latent gradients are scaled by the number of query points**  
`REPRODUCES`

Reproduces exactly, and I sharpened it past the entry: the amplification is asymmetric between
the reconstruction term and the L2/norm-penalty terms, so the configured latent regularization
is silently diluted by the number of query points on the production reconstruction path. Not a
regression (the legacy path always did this), so any fix needs a KNOWN_ISSUES.md §History
entry per CLAUDE.md § Numerical-behaviour changes. 'Fixed' means the recon and regularization
gradients on the latent are on the same scale, with the chosen convention stated.

*Evidence:* FastUnique is now triplanar.py:153-171; `expanded_grad =
grad_output.repeat(ctx.num_points, 1)` at :170 returns (N,D) for a (D,) input and relies on
autograd's sum-to-size. Ran v5.py — fast path, legacy path, and a manual decode-once
reference, N=10: fast/ref …

**NSM/models/loader.py:123 — conv_norm_type default differs depending on which code path builds the triplanar model**  
`REPRODUCES`

The divergence is real, undocumented, and duplicated six ways; both shipped models use 'layer'
while three of the six sites default to 'batch'. Low severity because the mismatch fails
loudly. 'Fixed' means one constant, sourced once, and a stated reason for whichever value
wins.

*Evidence:* Five sites on current main, executed via loader:
_get_triplanar_params({'latent_size':16})['conv_norm_type'] -> 'batch' (loader.py:124)
_get_two_stage_params({'latent_size':16})['triplanar_params']['conv_norm_type'] -> 'layer'
(loader.py:190) …

**NSM/models/triplanar.py:262 — sum_sdf_features=False silently produces two empty feature planes**  
`REPRODUCES`

Reproduces and produces a plausible number with two thirds of the representation structurally
absent — the definition of a landmine. Untracked. 'Fixed' means either slicing by
sdf_latent_size//3 in the non-summing case, or refusing the configuration at construction; the
correct divisor is stated in the entry and I confirmed the arithmetic.

*Evidence:* Sizing at triplanar.py:231-237 (`vae_out_features = self.sdf_latent_size` when not
summing), slicing at :275-281 (`latent_size = self.sdf_latent_size + self.conv_pred_sdf`).
Executed: m = TriplanarDecoder(latent_dim=16, sdf_latent_size=6, …

**NSM/models/two_stage.py:24 — TwoStageDecoder cannot be constructed with its own defaults**  
`REPRODUCES`

Reproduces — the class is unconstructible by any direct caller, at any argument. Same class as
the three deep_sdf entries; one issue, four instances. ARCHITECTURE.md §7 already names it in
the class table but no issue owns it.

*Evidence:* default_mlp_params['dims'] is still a tuple at two_stage.py:24; deep_sdf.py:84
does `self.dims = [latent_size + 3] + dims`. In a fresh process: TwoStageDecoder() ->
TypeError: can only concatenate list (not "tuple") to list TwoStageDecoder(latent_size=8, …

**NSM/utils.py:9 — Importing NSM prints to stdout unconditionally when schedulefree is absent**  
`REPRODUCES`

Reproduces exactly on current main. One caveat: the entry's stated consequence is stale — it
says the consumer 'parses stdout (progress lines followed by a JSON result as the last line)',
but kneepipeline now reads `_step_result.json` and its CLAUDE.md explicitly says the result is
NOT the last line of stdout. The remaining, still-valid point is the plain one: a library must
not write to stdout at import. Three-line fix (warnings.warn, move `import warnings` above the
try).

*Evidence:* RAN: `/mnt/data/conda-envs/nsm-dev/bin/python -c "import schedulefree"` ->
ModuleNotFoundError: No module named 'schedulefree' (it is genuinely absent from nsm-dev).
RAN: `/mnt/data/conda-envs/nsm-dev/bin/python -c "import NSM" 2>/dev/null` -> stdout: …

**NSM/utils.py:283 — save_model's on-disk subdirectory naming is an undocumented contract**  
`REPRODUCES` · owned by Adjacent to KNOWN_ISSUES.md §2 / SCOPE.md §2.1 (train_deep_sdf_multi_head), but the loader gap is not stated in either.

Stronger than the entry states. It is not merely undocumented — nothing in the repo can read a
`model_N/` checkpoint back. Multi-decoder runs (train_deep_sdf_multi_head, the only producer)
write checkpoints no loader in NSM/ or examples/ can consume. 'Fixed' means: one documented
naming rule plus a loader that handles both shapes, or drop the branch.

*Evidence:* RAN utils_check.py: list of 1 decoder -> `['model']`; list of 2 ->
`['model_0','model_1']`; list of 3 -> `['model_0','model_1','model_2']`; bare decoder ->
`['model']`. Docstring (NSM/utils.py:255-262) documents only the optimizer-target validation.
Then RAN …

**NSM/utils.py:312 — save_model_params silently refuses to overwrite and silently drops non-JSON config values**  
`REPRODUCES`

Reproduces exactly and it is user-visible: model_params_config.json is what every downstream
consumer reads to rebuild a model, and on a resumed or re-run experiment it silently describes
different hyperparameters than the weights beside it. 'Fixed' = overwrite (or epoch-stamp) and
log the keys filter_non_jsonable removes.

*Evidence:* RAN utils_check.py: first call with `{lr:0.001, resume_epoch:0, bad:<object>}`
writes the file, and `'bad' key present? False` (filter_non_jsonable dropped it with no log).
Second call with `{lr:0.9999, resume_epoch:500}` and `list_mesh_paths=['b.vtk']` -> on …

**NSM/utils.py:343 — get_latent_vecs silently ignores config['latent_bound'] and doubles latent_size when variational**  
`REPRODUCES` · owned by Same class as #20 ('parameters accepted and never read'); could be swept there rather than filed alone.

Reproduces exactly. A user who sets `variational: true` alongside the shipped `latent_bound:
1.0` silently gets an effectively unbounded latent, and the `latent_size` recorded in
model_params_config.json is half the real embedding width. 'Fixed' = honour latent_bound (or
reject the combination) and record the effective latent_size.

*Evidence:* RAN utils_check.py: `get_latent_vecs(10, {latent_size:8, latent_bound:1.0,
variational:True})` -> `embedding dim 16, max_norm 1000`; with `variational:False` ->
`embedding dim 8, max_norm 1.0`. `get_latent_vecs.__doc__ = None`. RAN `grep -rn variational …

**NSM/utils.py:410 — symmetric_chammfer is an empty stub with an empty docstring**  
`REPRODUCES`

Reproduces; three dead lines in a public module namespace that hand any caller None. On its
own this is a deletion, not an issue — I have bucketed it ISSUE only so it can carry the dead-
public-symbol class alongside sdf_gradients and CLASSIFICATION_HEADS_GROUP_NAME. If the parent
files one sweep issue for that class, these three are its instances; if not, just delete the
function.

*Evidence:* RAN utils_check.py: `utils.symmetric_chammfer(1, 2, 3)` -> `None`; `__doc__` -> `'
'` (whitespace only, so it passes a naive has-docstring check). RAN `grep -rn
symmetric_chammfer --include=*.py NSM/ testing/` -> only NSM/utils.py:410, the definition.
Still …

**NSM/mesh/correspondence_metrics.py:537 — Adjacent metrics take the same two arrays in opposite positional order**  
`REPRODUCES` · owned by docs/ARCHITECTURE.md:304 already tracks this class with a count of ~12 instances, but names other sites, not this one. · folded into the sibling-API disagreement issue

Reproduces exactly as described, and both failure modes are silent. Mitigation: the module has
no non-test importer, and the only in-repo caller (score_correspondence) passes them
correctly. Fix is free and lossless — make the two signatures agree, or make both keyword-
only. 'Fixed' = a caller cannot swap them without a TypeError.

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/corr_check.py: `roundtrip_distance(A,B)` and
`roundtrip_distance(B,A)` return identical per_vertex arrays and identical mean
0.11065713875408247 — a swap is completely invisible. `forward_backward_disagreement(B,A)` vs
…

**NSM/mesh/correspondence_metrics.py:676 — score_correspondence fabricates an 'original' when source_mesh is missing**  
`REPRODUCES` · folded into the missing-input-validation issue

Reproduces: a plausible-looking number where every other missing-input path in the same
function returns {'skipped': True, ...}. 'Fixed' = when roundtrip_points is given and
source_mesh is None, skip both metrics with a reason (or raise), and add the test case that
currently does not exist.

*Evidence:* RAN corr_check.py: `score_correspondence(warped, target, source_mesh=None,
roundtrip_points=warped.points+0.01)`. Result: `foldover_count -> {'skipped': True, 'reason':
'source_mesh not provided'}` but `roundtrip_distance -> {'min': 0.01732049, ..., 'mean': …

**NSM/mesh/correspondence_metrics.py:30 — Unused import, masked by a project-wide flake8 ignore**  
`REPRODUCES`

Reproduces, and contradicts the task brief's 'flake8 is at zero; unused imports are gone' —
they are not gone, they are invisible, because F401 is ignored package-wide. The single import
is a one-line deletion, but the durable item is the class: 43 masked F401s and a `make lint`
that cannot see any of them. 'Fixed' = drop F401 from extend-ignore, annotate the deliberate
re-exports, delete the rest.

*Evidence:* RAN `flake8 --isolated --select=F401,F811,F841 NSM/mesh/correspondence_metrics.py`
-> `NSM/mesh/correspondence_metrics.py:30:1: F401 'NSM.mesh.triangle_metrics.get_edge_lengths'
imported but unused` (exit 1). RAN `flake8 NSM/mesh/correspondence_metrics.py` …

**NSM/mesh/correspondence_metrics.py:291 — faces.reshape(-1, 4) assumes an all-triangle mesh in three places**  
`REPRODUCES`

Reproduces, including the dangerous mixed-mesh case, which I had to construct deliberately
(the entry's own numbers did not include a worked example). Same class as the interpolate.py
`reshape(-1, 3)` entry, so one issue should cover all five sites. 'Fixed' = the face accessor
validates (or uses `regular_faces` and raises on non-triangular input) in one shared helper.

*Evidence:* RAN corr_check.py + corr_check2.py. Pure-quad PolyData (2 quads, flat length 10):
`self_intersection_count` and `foldover_count` both raise `ValueError: cannot reshape array of
size 10 into shape (4)`. MIXED mesh of 4 quads + 3 triangles (flat length 4*3+5*4 …

**NSM/mesh/interpolate.py:116 — sdf_gradients returns a gradient array whose first D_lat columns are always zero**  
`REPRODUCES`

Reproduces exactly. With a 256-dim production latent the returned array is 98.8% zero padding
presented as a gradient, and a caller slicing `[:, :D_lat]` gets silent zeros. Given zero
callers, the cheap fix is to return `grad_pos` (B,3) or delete the function — same sweep as
symmetric_chammfer. 'Fixed' = the returned shape contains no fabricated zeros.

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/interp_check.py with a 2-surface decoder and
D_lat=8: `sdf_gradients(model, pts(5,3), z, surface_idx=0)` -> shape `(5, 11)`, `first D_lat
cols all zero: True`, `last 3 cols nonzero: True`. With `surface_idx=None` -> list of …

**NSM/mesh/interpolate.py:307 — faces argument silently accepts pyvista's VTK-style face array and builds garbage**  
`REPRODUCES`

Reproduces, with numbers the entry did not have. This is the highest-consequence instance of
the face-array class: the wrong input silently changes the smoothing operator and the pinned-
vertex set, so the interpolation output is wrong rather than absent. One issue with the
correspondence_metrics/refine_mesh sites.

*Evidence:* RAN interp_check.py. `pv.Sphere(theta_resolution=8,
phi_resolution=8).triangulate()`: 96 triangles, flat `.faces` length 384, 384 % 3 == 0.
`build_mesh_laplacian(mesh.faces, ...)` **did not raise**: nnz 373 (flat) vs 288
(regular_faces), and …

**NSM/mesh/interpolate.py:519 — The is_mesh path mutates the caller's mesh in place and does not say so**  
`REPRODUCES` · owned by Same class as the get_pts_center_and_scale in-place mutation just fixed by PR #38 (issues #20/#21); that fix did not sweep this site.

Reproduces, and far more destructively than the entry conveys — a 341x point-count change to
an object the caller still holds, from a function whose `return data` reads as pure. The repo
has just fixed this exact class elsewhere (PR #38 on get_pts_center_and_scale), which is the
argument for sweeping rather than leaving it. 'Fixed' = copy the mesh (or document the
mutation and stop returning it).

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/interp_mesh_check.py: built a
`pymskt.mesh.Mesh` from an 82-point sphere and called `interpolate_mesh(model, z1, z2,
n_steps=2, mesh=m, adaptive=True, smooth=True)`. Result: `returned object IS the argument:
True`; **the …

****NSM/dependencies/sinkhorn.py:49 — Unreachable duplicated type check in sinkhorn; max_iters is never type-checked****  
`REPRODUCES`

Executed and reproduces. Fixed = line 48 tests max_iters, not p; the message it already
carries then becomes true. Group with the other three sinkhorn validation findings — one
issue, four sites. Note the file is a vendored copy of fwilliams/scalable-pytorch-sinkhorn
(cited in requirements.txt), so the fix is a local-fork decision.

*Evidence:* Lines unchanged: sinkhorn.py:40 `if not isinstance(p, int)` and :48-49 `if not
isinstance(p, int): raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")`.
Ran: `sinkhorn(torch.rand(5,3), torch.rand(5,3), max_iters="banana")` -> `TypeError: …

****NSM/dependencies/sinkhorn.py:92 — sinkhorn validates w_x's length twice and never validates w_y's****  
`REPRODUCES`

Executed and reproduces. Fixed = line 91 reads `w_y.shape[0] != y.shape[0]`. Same one-issue
group as the other sinkhorn validation defects.

*Evidence:* sinkhorn.py:76 and :91 both read `if w_x.shape[0] != x.shape[0]`; the message at
:93-96 talks about w_y and y. Ran: `sinkhorn(x(5,3), y(5,3), w_x=ones(5)/5, w_y=ones(9)/9)` ->
passed validation and died inside keops with `ValueError: Incompatible values for …

****NSM/dependencies/sinkhorn.py:107 — sinkhorn's default uniform weights cannot sum equal when the two point clouds differ in size, so it always raises****  
`REPRODUCES`

The sinkhorn defect is real and executed. But the register attributes the EMD failure to
unequal sizes when in fact `calc_emd=True` NEVER works from compute_recon_loss for any input,
because the only caller passes numpy. That is the bigger, unrecorded finding and it should be
the issue: fixed = compute_recon_loss(calc_emd=True) returns a number for meshes of unequal
point count, which needs both the numpy->torch conversion and the sinkhorn weight
normalisation.

*Evidence:* Ran: `sinkhorn(torch.rand(5,3), torch.rand(7,3))` -> `ValueError: Weights w_x and
w_y do not sum to the same value, got w_x.sum() = 1.0 and w_y.sum() = 0.7142857313156128`.
Code at sinkhorn.py:106-118 unchanged. BUT the stated caller impact is wrong. …

****NSM/reconstruct/recon_evaluation.py:95 — compute_recon_loss mutates the caller's meshes to float32 in place****  
`REPRODUCES`

Executed and reproduces on both the recon mesh and the caller's original. Fixed = cast a local
copy for the ASSD call, or document that calc_assd downcasts. Unlike the cartilage_func
sibling this one really does alias the caller's objects, because it assigns to `.point_coords`
rather than constructing a new mesh.

*Evidence:* Now at recon_evaluation.py:103-106. Ran compute_recon_loss on float64 pymskt
Meshes: ASSD before: float64 float64 ASSD after : float32 float32 {'assd_0':
0.04966860369175949} Same call with calc_assd=False leaves both at float64. These are `meshes`
and …

****NSM/reconstruct/reconstruct_latent_S3.py:127 — reconstruct_latent_S3 references an undefined name in its own error path and calls wandb without importing it****  
`REPRODUCES` · owned by #35 (covers only the latent_loss_ half)

Two of the five sub-claims are fixed; three reproduce. The div-by-zero is unrecorded and is
strictly broader than described (it fires on any num_iterations < n_lr_updates, defaults
included). Fixed = S3 uses reconstruct_latent_get_lr_update_freq instead of its own division,
and latent.norm is called. Best folded into #35 as "the S3 copy of the main-path arithmetic is
unguarded".

*Evidence:* Ran reconstruct_latent_S3 on CPU with an 8-latent MLP decoder and a 32x4 SDF
tensor: - bad shape -> `ValueError: Inputted SDF must have shape Nx3 or Nx4 got:
torch.Size([32, 5])` == FIXED (d2ba1c7; line 129 now reads new_sdf.shape). `import wandb` is
present …

**### [misleading] `docs/MULTI_SURFACE_REGISTRATION.md:75`**  
`REPRODUCES`

Executed and reproduces on current main. This is the one entry in my slice that is a genuine
untracked code defect wearing a documentation anchor. Fixed = MultiSurfaceSDFSamples with a
list-valued mesh_to_scale returns the combined reference mesh instead of raising; the
`Mesh(mesh)` at :1440 has to move into the branches that bind `mesh`.

*Evidence:* Code moved to NSM/datasets/sdf_dataset.py:1424-1440. Line 1435 sets
`self.reference_mesh = combine_meshes(...)` without binding `mesh`; line 1440 unconditionally
runs `self.reference_mesh = Mesh(mesh)`. Reproduced by subclassing MultiSurfaceSDFSamples to
skip …

</details>


---

# 2. Fold into existing issues

**parameter accepted and never read / silently rebound before it is read** → No new issue — append to #20 (Sweep: parameters accepted and never read)

- `NSM/datasets/sdf_dataset.py:184 (n_pts_random)`
- `NSM/datasets/sdf_dataset.py:169 (mean — the only unread param in either sampler)`
- `NSM/mesh/main.py:171 (scale_mesh rebinds scale/offset before reading them)`
- `NSM/models/deep_sdf.py:47 (xyz_in_all)`
- `NSM/models/deep_sdf.py:87 (latent_noise_sigma)`
- `NSM/models/triplanar.py:312 (normalize_coordinates ignores padding)`
- `NSM/utils.py:394 (get_optimizer drops weight_decay for Adam)`
- `NSM/utils.py:343 (get_latent_vecs ignores latent_bound)`
- `NSM/reconstruct/main.py:873 (batch_size_latent_recon absorbed by **kwargs)`
- `NSM/train/train_deep_sdf.py:279 (train_epoch return_loss / verbose)`
- `NSM/datasets/sdf_dataset.py:87 and :91 — already fixed by b0c8bf5 / PR #38`

**cache key does not cover what changes cached content** → No new issue — all four are #19, already annotated in-source and pinned by strict xfails

- `NSM/datasets/sdf_dataset.py:1396 (uniform_pts_buffer, subsample)`
- `NSM/datasets/sdf_dataset.py:1406 (reference_mesh hashed by str())`
- `NSM/datasets/sdf_dataset.py:1973 (mesh_to_scale)`
- `NSM/datasets/sdf_dataset.py:1310 (find_hash matches across date folders)`
- `NSM/datasets/sdf_dataset.py:1595 (joint_scale_buffer is also unhashed — but its own issue is that it is unreachable)`

**store_data_in_memory=True / save_cache=False is unusable end to end** → No new issue — widen #22's fix statement to cover the whole configuration, then close #1

- `NSM/datasets/sdf_dataset.py:2158 (UnboundLocalError 'time_' in __getitem__ — #22 verbatim)`
- `NSM/datasets/sdf_dataset.py:1061 (KeyError 'new_pts_0' at construction with scale_jointly=True)`
- `NSM/datasets/sdf_dataset.py:1726 (FileNotFoundError writing list_meshes_started_loading.log when save_cache=False)`
- `NSM/datasets/sdf_dataset.py:1046 (joint_scale_buffer asymmetry — unobservable, the branch crashes first)`

**zero-count / degenerate probability allocation dies deep inside the sampler** → No new issue — widen #23 to cover the whole degenerate-probability path

- `NSM/datasets/sdf_dataset.py:671 (mixed zero allocation → ValueError at the concat, a second crash site)`
- `NSM/datasets/sdf_dataset.py:849 (check_probabilities rejects scalar int 0 or 1 with a message that is wrong about what it accepts; the list branch has no float check at all)`
- `#23's own trigger: uniform p_near=p_far=0 → ValueError inside point_cloud_utils`

**two calling conventions for decoders in one library** → No new issue — SCOPE.md §1 already rules it a Phase 4 work item

- `NSM/reconstruct/main.py:588 (two verbatim-duplicate entries under different titles — dedupe)`
- `NSM/mesh/interpolate.py:98 (three divergent hand-rolled invocation conventions)`
- `NSM/mesh/main.py:862 (refuted — the two latent shapes are each interface's correct contract, and both named edge cases raise ValueError rather than going silent)`


---

# 3. Prose corrections (Phase 2)

False or missing docstrings and comments, corrected in the same commit as the code they
describe. No issue is filed for any of them.

<details><summary>62 entries</summary>

**### [cosmetic] `CLAUDE.md:152`**  
`REPRODUCES`

Executed and reproduces exactly as written. One clause: "...if `objects_per_decoder > 1` and
`mesh_names` is not provided".

**### [cosmetic] `CLAUDE.md:193`**  
`REPRODUCES`

Executed and reproduces, and is stronger than recorded: nothing anywhere constructs the group.
The code's docstring hedges correctly; CLAUDE.md should adopt the same hedge.

**### [cosmetic] `CONTRIBUTING.md:19`**  
`REPRODUCES`

Reproduces. Two-word find-and-replace.

**### [cosmetic] `DEVELOPMENT.md:131`**  
`REPRODUCES`

Reproduces and is understated by one line. Prose fix.

**### [cosmetic] `Makefile:20`**  
`REPRODUCES`

Reproduces in a changed and more damaging form — the help text now names targets that were
deleted, so following it errors immediately. Fix the help block, CLAUDE.md:22/:25 and
requirements-dev.txt:10 in one pass.

**### [cosmetic] `README.md:11`**  
`REPRODUCES`

Reproduces. Cookiecutter leftover in the Introduction; delete it, or promote the dependency
TODO somewhere real.

**### [cosmetic] `docs/KNOWN_ISSUES.md:25`**  
`REPRODUCES`

Reproduces at a new line number. This matters more than "cosmetic" suggests, because
KNOWN_ISSUES.md's stated purpose is letting a 2031 reader date a checkout against the fix —
and it names the branch that shipped only half of it.

**### [misleading] `.claude/plans/BREAKING_CHANGE_PROPOSAL.md:50`**  
`REPRODUCES`

Reproduces. Same one-character fix as its sibling; group them.

**### [misleading] `.claude/plans/BREAKING_CHANGE_PROPOSAL.md:51`**  
`REPRODUCES`

Reproduces. A ticked box for work that does not exist survives the State block, because the
State block does not contradict it. Untick it — that is the whole fix.

**### [misleading] `CLAUDE.md:120`**  
`REPRODUCES`

Reproduces. CLAUDE.md contradicts the module's own DeprecationWarning and SCOPE.md. One clause
fixes it.

**### [misleading] `CONTRIBUTING.md:87`**  
`REPRODUCES`

Half reproduces (the CONTRIBUTING commands), half is fixed (the CI wiring). Prose fix: point
the reader at `make install-dev`.

**### [misleading] `DEVELOPMENT.md:18`**  
`REPRODUCES`

Reproduces on everything I could execute or read. Prose fix: add the requirements.txt line to
both recipes, or point at `make install-dev`.

**### [stale] `CONTRIBUTING.md:53`**  
`REPRODUCES`

Reproduces, and now contradicts CLAUDE.md as well as practice. Prose fix in CONTRIBUTING.md;
the true policy is already written down elsewhere.

**### [stale] `DEVELOPMENT.md:202`**  
`REPRODUCES`

Reproduces. Point it at testing/testing_sdf_calculation_times/ or delete the block.

**### [stale] `README.md:215`**  
`REPRODUCES`

Reproduces and has got worse since the entry was written (three more unlisted files). Prose
fix.

**### [stale] `README.md:217`**  
`REPRODUCES`

Still inaccurate, but for the opposite reason to the one recorded: the README now UNDERSTATES
the state (it says planned/TODO when `make docs` works and the workflow is written). Rewrite
the paragraph to say the site builds and is gated on Phase 2.

**### [stale] `docs/MULTI_SURFACE_REGISTRATION.md:170`**  
`REPRODUCES`

Reproduces. It is the same defect as the sdf_dataset.py entry at docs/AUDIT_FINDINGS.md:716
(other slice) seen from the documentation side — file one prose fix covering both sites and
this doc's claim, not two.

****NSM/dependencies/sinkhorn.py:12 — sinkhorn's `p` is annotated float but rejected unless it is an int****  
`REPRODUCES`

Executed and reproduces. It is an annotation that contradicts the code one line later — a
prose/signature correction (`p: int = 2`), not an issue.

****NSM/dependencies/sinkhorn.py:31 — sinkhorn docstring calls eps the 'reciprocal' of the regularization parameter****  
`REPRODUCES`

Executed against the code rather than reasoned about. Docstring correction. Note it is
inherited from the vendored upstream, so fixing it forks the file's docstring.

****NSM/reconstruct/utils.py:42 — get_pt_cloud_distances docstring has d1 and d2 swapped****  
`REPRODUCES`

Executed and reproduces exactly, including the trap: the ASSD denominator is right only by
coincidence, so a reader who "fixes" it from the docstring introduces a bug. Pure docstring
correction — swap the two lines and say why the denominator is what it is.

****testing/NSM/test_lr_schedules.py:569 — Stale comment claims the config generator still writes on import****  
`REPRODUCES`

Executed and reproduces. A false comment in a test, contradicted by a sibling test — delete
the comment and the now-pointless chdir. Prose fix.

**NSM/datasets/sdf_dataset.py:1366 — Stale TODOs that name work the refactor plan should absorb**  
`REPRODUCES`

Comment rot, exactly as described. 'TODO: crat' is meaningless and should go; the
reference_object vs mesh_to_scale question is a real documentation gap that belongs in the
class docstring or ARCHITECTURE.md, not in a five-year-old inline TODO. Prose fix, no issue.

**NSM/datasets/sdf_dataset.py:1983 — Unexplained bare `False` literal inside the multi-surface hash parameter list**  
`REPRODUCES`

The literal reproduces; the entry's rationale for it ('presumably a frozen placeholder for a
removed parameter, kept so existing cache hashes still resolve') is falsified by git. This is
a missing-comment item, not a defect: annotate or drop the literal when #19 rewrites
get_hash_params, and do not repeat the compatibility story.

**NSM/datasets/sdf_dataset.py:2 — Seven unused imports at module top**  
`REPRODUCES`

Worth correcting the prompt's premise: the 'flake8 is at zero, unused imports are gone' item
is true for the linter, not the code. F401 was silenced globally with the rationale 'several
scratch / timing scripts', which also silences it for this production module. Still only
cosmetic plus a slightly wider star-export surface, so the entry itself has no fixable
statement beyond 'delete seven lines' — DELETE the entry, but do not record it as fixed.

**NSM/datasets/sdf_dataset.py:457 — False comment: 'vtkAppendPolyData' is claimed in three places and used nowhere**  
`REPRODUCES`

Wrong prose, correct code — exactly the Phase-2 shape. Minor correction to the entry: two
sites in this file, not three (the third is docs/MULTI_SURFACE_REGISTRATION.md, which the
Documentation-inaccuracies section already tracks separately at AUDIT_FINDINGS.md:1283).

**NSM/datasets/sdf_dataset.py:900 — Undocumented subclass initialization-order contract enforced by hasattr**  
`REPRODUCES`

The mechanism is real and executable, but nothing in-repo gets it wrong and there is no user-
visible defect today. What is actually missing is a comment at the hasattr block saying
subclasses must set these before super().__init__ — prose, not code.

**NSM/mesh/correspondence_metrics.py:1 — Plan and lint config point at an experiments/ tree that does not exist on main**  
`REPRODUCES`

Every sub-claim checks out, including the entry's own concession that the 39 is accurate.
Cheap prose fixes: drop the inert `experiments` line from `.flake8`, and either correct the
plan's 31 or (better, since CLAUDE.md says a completed plan keeps its body) add it to that
plan's Diverged section — a hand-transcribed count is exactly what CLAUDE.md § Four rules #1
forbids. No issue.

**NSM/mesh/correspondence_metrics.py:604 — score_correspondence's documented return shape omits its error branch**  
`REPRODUCES`

Reproduces. It is a docstring omission, not a behaviour bug — the error dicts are a deliberate
don't-crash design. Correct the Returns block to name the third shape; no issue needed.

**NSM/mesh/interpolate.py:291 — build_mesh_laplacian does not return a Laplacian**  
`REPRODUCES`

The matrix is what the entry says it is, but 'does not return a Laplacian' implies the reader
is misled, and the docstring tells them plainly. Only the name is loose. Cheapest honest fix
is a rename (`build_row_normalized_adjacency`) or a one-word summary-line change; either way
it is prose-level, and the entry overstates the hazard.

**NSM/mesh/interpolate.py:473 — interpolate_common is public, 18 parameters, zero docstring**  
`REPRODUCES`

Reproduces. The bulk of the fix is prose — a docstring on the only undocumented public
function in an otherwise thoroughly documented module. The adjacent 3-line code fix (make
`points1`/`mesh` required, replace the bare `Exception('Not implemented')` with a TypeError
naming the missing argument) can ride along. No issue.

**NSM/mesh/main.py:126 — scale_mesh_ 's trailing underscore promises in-place semantics it only sometimes has**  
`REPRODUCES`

The two aliasing contracts are real and executed. But the only caller is scale_mesh (:181)
inside the same module — there is no external consumer to be surprised. The whole remedy is a
docstring, which is the same work item as the `main.py:185` entry.

**NSM/mesh/main.py:185 — Five public functions in main.py have no docstring at all**  
`REPRODUCES` · owned by The plan .claude/plans/NSM_CODE_HEALTH_REFACTOR.md already carries "Deliverable: docstring coverage >=90% on surviving public API, lint-enforced."

Reproduces precisely. It is missing prose, already inside the plan's docstring-coverage
deliverable, and it absorbs the scale_mesh_ entry above.

**NSM/mesh/main.py:323 — band_width documented as world units, used as a voxel multiplier**  
`REPRODUCES`

The two halves of one sentence contradict each other, measured. Delete "in world units" in
both places. No behaviour change.

**NSM/mesh/main.py:341 — crop_sdf_to_narrow_band names every index variable for the wrong axis**  
`REPRODUCES`

Confirmed as stated, including the entry's own concession that it is functionally correct.
Rename the locals; nothing else changes.

**NSM/mesh/main.py:525 — create_grid_samples_in_bounds silently requires numpy arrays, not the tuples its docstring implies**  
`REPRODUCES`

Reproduces, but it raises immediately and loudly, has one internal caller that always passes
ndarrays, and the fix is `np.asarray(...)` or a clearer docstring. Lowest-value entry in the
slice; keep it only as prose.

**NSM/mesh/main.py:603 — create_mesh_adaptive docstring understates what n_pts_per_axis controls**  
`REPRODUCES`

"(for fallback only)" is false whenever voxel_size is None, which is the mean-mesh caller's
case. One-clause docstring fix.

**NSM/mesh/main.py:76 — coarse_bounds_from_sign_change returns None for two different reasons, one undocumented**  
`REPRODUCES`

Reproduces; the remedy is one clause in the Returns line (or a distinguishing warning). Not
worth an issue on its own — fold into the mesh/main.py docstring pass.

**NSM/mesh/refine_mesh.py:142 — Stale 'Implement this' comment on already-implemented code**  
`REPRODUCES`

Two stale TODO-style comments on working code. Delete both lines' trailing comments.

**NSM/mesh/refine_mesh.py:239 — add_vertex_if_new returns a tuple, its docstring promises an int, and its index lives in a third array's space**  
`REPRODUCES`

Two of three sub-claims hold: the Returns section describes an int and the function returns
(new_vertices, index), and `threshold` is undocumented. The third is wrong — the docstring
literally says "combined list of original mesh vertices and new_vertices", so the index space
IS stated. The whole remedy is prose.

**NSM/mesh/refine_mesh.py:278 — create_new_faces depends on an unstated midpoint ordering produced two functions away**  
`REPRODUCES`

Confirmed: a permuted midpoint list yields a different, wrong triangulation with no error. But
it can only be triggered by editing new_vertices_faces — there is no external caller. One
sentence in each docstring closes it.

**NSM/mesh/refine_mesh.py:46 — find_all_faces_to_split: docstring promises a 2-tuple, mutates its loop target, dead counters**  
`REPRODUCES`

Two of three sub-claims hold and both are prose/dead-code cleanup. The third — the one that
made this a 'defect' rather than 'rot' — I tried to reproduce and could not. Demote and fix
the docstring.

**NSM/mesh/triangle_metrics.py:1 — triangle_metrics.py has zero docstrings on every public symbol**  
`REPRODUCES` · owned by docs/SCOPE.md:205 lists triangle_metrics.py as "keep — scope under investigation".

Exactly as stated, verified by introspection rather than reading. Prose work; it should be
done in the same pass as the two triangle_metrics semantics entries (edge ordering, areas
normalisation) so the file is documented once.

**NSM/mesh/triangle_metrics.py:37 — Undocumented edge ordering shared by two modules with no cross-reference**  
`REPRODUCES`

The convention and its duplication are confirmed. It is a comment/docstring gap with no
behavioural consequence today — the two implementations agree. Fold into the triangle_metrics
docstring pass.

**NSM/mesh/triangle_metrics.py:51 — TriangleProperties.areas returns a dimensionless deviation, not areas, by default**  
`REPRODUCES` · owned by docs/SCOPE.md §2.3 condition 3 and the module ledger at docs/SCOPE.md:205 both already record this.

Exactly reproduces, including the numbers, and the 0.01 case shows a caller reading the
docstring gets 144 cells where they expected 240. Purely a prose fix (three refine_mesh
docstrings plus one on areas) — already ruled as condition 3 of the refine_mesh keep.

**NSM/models/loader.py:147 — Two contradictory deprecation messages for latent_dropout, and the shipped config triggers one**  
`REPRODUCES`

Reproduces. Neither message names a true replacement (latent_dropout was a boolean enabling
dropout on the latent portion; neither `dropout` nor `dropout_prob` is that), so both texts
are wrong. Prose fix: one message, naming what a user should actually do, in both places.

**NSM/models/modulated_periodic_activations.py:196 — ModulationNetwork concatenates in the opposite order from its docstring**  
`REPRODUCES`

Reproduces; harmless numerically but the weight columns are laid out opposite to the only
written description of them. One-line prose fix — correct the docstring to match `cat([out,
input])`, since the ordering is baked into any trained weights.

**NSM/models/modulated_periodic_activations.py:244 — Debug print left on ImplicitDecoder's forward path**  
`REPRODUCES`

Reproduces. One-line deletion — code, not prose, but it belongs in the same Phase 2 cleanup
sweep rather than a GitHub issue. Nothing depends on the output; the passing suite merely
tolerates it.

**NSM/models/triplanar.py:12 — That same block documents the plane channel order backwards**  
`REPRODUCES`

Reproduces. Prose-only fix, and the ordering is baked into every trained checkpoint so the
comment must change, not the code. Fix together with the entry below (the same block is not a
docstring).

**NSM/models/triplanar.py:219 — Assertion message states the opposite of the branch it guards**  
`REPRODUCES`

Reproduces verbatim. One-word prose fix (True -> False). Note it sits three lines above the
sum_sdf_features=False defect filed as an issue, so fix them in the same pass.

**NSM/models/triplanar.py:384 — Legacy triplanar path has a silent performance cliff on ungrouped latents**  
`REPRODUCES`

Reproduces, ordering caveat included — the entry is accurate and not overstated. Correctness
is unaffected; it is purely a documented-nowhere performance requirement. One or two lines in
the forward docstring, no issue.

**NSM/models/triplanar.py:5 — Unused imports**  
`REPRODUCES`

Reproduces; zero behavioural impact. Belongs in the Phase 2 cleanup sweep, not an issue. Worth
flagging that 'flake8 is at zero' does not mean unused imports were removed — F401 is globally
suppressed.

**NSM/models/triplanar.py:9 — triplanar.py's apparent module docstring is a no-op string literal**  
`REPRODUCES`

Reproduces. Fix together with the plane-order entry: move the literal above the imports so it
becomes a real docstring, and correct xy/xz/yz to xz/yz/xy in the same edit.

**NSM/reconstruct/main.py:1140 — reconstruct_mesh switches return type between a list and a dict based on seven unrelated flags**  
`REPRODUCES`

The switch is real and I executed both branches, but it is a deliberate convenience API, not a
defect — every first-party caller trips the dict branch and changing the return type would be
a public-surface break for unknown forks. What is missing is a Returns block saying so. Prose
fix.

**NSM/reconstruct/main.py:1299 — get_mean_errors sets register_similarity twice and its error message contains a typo**  
`REPRODUCES`

Both halves reproduce and both are cosmetic: the duplicate assignment cannot change behaviour,
and the message is a stray 'm' plus an advertised value ('diffusion') no branch accepts. Prose
fix, no issue.

**NSM/reconstruct/main.py:253 — project_latent is labelled legacy but is still the only path honoured under LBFGS-with-hard-constraint**  
`REPRODUCES`

The docstring's "Legacy" label is wrong — the function is live under ANY optimizer whenever
use_soft_norm_constraint=False, not only LBFGS as the entry's title says. It also returns None
and mutates its argument, undocumented. Both are prose corrections, not behaviour changes.
Production never sets latent_norm, so neither branch runs today.

**NSM/reconstruct/main.py:536 — pts_surface encoding is an undocumented positional contract**  
`REPRODUCES`

Reproduces exactly as stated: swapping the sdf_gt order relative to the pts_surface labels
silently produces a different fit with no error, and the function that implements the contract
has no docstring at all. The fix is prose (document the contract on reconstruct_latent) —
validating it would need a surface-name mechanism, which is the SCOPE.md §3.1 work item, not
this.

**NSM/reconstruct/main.py:826 — `mesh_to_scale` inline comment is stale since multi-surface registration landed**  
`REPRODUCES`

Executed, not inferred. Pure comment correction. Note the `decoder_to_scale` half of the entry
is fine — that one IS still an int index (`decoders[decoder_to_scale]`, :953).

**NSM/train/train_deep_sdf.py:433 — surface_accuracy curriculum is inverted relative to sample_difficulty, so schedule='constant' disables it entirely**  
`NOT_A_DEFECT`

The 'constant disables it' half reproduces. The 'inverted' headline does not: `1 -
calc_weight` is the correct direction for Curriculum-DeepSDF eq. 5, where the surface-accuracy
tolerance SHRINKS over training (the comment at lines 443-444 says so). The two features
legitimately move in opposite directions; what is missing is any record of that, since
calc_weight has no docstring. Correct the prose (docstring on calc_weight naming the
convention and what 'constant' means for each consumer); no issue.

**NSM/train/train_deep_sdf_multi_head.py:27 — CLAUDE.md still advertises train_deep_sdf_multi_head as a supported training pipeline**  
`REPRODUCES`

A live contradiction between CLAUDE.md and SCOPE.md. Prose fix only: qualify the CLAUDE.md
line. Note SCOPE.md 2.1 also records that the module's own warning text is wrong (it names
train_deep_sdf as a replacement, which is a different architecture) — that is still unfixed
and belongs in the same prose pass.

**NSM/train/utils.py:51 — get_kld's docstring describes a different computation than the code performs**  
`REPRODUCES`

The docstring describes a different estimator with a different return shape, and the code's
value swings 67x with batch size — none of which is written down. Prose fix: rewrite the
docstring to state what is computed, that it is a scalar, that Bessel's correction applies,
and that the value depends on batch size. No issue; the function is only reachable via
code_regularization_type_prior='kld_diagonal', which is not the shipped default.

**NSM/train/utils.py:87 — add_plain_lr_to_config mutates the caller's config in place while also returning it**  
`REPRODUCES`

Both halves reproduce. The behaviour is arguably fine — all four call sites use `config =
f(config)` and testing/NSM/test_lr_schedules.py:520 already deepcopies defensively — but the
docstring says nothing about mutation. Prose fix: say it mutates and returns the same object.
No issue.

**NSM/utils.py:19 — CLASSIFICATION_HEADS_GROUP_NAME is documented as a real param group but nothing ever creates one**  
`REPRODUCES`

Reproduces. The harm is prose, not behaviour: two docstrings in utils.py assert a param group
that no code path creates (utils.py:213-214 drops the hedge that utils.py:77-78 keeps). Fix is
to correct those two lines and delete the constant; no issue needed. The constant deletion
also belongs to the dead-public-symbol sweep named in class_group.

</details>


---

# 4. `SCOPE.md` rulings

****NSM/configs/deep_sdf_config:25 — A scratch notes file ships inside the package and preserves the obsolete two-positional-entry LR shape****  
`REPRODUCES`

Reproduces, but the title is wrong: NSM/configs has no __init__.py, so find_packages excludes
it and the file does NOT ship in a wheel (see the packaging entry). It is a dead 404-byte
scratch file in the source tree. That is a status ruling — SCOPE.md "dead, delete it" — not a
defect.

*Evidence:* `file NSM/configs/deep_sdf_config` -> ASCII text; `wc -c` -> 404. Lines 25-29
read: 'LearningRateSchedule': [ {}, {} ] `git log --follow` -> only 5188417 and fa33adb, i.e.
untouched since the initial NSM commit. Nothing imports or reads it …

****NSM/losses.py:82 — Three of losses.py's five public functions have never been called by anything****  
`REPRODUCES`

Reproduces and is understated (four, not three). This is a status question — are these
supported API or dead? — which is what SCOPE.md is for. SCOPE.md §1 currently rules only on
eikonal_loss and says nothing about the other four.

*Evidence:* Definitions now at losses.py:19 (eikonal_loss), :90 (compute_sdf_gradients), :168
(combined_sdf_loss), :236 (l1_loss), :241 (l2_loss). `grep -rn '\bcompute_sdf_gradients\b'
NSM/ testing/ examples/` -> only losses.py:90. Same for …

****NSM/reconstruct/utils.py:58 — compute_assd is defined but its only import is commented out****  
`REPRODUCES` · owned by #20 (for the n_samples_assd half only)

Reproduces. Two ASSD implementations, one unreachable — that is a keep-or-delete ruling for
SCOPE.md, same class as the four dead losses.py functions. The dead parameter half is already
#20's.

*Evidence:* `grep -rn 'compute_assd' --include=*.py .` -> exactly two hits:
NSM/reconstruct/recon_evaluation.py:13: `from .utils import compute_chamfer # , compute_assd`
NSM/reconstruct/utils.py:58: `def compute_assd(` ASSD is computed instead by …

**NSM/mesh/correspondence_metrics.py:224 — Two divergent implementations of the edge-ratio statistic**  
`REPRODUCES` · owned by docs/SCOPE.md §2.6 already carries an open ruling on mesh/triangle_metrics.py ('is all five of its public symbols live, or only the part correspondence_metrics uses; whether it stays a separate file or merges').

Reproduces, and the entry itself explains the divergence is deliberate (raise vs degrade).
That makes it a status question, not a defect: it is direct input to the already-open SCOPE.md
§2.6 ruling on whether triangle_metrics survives as a separate file. Route it there rather
than filing an issue; a one-line comment at triangle_health naming the deliberate divergence
would also close it.

*Evidence:* RAN corr_check.py on a 2-triangle PolyData whose second triangle has a duplicated
vertex (zero-length edge): `TriangleProperties.edge_ratio()` raises `Exception: edge length
zero! triangle with zero length edge: (array([1]),)` …

**NSM/mesh/main.py:440 — find_object_bounds_random_sampling is dead and was explicitly superseded**  
`REPRODUCES` · owned by Not ruled anywhere: `grep -n find_object_bounds_random_sampling docs/SCOPE.md docs/ARCHITECTURE.md docs/KNOWN_ISSUES.md .claude/plans/NSM_CODE_HEALTH_REFACTOR.md` returns nothing.

This is a keep-or-delete ruling on ~60 lines of superseded, non-deterministic, star-exported
code, not a defect. It belongs in SCOPE.md §2 next to the other mesh rulings. Note the stale
build/ tree is the only thing that still calls it — do not let a grep over build/ talk anyone
out of deleting it.

*Evidence:* Function now at NSM/mesh/main.py:444. `grep -rn
'find_object_bounds_random_sampling' --include=*.py --include=*.md .`: ./NSM/mesh/main.py:444
(the definition) ./docs/AUDIT_FINDINGS.md:1034 ./build/lib/NSM/mesh/main.py:326 and :474 `git
…

**NSM/mesh/refine_mesh.py:465 — subdivide_triangles_on_base_mesh assumes two meshes share cell indexing**  
`REPRODUCES` · owned by docs/SCOPE.md §2.3 condition 2 already states this verbatim, including the same line reference (:465, now :466).

The precondition is real and undocumented, but the failure I could produce is a loud
IndexError, not the silent wrong mesh the entry implies — silence needs equal cell counts with
different ordering. SCOPE.md already owns this as condition 2 of the refine_mesh ruling;
nothing further to file.

*Evidence:* Ran t_refine2.py with base = 240-cell sphere, mesh = 720-cell sphere: target cells
computed on `other`: n = 560 max index = 717 base has only 240 cells -> indices out of range:
393 RESULT: IndexError: index 241 is out of bounds for axis 0 …

**NSM/models/loader.py:228 — The 'implicit' config vocabulary is incompatible with real training configs**  
`REPRODUCES`

Reproduces, but this is a status ruling rather than a bug to fix in isolation.
docs/SCOPE.md:41-48 already rules that load_model 'advertises four model types and three of
them cannot be reconstructed' with a Phase 4 work item. The specific vocabulary split
(latent_dim/hidden_dim/num_layers vs latent_size/layer_dimensions) is a sharpening that
belongs in that SCOPE ruling, together with the sigmoid-default entry below.

*Evidence:* _get_implicit_params now at loader.py:227-263, required_keys
['latent_dim','hidden_dim','num_layers'] at :229. Ran against the shipped
NSM/configs/default_config.json: default_config has latent_size: True, latent_dim: False,
hidden_dim: …

**NSM/models/modulated_periodic_activations.py:211 — ImplicitDecoder defaults to a sigmoid output, which cannot represent a signed distance**  
`REPRODUCES`

Reproduces: the default output range is (0,1), so the decoder cannot express a negative
distance at all. Same ruling as loader:228 — this is why 'implicit' is not a usable SDF model
type, not a defect to patch in isolation. Fold both into the SCOPE.md §1 statement about the
three unusable model types.

*Evidence:* Default now at modulated_periodic_activations.py:212
(`final_activation=torch.sigmoid`). Executed:
inspect.signature(ImplicitDecoder.__init__).parameters['final_activation'].default -> <built-
in method sigmoid> dec = …

**NSM/reconstruct/main.py:1202 — tune_reconstruction is uncalled and passes a parameter get_mean_errors no longer honours**  
`REPRODUCES`

Every claim reproduces (the key count is 27, not 24, and 22 of them are absent from the
shipped default config, so no shipped config can drive it). This is a status question — dead
research entry point, plus compute_correlation_coefficient (:1481, a 4-line np.corrcoef
wrapper with no callers) — and SCOPE.md §2.6 adjudicates modules but has no ruling for these
two functions. Rule them there; do not open an issue.

*Evidence:* Now at :1246. `grep -rn 'tune_reconstruction\|compute_correlation_coefficient'
--include=*.py .` -> only the two `def` lines. Same grep over
/mnt/data/programming/kneepipeline -> zero hits. Ran scratchpad/t_tune.py: ``` config keys …

**NSM/reconstruct/main.py:588 — Only TriplanarDecoder can actually be reconstructed; the other three loader targets cannot**  
`REPRODUCES` · owned by docs/SCOPE.md §1 already rules this a Phase 4 work item ("a common decoder interface plus a registration pathway")

Reproduced by execution, and already adjudicated in SCOPE.md §1 with a named Phase 4 work item
and the same evidence. It is a design ruling that exists, not a new issue.

*Evidence:* Call sites now :593 and :673. Ran scratchpad/t_decoders.py: ``` Decoder (self,
input_, epoch=None) TwoStageDecoder (self, input, epoch=None) ImplicitDecoder (self, input_,
epoch=None) TriplanarDecoder (self, x=None, latent=None, xyz=None, …

**NSM/reconstruct/main.py:616 — In-code TODO admits the multi-surface truncation is a hack that assumes surface 0 is the bone**  
`REPRODUCES` · owned by Same class as docs/SCOPE.md §3.1's surface-ordering ruling

Reproduces exactly as the TODO admits. It is a deliberate, documented-in-code design
compromise, and it is the same positional-surface-identity problem SCOPE.md §3.1 already owns
— record it there alongside the ordering contract rather than filing it as a defect.

*Evidence:* TODO now at :621-624; the `break` at :629. Ran scratchpad/t_trunc.py — the
committed 2-surface decoder fit against a single ground-truth surface: ``` bone-only gt
against a 2-surface decoder: loss = 0.08817599713802338 verbose log: …

**NSM/reconstruct/main.py:873 — The consumer's `batch_size_latent_recon` is a no-op absorbed by **kwargs, while the real `batch_size` is left at its default**  
`REPRODUCES`

Every factual claim holds, but calling it a silent "no-op" overstates: it prints a deprecation
warning on every call, which is the intended behaviour of a deprecation shim. What is actually
wrong is that the shim is inline and undated (CLAUDE.md § "Separate permanent from
transitional at write time"). Record in SCOPE.md as deprecated-with-a-delete-when condition;
the consumer-side cleanup is a kneepipeline change, not an nsm defect.

*Evidence:* Ran scratchpad/t_bslr.py against the committed regression decoder: ``` batch_size
default: 32768 (:833) 'batch_size_latent_recon' in signature: False (commented out at :834)
deprecation warning printed: True (:906-910) latent identical …

**NSM/train/train_deep_sdf.py:422 — multi_object_overlap is a config key whose only implementation is an unconditional raise**  
`REPRODUCES`

Reproduces, but it is a status question, not a defect: the key names an unimplemented feature.
It belongs in SCOPE.md alongside the eikonal ruling — 'accepted by config, not implemented,
crashes mid-epoch if enabled' — rather than as an issue.

*Evidence:* scratchpad/e13_misc.py — set config["multi_object_overlap"]=True on an otherwise
working run: Exception: Not implemented yet | raised at line 435 (inside train_epoch, after
data loading and the first forward pass) Same construct at …


---

# 5. `KNOWN_ISSUES.md` entries

**NSM/train/train_deep_sdf.py:573 — grad_clip is applied to the model only, never to the latent codes**  
`REPRODUCES`

Reproduces exactly. Not worth an issue — clipping the latents would silently change the
numerics of every run that sets grad_clip — but a user setting a knob named grad_clip will
reasonably assume it is global. That is a durable, user-visible fact: KNOWN_ISSUES § Open, one
short entry.

*Evidence:* scratchpad/e14_misc2.py — ran a real epoch with grad_clip=1e-8, wrapping
torch.nn.utils.clip_grad_norm_ to record what it is handed: clip_grad_norm_ called 2 times,
each on {21} tensors model param tensors: 21 The latent nn.Embedding is a first-class
optimizer param group (NSM/utils.py:376-383) and …


---

# 6. Deleted

Fixed, refuted, not a defect, or already owned by an open issue. Evidence is kept for the
refuted ones until this file goes: 'we checked and it is not true' is the expensive part to
redo.


## Does not reproduce (9)

****NSM/reconstruct/cartilage_func.py:116 — compare_cart_thickness mutates the reconstructed meshes it is asked to evaluate****  
`DOES_NOT_REPRODUCE`

The claim "These are the same objects reconstruct_mesh returns in result['mesh']" is false for
the production path. The register inferred aliasing from reading the assignment and did not
check pymskt's copy semantics — the known failure mode.

*Evidence:* Built two spheres, ran compare_cart_thickness with the mesh types reconstruct_mesh
actually produces (plain `mskt.mesh.Mesh` — create_mesh_adaptive -> create_mesh ->
`mskt.mesh.Mesh(...)` at mesh/main.py:315,436): recon_bone arrays BEFORE: ['Normals']
recon_bone arrays AFTER : ['Normals'] has list_cartilage_meshes attr AFTER: False The …

****NSM/reconstruct/reconstruct_latent_S3.py:58 — reconstruct_latent_S3 is exported as public API but has never been exercised****  
`DOES_NOT_REPRODUCE` · owned by #35

"Has never been exercised" is now false — the default path runs. Status is already ruled in
docs/SCOPE.md §2.4 ("deferred research, scheduled for repair; keep the re-export") and the
live defect is #35.

*Evidence:* It runs. Executed on CPU with an 8-dim latent, a 3-layer-free Linear decoder and a
32x4 SDF tensor, num_iterations=4: returns a tuple, prints `Step: 0 Loss: 0.207...`. Of the
four defects the entry lists as proof it was never run: the NameError at :127 and the missing
wandb import are FIXED (d2ba1c7); the latent_loss_ UnboundLocalError …

**NSM/datasets/sdf_dataset.py:1046 — joint_scale_buffer is applied on the disk path and silently ignored on the in-memory path**  
`DOES_NOT_REPRODUCE` · owned by see :1061 / #22

The stated consequence — "the same dataset config produces two different normalizations" —
cannot happen: the in-memory branch raises before it computes any normalization at all. The
source asymmetry is real but unobservable, so this is an overstatement; the observable fact is
the :1061 crash and belongs there.

*Evidence:* Buffer now applied at :1119 (disk branch only). Disk path, SDFSamples with
scale_jointly=True: joint_scale_buffer=0.0 -> max_radius 13.834341 joint_scale_buffer=0.1 ->
max_radius 15.217775 (x1.1 exactly) joint_scale_buffer=0.5 -> max_radius 20.751511 (x1.5
exactly) In-memory path with the same config: joint_scale_buffer=0.0 -> KeyError: …

**NSM/datasets/sdf_dataset.py:1759 — Cache-upgrade path resaves stale pos/neg indices after removing overlapping points**  
`DOES_NOT_REPRODUCE`

The order-of-operations observation is correct but its stated consequence is wrong. The entry
dismisses test_if_idx_in_range as catching 'only out-of-range indices'; every deletion makes
the top stored index out of range, so the guard always fires. The real cost is a silently
discarded cache and a re-sample, not stale indices baked to disk.

*Evidence:* Built a real multi-surface dataset, then doctored its .npz into a 'legacy' cache:
5 rows inside both surfaces prepended, and pos/neg/surf indices recomputed over the full
1205-row array (scratchpad/e5_stale.py). Reloading with load_cache=True printed: File found in
cache: .../c_stale/Aug_20_2026/d23329a1....npz Indices out of range! …

**NSM/mesh/correspondence_metrics.py:333 — self_intersection_count's runtime guard does not guard against its actual runtime**  
`DOES_NOT_REPRODUCE`

The central claim — 'a 50k-triangle mesh under this implementation will not finish in a usable
time' — is measurably false: ~34 s for a real surface mesh. The O(n^2) worst case exists but
needs a mesh whose triangles all overlap in x, which no surface mesh does. Classic
overstatement in the direction of alarm; delete. (If anyone wants the residual fact, it is
that the guard is sized in triangles when the real cost driver is x-extent overlap — but 50k
is a defensible number as measured.)

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/corr_check2.py, timing
`self_intersection_count` on triangulated spheres: 448 tris -> 0.139 s; 1,920 -> 0.671 s;
7,936 -> 3.227 s; 19,600 -> 10.094 s; 38,640 -> 25.145 s. log-log slope = **1.16** (near-
linear, not quadratic). Extrapolated to the 50,000-triangle max_triangles default: **~34 s**.
Flat …

**NSM/mesh/main.py:667 — Multi-object adaptive meshing shares one AABB across all surfaces**  
`DOES_NOT_REPRODUCE`

The union AABB costs evaluated points (speed), not resolution — the small surface comes out
point-for-point identical. The note's stated harm is wrong, and the inline comment "Union
across objects" already says what actually happens. Overstatement in the direction of alarm;
delete.

*Evidence:* Union at NSM/mesh/main.py:671 (`coarse_sdf_flat =
torch.min(coarse_sdf_values_flat, dim=1)[0]`). The entry's stated consequence is that the
small surface "los[es] the resolution benefit". Ran t_aabb.py: a radius-0.05 sphere meshed
alone (objects=1) versus the same sphere as object 1 of a pair with a radius-0.9 sphere
(objects=2), both at …

**NSM/reconstruct/main.py:919 — Mean-mesh generation and final-mesh generation call create_mesh_adaptive with different grid parameters**  
`DOES_NOT_REPRODUCE`

Executed and disproved. The two call styles produce bit-identical geometry even with
recon_grid_origin != 1.0, so "the registration target and the reconstruction live on
inconsistent grids" is false. The only residue is the fallback branch, which is a different
AUDIT entry anchored to NSM/mesh/main.py and is latent anyway (recon_grid_origin defaults to
1.0).

*Evidence:* Mean-mesh call now :951-960 (search_bounds only); reconstruction call :1145-1161
(explicit voxel_origin and voxel_size). Ran scratchpad/t_grid.py at recon_grid_origin = 1.5,
n_pts_per_axis = 48, on the committed 2-surface decoder — calling create_mesh_adaptive both
ways: ``` surface 0: npts (2497, 3) vs (2497, 3) identical: True surface …

**NSM/train/utils.py:115 — get_profiler hardcodes a schedule that only profiles the first 8 steps and has no docstring**  
`DOES_NOT_REPRODUCE`

The load-bearing assertion — "captures epochs 3-8 and then goes inert for the rest of the run"
— is wrong, and it is exactly the read-only overstatement the register's banner warns about.
What survives is a fixed relative output directory and a missing docstring, neither worth
carrying.

*Evidence:* The central claim is false. torch.profiler.schedule defaults to repeat=0, which
means repeat FOREVER, so the cycle restarts rather than going inert. scratchpad/e7_lrutils.py,
evaluating the schedule directly: steps 0..12: WARMUP WARMUP RECORD RECORD RECORD RECORD
RECORD RECORD_AND_SAVE WARMUP WARMUP RECORD RECORD RECORD ^ cycle restarts …

**NSM/train/utils.py:4 — Unused imports and a duplicated import in train/utils.py**  
`DOES_NOT_REPRODUCE`

Half the entry is factually wrong now, and the surviving half is four dead import names that
repo policy already chose to suppress. Nothing to track; sweep whenever.

*Evidence:* grep -n '^import torch|^from torch' NSM/train/utils.py: 2:import torch 3:from
torch.profiler import profile, tensorboard_trace_handler There is NO duplicate `import torch`
— the entry's headline claim is false on current main. The unused-import half does hold.
`flake8 NSM/train/` exits 0 only because .flake8 sets `extend-ignore = ... …


## Not a defect — the entry overstates (17)

**### [stale] `.claude/plans/SIGMA_COORDINATE_IMPLEMENTATION_PLAN.md:4`**  
`NOT_A_DEFECT`

The entry itself already flags this as "aspirational rather than wrong". A plan that says it
is blocked and has not been done is doing its job. Nothing to fix.

*Evidence:* `grep -rn 'sigma_coordinate_space' --include=*.py NSM/ testing/` -> no matches, so
the plan is still unimplemented. But the file now opens with a State block: "**Updated:**
2026-08-17 · **Status:** blocked", with Next/Blocked-on lines pointing at
NSM_CODE_HEALTH_REFACTOR.md §8. All its checkboxes for the parameter work are `[ ]`, …

****NSM/losses.py:110 — losses.py builds model input as cat([latent, points]) — an undocumented latent-first ordering, the same bug class as the LR mapping****  
`NOT_A_DEFECT`

Overstated in the direction of alarm. The ordering is correct and identical to the live
trainer's; calling it "the same bug class as the LR mapping" is wrong — the LR bug was a
mismatch, this is a match. What is left is a missing docstring sentence in code nothing calls.

*Evidence:* The ordering is now at losses.py:118 (compute_sdf_gradients) and :225
(combined_sdf_loss). Checked it against the two consumers of that convention: -
TriplanarDecoder.forward legacy branch: `xyz = x[:, -3:]` / `latent = x[:, :-3:]`
(NSM/models/triplanar.py, legacy mode). - Live trainer: `inputs = torch.cat([batch_vecs,
xyz[split_idx]], …

**NSM/_lr_migration.py:55 — migration_error decides the historical LR mapping by substring-sniffing the optimizer name**  
`NOT_A_DEFECT`

The mechanism reproduces but the consequence cannot occur, which is the register's known
failure mode. Over the closed set of optimizers the library will actually build,
`'schedule_free' in name` is exactly equivalent to a membership test. A 'typo'd or future
optimizer name' has no historical run to reproduce, because get_optimizer refuses to build it.
Overstated; delete.

*Evidence:* RAN /tmp/claude-1000/.../scratchpad/lrmig_check.py. `resolve_schedule_targets` on
a Target-less config gives: Adam/AdamW/'schedual_free_AdamW'(typo)/'Lion'/''/None all ->
'entry 0 -> latent, entry 1 -> model', caution absent; only 'schedule_free_AdamW' -> 'entry 0
-> model, entry 1 -> latent' with the CAUTION block. So the substring test …

**NSM/_lr_migration.py:7 — _lr_migration.py states its own delete-when condition and it is not yet met**  
`NOT_A_DEFECT` · owned by docs/KNOWN_ISSUES.md:526 already records the module as non-permanent with its delete-when condition.

Executed, and the condition is objectively unmet — the two production model configs the
downstream consumer ships are both pre-Target, so the module is still load-bearing. But that
is the module working as designed: it carries its own delete-when condition in its header,
exactly as CLAUDE.md § 'Separate permanent from transitional' requires, and
KNOWN_ISSUES.md:526 repeats it. The entry restates the header; nothing to act on. Delete.

*Evidence:* RAN a loop over NSM/configs/*.json and
/mnt/data/programming/kneepipeline/NSM_MODELS/*/model_params_config.json:
`NSM/configs/default_config.json: targets=['model','latent']` but BOTH shipped production
model configs are pre-Target — `647_nsm_femur_v0.0.1: targets=[None, None] optimizer=AdamW`
and `551_nsm_femur_bone_v0.0.1: …

**NSM/datasets/sdf_dataset.py:1635 — n_meshes and n_pts are derived from len(list_mesh_paths[0]), which is a character count for a string path**  
`NOT_A_DEFECT`

The character-count derivation is real, but it is filed as a landmine — 'no error, plausible
number' — and the configuration that triggers it cannot construct: it raises RuntimeError
before any sample is produced. Loud, not silent. Overstated.

*Evidence:* Mechanism confirmed (scratchpad/e4_npts.py, patching SDFSamples.__init__ to a
recorder so only MultiSurfaceSDFSamples.__init__ :1718-1734 runs), flat list of two 33-char
paths: n_meshes after __init__ : 33 ; len(self.n_pts): 33 ; total_n_pts: 16500000 after
preprocess_inputs -> n_meshes: 2 ; len(n_pts): 33 ; total_n_pts: 16500000 (not …

**NSM/mesh/interpolate.py:625 — interpolate_points / interpolate_mesh call interpolate_common with 8 positional args**  
`NOT_A_DEFECT`

Nothing is wrong today — the binding is correct, both wrappers are internal, and the
hypothetical is 'someone inserts a parameter before data'. That is a maintenance hazard, not a
defect, and converting eight positionals to keywords is a two-line change anyone can make
while in the file. Delete the entry; the class name is there in case the parent files one
positional-coupling sweep.

*Evidence:* RAN interp_check.py: `inspect.signature(interpolate_common).bind(model, 'L1',
'L2', 100, 'POINTS1', 0, False, True, is_mesh=False)` -> `{'model': ..., 'latent1': 'L1',
'latent2': 'L2', 'n_steps': 100, 'data': 'POINTS1', 'surface_idx': 0, 'verbose': False,
'spherical': True}`, i.e. `points1`/`mesh` binds to `data` purely by position, …

**NSM/mesh/main.py:277 — narrow_band default flips with the use_vtk flag**  
`NOT_A_DEFECT`

The asymmetry is real but I measured its consequence and it is zero — the narrow band is a
speed optimisation that preserves the extracted surface exactly. "Silently toggles whether the
volume is cropped" is true and uninteresting. Roll the default alignment into the :280 twins
issue; there is nothing to record here.

*Evidence:* Defaults confirmed by introspection (t_main.py section D): sdf_grid_to_mesh
narrow_band default False (:278); sdf_grid_to_mesh_vtk default True (:382). But the flip is
behaviourally inert. t_nb.py, 64^3 grid, sphere of radius 0.25 in [-1,1]^3 so the crop is
substantial: use_vtk=False: nb=False n=1152 nb=True n=1152 same_pts=True …

**NSM/mesh/main.py:690 — Fallback path passes 17 positional arguments to create_mesh**  
`NOT_A_DEFECT`

Nothing is wrong today and I demonstrated it mechanically. It is a maintenance hazard of the
shape CLAUDE.md names, but there is one call site, in the same file, 40 lines from the
signature. If we want a guard it is a lint/AST assertion, not an issue with a reproduction.

*Evidence:* Call now at NSM/mesh/main.py:693. Ran t_pos.py, which AST-parses the module and
diffs the call against the signature: create_mesh call: positional args = 17 keyword args = 0
create_mesh signature (17 params): …

**NSM/mesh/main.py:711 — create_mesh_adaptive silently discards the caller's voxel_origin**  
`NOT_A_DEFECT`

Documented as fallback-only, and the value discarded in production is the default. The entry
overstates on both counts. The real risk in this area is the separate :638 entry, which I
kept.

*Evidence:* Rebinding now at NSM/mesh/main.py:727 (`samples, grid_dims, voxel_origin =
create_grid_samples_in_bounds(...)`). Ran t_adaptive.py section G on CPU with a sphere
decoder: create_mesh_adaptive(..., voxel_origin=(-1,-1,-1)) vs (...,
voxel_origin=(-500,-500,-500)) adaptive path, voxel_origin ignored? True | bounds: [-0.3 -0.3
-0.3] [-0.3 …

**NSM/mesh/main.py:862 — decode_sdf's fast path passes an unbatched latent, the legacy path an expanded one**  
`NOT_A_DEFECT`

Every sub-claim either describes the correct contract of each interface or ends in a loud
ValueError, and the per-batch inspect.signature cost is sub-millisecond over a whole
reconstruction. Textbook overstatement in the direction of alarm.

*Evidence:* Now NSM/mesh/main.py:884 / :887-889. Ran t_main.py section B with shape-recording
stub decoders: {'fast_latent': (8,), 'fast_xyz': (10, 3), 'legacy_inputs': (10, 11)} Those are
the shapes each interface requires, not a divergence: NSM/models/triplanar.py:370-376 accepts
(D,) or (1,D) and raises ValueError otherwise, while the legacy …

**NSM/mesh/refine_mesh.py:438 — Plan claims a symbol that exists nowhere, leaving refine_mesh.py orphaned**  
`NOT_A_DEFECT` · owned by docs/SCOPE.md §2.3 now rules refine_mesh.py "research, keep" on grounds that do not depend on interpolate_points_refined.

The entry's conclusion — "the only stated reason for keeping it no longer exists" — has been
superseded by an explicit SCOPE ruling that gives an independent reason. The stale sentence
lives in a completed plan, which per CLAUDE.md keeps its body as a record of what was
believed. Nothing to do.

*Evidence:* `grep -rn 'interpolate_points_refined' --include=*.py --include=*.md .` returns
only .claude/plans/completed/NSM_MESH_INTERPOLATION_IMPROVEMENTS_COMPLETED.md:146, :320, :380
— no Python anywhere. So the premise is factually right. But docs/SCOPE.md:135-148 now reads:
"Zero importers is confirmed. 'Therefore dead' is not. …

**NSM/models/two_stage.py:65 — TwoStageDecoder permanently corrupts its own module-level default dicts**  
`NOT_A_DEFECT`

The mutation is real but inert: the only four keys it writes are the same four every
subsequent construction rewrites, and no code in the repo reads the module dicts. 'A
subsequent default-constructed model silently inherits the previous model's geometry' is
exactly the alarm-direction overstatement the banner warns about. Delete.

*Evidence:* Mutation is now two_stage.py:65-68. The mutation itself reproduces: before:
default_triplanar_params latent_dim=256 n_objects=2; default_mlp_params latent_size=256
n_objects=2 after a single TwoStageDecoder(latent_size=8, n_objects=5) attempt: latent_dim=4
n_objects=5; latent_size=4 n_objects=5 (and it persists even though construction …

**NSM/reconstruct/main.py:1002 — mean_mesh is passed to the multi-object reader even when register_similarity is False**  
`NOT_A_DEFECT`

Executed and shown inert. The parameter is unread when register_to_mean_first is False, so the
two call sites differ cosmetically and produce identical output. Overstated entry.

*Evidence:* Asymmetry confirmed textually — :1026 `mean_mesh=mean_mesh if register_similarity
else None`, :1044 `mean_mesh=mean_mesh`. Ran scratchpad/t_meanmesh.py —
read_meshes_get_sampled_pts called twice with register_to_mean_first=False, once with
mean_mesh=None and once with a real sphere: ``` pts identical: True sdf identical: True icp
a,b: …

**NSM/train/train_deep_sdf.py:152 — The param-group target key is duplicated as a bare string literal in the train loop**  
`NOT_A_DEFECT`

Not a defect — a one-line style coupling with no observable failure. Filed under "Defects" it
is overstated. Fold into whatever edit next touches the resume block; nothing to track.

*Evidence:* scratchpad/e13_misc.py: constant imported into train_deep_sdf? False literal used:
['if any(group.get("target") is None for group in optimizer.param_groups):'] (line 152)
NSM.utils.PARAM_GROUP_TARGET_KEY = 'target' Behaviour is correct today: the literal and the
constant are the same string, and the 57 tests in …

**NSM/train/train_deep_sdf.py:210 — TODO in the validation block describes work the refactor should absorb**  
`NOT_A_DEFECT`

An observation about an existing TODO, not a finding. It reports no behaviour and asserts
nothing testable. Git and the TODO itself already carry it; the register adds nothing.

*Evidence:* Read at current line numbers: the TODO is at 212-214, the get_mean_errors call
spans 215-260 with six commented-out placeholder arguments (222-223, 225-226, 228, 244-245).
No execution possible — it is a comment about a duplication, and the duplication is real
(multi_head:125-149, deprecated multi_surface_orig:171-215).

**NSM/train/train_deep_sdf.py:620 — mesh_names exists in config but is never used to label anything**  
`NOT_A_DEFECT`

The entry says the names are "never used to label anything". They are: save_model_params
persists them, which is exactly the purpose CLAUDE.md gives them ("downstream consumers must
infer mesh identity from the output count, which is fragile"). Only the wandb metric names
stay positional, which is cosmetic. Overstated; its one real residue — that the persisted
names are never checked against the dataset's order — is carried by the :333 entry.

*Evidence:* scratchpad/e5_run.py — real run with mesh_names=['bone','cart']: log_dict keys
include 'l1_loss_0', 'l1_loss_1' (positional, as claimed) scratchpad/e13_misc.py — but the
names ARE consumed: mesh_names in model_params_config.json: ['bone', 'cart']

**NSM/train/utils.py:76 — The positional indexing the LR fix removed still survives in the logging helper**  
`NOT_A_DEFECT`

The entry itself concedes "It is correct today". targets.index() is an internal implementation
detail of a lookup that is keyed by Target, not a surviving positional contract. The only
substantive half — the idx_model/idx_latent override — is the :63 entry. Duplicate.

*Evidence:* scratchpad/e7_lrutils.py — both entry orders through the target-keyed path:
('model','latent') -> model_lr_initial 0.005 latent_lr_initial 0.001 ('latent','model') ->
model_lr_initial 0.005 latent_lr_initial 0.001 Correct in both. Pinned by
testing/NSM/test_lr_schedules.py::test_labels_survive_reordered_entries; all 57 tests in …


## Already fixed since the audit (19)

<details><summary>Show</summary>

**### [misleading] `CLAUDE.md:34`**  
`ALREADY_FIXED`

Fixed. NOTE for the parent: a NEW inaccuracy replaced it at CLAUDE.md:22 and :25 — those lines
advertise `make format` and `make format-check`, and neither target exists any more (`make
format` -> "No rule to make target 'format'", same for format-check). requirements-dev.txt:10
also still references `make format-check`.

**### [misleading] `CONTRIBUTING.md:102`**  
`ALREADY_FIXED`

Fixed — the Makefile was renamed to match, rather than the doc. The entry's parenthetical
"(which itself fails, see the CLAUDE.md quick-test finding)" is also stale: black is clean.

**### [stale] `CLAUDE.md:28`**  
`ALREADY_FIXED`

Fixed by edb1048. Both the code and the documentation moved.

**### [stale] `CLAUDE.md:39`**  
`ALREADY_FIXED`

Fixed. CLAUDE.md:39-40 is now true and CI enforces it.

**### [stale] `CLAUDE.md:42`**  
`ALREADY_FIXED`

Fixed. Same fix as the pyproject.toml:95 landmine entry — both should go together.

**### [stale] `docs/KNOWN_ISSUES.md:192`**  
`ALREADY_FIXED`

Fixed, and the entry says so. The failure-mode anecdote it asks to preserve (two branches each
holding half the story) belongs in the plan's Diverged section if anywhere, not in a findings
register that is itself scheduled for deletion.

****NSM/losses.py:1 — losses.py is the one file in the subsystem that fails the repo's own Black check****  
`ALREADY_FIXED`

Fixed, and the CI lint job now gates rather than continue-on-error.

****NSM/losses.py:13 — The eikonal loss is still untested, as CLAUDE.md says, while being wired into both live loss paths****  
`ALREADY_FIXED`

Fixed, and the entry already carries its own CORRECTION note saying so. The remaining status
question (is the eikonal loss supported?) is answered in docs/SCOPE.md §1 under "Genuinely
experimental" and the repair is scheduled in the code-health plan §8.2.

****NSM/reconstruct/cartilage_func.py:141 — Dead locals left behind by a commented-out KL-divergence metric****  
`ALREADY_FIXED`

The named defect (dead locals) is fixed. What is left is a commented-out block plus an import
kept alive by it — worth removing when someone next opens the file, not worth an entry.

****NSM/reconstruct/recon_evaluation.py:34 — compute_recon_loss docstring documents a parameter that no longer exists and omits three that do****  
`ALREADY_FIXED` · owned by #20

The headline claim is fixed. The dead `n_samples_assd` belongs to issue #20's sweep
(parameters accepted and never read), not to a fresh entry; the assert wording is a one-word
tidy.

****pyproject.toml:95 — pytest testpaths points at a directory that does not exist****  
`ALREADY_FIXED`

Fixed. Both halves of the entry (wrong testpaths, phantom -k filter) are gone from the file.

**NSM/datasets/sdf_dataset.py:87 — get_pts_center_and_scale ignores its center= and scale= flags (verified at runtime)**  
`ALREADY_FIXED` · owned by #20 / #21 (both closed by b0c8bf5)

The parameters no longer exist. The residual behaviour the entry flags (centering implies
scaling, no reachable centre-without-scale config) survives — `(center_pts is True) or
(norm_pts is True)` at :330 and :624 still triggers both — but that was ruled intended in #20,
not a defect.

**NSM/datasets/sdf_dataset.py:91 — get_pts_center_and_scale mutates the caller's array in place, undocumented**  
`ALREADY_FIXED` · owned by #21 (closed by b0c8bf5)

No longer mutates. Pinned by test_dataset_cache.py:776-780.

**NSM/datasets/sdf_dataset.py:989 — os.sched_setaffinity is called unguarded on a platform-conditional API that is guarded 55 lines earlier**  
`ALREADY_FIXED`

Fixed before this pass. Listed in the prompt's already-fixed set and confirmed in the source.

**NSM/mesh/main.py:169 — Dead local and formatting drift against the project's own stated standard**  
`ALREADY_FIXED`

Every checkable claim in this entry is now false. Fixed by edb1048.

**NSM/models/deep_sdf.py:241 — NameError disguised as an error path in progressive_layer**  
`ALREADY_FIXED`

The headline claim is fixed. The surviving off-by-one (epoch == start applies the layer at
full weight for one epoch before the warmup ramp starts from ~0) is a one-line rider that
belongs on the progressive_add_depth issue filed for deep_sdf:171, not a separate entry — the
feature crashes for the first 1009 epochs anyway.

**NSM/reconstruct/main.py:420 — `latent_input` is computed and never used**  
`ALREADY_FIXED`

Removed by edb1048. Nothing left to fix.

**NSM/train/train_deep_sdf.py:510 — Enabling eikonal loss silently doubles the forward-pass cost and is untested**  
`ALREADY_FIXED`

The behaviour the entry describes can no longer happen — enabling it now raises at the top of
train_deep_sdf. One knock-on: docs/SCOPE.md:58 still says the eikonal loss is "Wired into both
live loss paths" and "needs a loud warning at the point of use"; it now raises instead. That
SCOPE line is stale and should be corrected in the same prose pass as the multi_head CLAUDE.md
line.

**NSM/train/utils.py:41 — cyclic_anneal_linear computes an unused `cycle` local**  
`ALREADY_FIXED`

The headline is fixed. The docstring residue is too thin to file; fold it into any prose pass
that touches train/utils.py.

</details>


## Reproduces, but an open issue already owns it (57)

<details><summary>Show</summary>

**## Document-level verdicts**  
`REPRODUCES`

The table is a rollup of the 25 entries above it and carries no fact those entries do not. Per
the register's own delete-when condition, it goes with them — after the individual
FIX_IN_PHASE_2 prose corrections land, there is nothing for it to summarise.

****NSM/__init__.py:1 — NSM/__init__.py leaks `os` into the public namespace for the sake of commented-out code****  
`REPRODUCES`

Reproduces exactly. But it is a two-line cleanup that belongs to the per-subpackage __all__
work docs/SCOPE.md §3.3 already scopes and explains, not a separate register entry.

****NSM/configs/default_config.json:1 — No shipped config can construct a triplanar model faithfully****  
`REPRODUCES`

Factually correct and executed, but it is already ruled on in docs/SCOPE.md §1 as a Phase 4
work item ("ship a default config per model type, derived from the ShapeMedKnee configs").
Keeping it here duplicates a doc that already owns it.

****NSM/configs/default_config.json:1 — Nothing in the library ever loads default_config.json****  
`REPRODUCES`

Reproduces, and it is an observation rather than a defect — docs/SCOPE.md §2.6 already rules
generate_sdf_default_config.py "supported: it owns the shipped default_config.json and is
pinned by test_default_config_sync.py", which says the same thing.

****NSM/configs/generate_sdf_default_config.py:1 — NSM/configs is not a package and will not ship in a built distribution****  
`REPRODUCES`

Reproduces exactly, but docs/SCOPE.md §5 already records it verbatim ("NSM.configs will not
ship in a built distribution... works today only because installs are editable"). Already
homed.

****NSM/mesh/__init__.py:1 — Package __init__ star-exports main.py's third-party imports and hides four modules****  
`REPRODUCES`

Reproduces, but docs/ARCHITECTURE.md §5 counts and names this surface and docs/SCOPE.md
§3.2/§3.3 carries the recommendation. Already homed.

****NSM/models/__init__.py:1 — Public API surface is polluted by a wildcard import****  
`REPRODUCES`

Reproduces, already counted and named in docs/ARCHITECTURE.md §5 and scoped in SCOPE.md §3.2.
Duplicate.

****NSM/reconstruct/__init__.py:1 — Star-import __init__ files re-export third-party modules as part of the package API****  
`REPRODUCES`

Reproduces with slightly different numbers, which is itself the argument against keeping hand-
transcribed counts in a doc (CLAUDE.md rule 1: a number is computed or it is not committed).
ARCHITECTURE.md §5 and SCOPE.md §3.2 own this.

****NSM/reconstruct/__init__.py:1 — `from .main import *` with no __all__ leaks the entire main.py import namespace onto the package****  
`REPRODUCES`

Reproduces, but it is already recorded in docs/ARCHITECTURE.md §5 ("The star-import surface",
138 de-facto exports) and ruled on in docs/SCOPE.md §3.2/§3.3, which recommends per-subpackage
__all__. Duplicate of a doc that owns it.

****NSM/reconstruct/cartilage_func.py:50 — cartilage_func's mesh slicing is a hardcoded positional layout with no validation****  
`REPRODUCES`

The structure reproduces, but "silently produces wrong numbers" does not — the whole-joint
variant raises before returning anything. docs/SCOPE.md §2.5 already rules the module
Production-and-clunky. Not worth an issue on the strength of a crash that is already loud.

****NSM/reconstruct/utils.py:104 — Two different `adjust_learning_rate` functions in the same package; the reconstruct one shadows via star-import****  
`REPRODUCES`

Duplicate entry for the same finding, and the finding is already in ARCHITECTURE.md §6 and
SCOPE.md §3.2. Delete both.

****NSM/reconstruct/utils.py:104 — Two unrelated functions named adjust_learning_rate; the reconstruct one is re-exported from the NSM.reconstruct package namespace****  
`REPRODUCES`

Reproduces, but docs/ARCHITECTURE.md §6 has it as a named row ("Two adjust_learning_rate") and
docs/SCOPE.md §3.2 cites it as the motivating example for __all__. Already homed twice.

****docs/KNOWN_ISSUES.md:183 — Open action recorded in the LR post-mortem that the refactor plan should absorb****  
`REPRODUCES`

Reproduces only as a pointer. It records that a document has an open action — which the
document already says, in the place a reader will find it. Nothing to fix here; the entry is a
stale line number wrapped around a sentence KNOWN_ISSUES.md owns.

****testing/testing_h5_vs_np_loading/save_and_load_h5_vs_np.py:74 — unpack_pts/unpack_numpy_data are duplicated verbatim in a testing script****  
`REPRODUCES`

Reproduces and "verbatim" is now half-false. But the file is an excluded scratch benchmark
(.flake8 excludes testing/testing_h5_vs_np_loading), its line 1 is a commented salloc
invocation, and nothing imports it. Not worth an issue; delete the entry and the drift will be
handled if Phase 4 touches the caching seam.

**NSM/datasets/sdf_dataset.py:1021 — norm_and_scale_all_meshes reads every cache file twice from disk**  
`REPRODUCES` · owned by #2 (adjacent — 'SDFSamples - slow loading')

Measured and reproduces exactly 2x. But it is a one-time startup cost on the
scale_jointly=True path only, and #2 is already the open home for SDFSamples loading cost. Not
worth its own issue; add the measurement to #2 if that is ever worked.

**NSM/datasets/sdf_dataset.py:1061 — ISSUE #1 REFUTED AS STATED, AND INVERTED: norm_and_scale_all_meshes works on disk and crashes in memory**  
`REPRODUCES` · owned by adjacent to #22 (different crash site, not covered by #22's fix statement); refutes legacy #1 · fold into #22; close #1 with a pointer

Live hard crash on a documented option combination, and it is NOT what #22 tracks: #22 is an
UnboundLocalError in __getitem__ over the four timing keys, and its stated fix (deciding the
batch-key contract) would leave this KeyError untouched. It also settles open issue #1, whose
premise is exactly backwards. "Fixed" means scale_jointly=True + store_data_in_memory=True
constructs, and #1 closes.

**NSM/datasets/sdf_dataset.py:1097 — ISSUE #3 CONFIRMED: sigma_near/sigma_far change coordinate space with scale_jointly**  
`REPRODUCES` · owned by #3

Confirms #3, which is already open and is the canonical home for it. Nothing new to file.

**NSM/datasets/sdf_dataset.py:1131 — save_data_to_cache serializes three keys that nothing ever produces**  
`REPRODUCES`

Reproduces (three permanently dead entries plus a commented-out alternative at :1216), but it
is six lines of inert list with no user-visible effect. No fixable statement beyond deleting
them.

**NSM/datasets/sdf_dataset.py:1146 — Cache key names are renamed on write and triple-guessed on read**  
`REPRODUCES`

Factually accurate but it is a description, not a defect — the fallbacks work and nothing
misbehaves. No statement of what 'fixed' would mean that is not just 'rewrite the cache
format', which #19 already has to do for its own reasons.

**NSM/datasets/sdf_dataset.py:1310 — find_hash returns the first match anywhere under loc_save, across all date folders**  
`REPRODUCES` · owned by #19

The behaviour is exactly as described, but it is the design: the md5 is the cache identity and
the date folder only organises writes, so serving a match from any date is intended. The harm
the entry attributes to it ('the reuse window is wide and silent') is entirely the incomplete
key, which is #19. Nothing to fix here that #19 does not already own.

**NSM/datasets/sdf_dataset.py:1396 — uniform_pts_buffer and subsample also affect cached content but are not hashed**  
`REPRODUCES` · owned by #19 (a)

Still true, but already owned by issue #19 (a), annotated in the source and pinned by strict
xfail tests. Keeping a duplicate copy in AUDIT_FINDINGS adds nothing.

**NSM/datasets/sdf_dataset.py:1406 — reference_mesh is hashed by str(), so passing a Mesh object makes the cache key its memory address**  
`REPRODUCES` · owned by #19 (c)

Every factual claim survives, but it is issue #19 (c) verbatim, is annotated in both
get_hash_params docstrings, and is pinned by TestReferenceMeshHashing. Duplicate.

**NSM/datasets/sdf_dataset.py:169 — `mean` parameter is documented and accepted by both sampling functions but never used**  
`REPRODUCES` · owned by #20

Reproduces, and #20 explicitly says 'the first task is the enumeration, not the fix'. This is
that enumeration's result for this file, and `mean` is not in #20's known-instances table — it
should be added there rather than filed as its own issue.

**NSM/datasets/sdf_dataset.py:1726 — get_sample_data_dict writes an unconditional append-only log into the cache root**  
`REPRODUCES` · fold into #22

All three claims reproduce: unconditional (not behind verbose), never truncated, and a hard
construction failure in the save_cache=False configuration because a debug log is written to a
directory only the caching path creates. Fixed = the log is behind verbose (or gone), and
nothing on the load path depends on loc_save existing. Same configuration family as #22 — file
it there or alongside it.

**NSM/datasets/sdf_dataset.py:184 — n_pts_random is silently swallowed by **kwargs in both read_*_get_sampled_pts functions**  
`REPRODUCES` · owned by #16 (and the #20 class sweep)

Still true and still exactly as described, but #16 already owns it with the same evidence, and
#20 owns the class. Filing again would duplicate.

**NSM/datasets/sdf_dataset.py:1973 — mesh_to_scale is not part of the multi-surface cache hash**  
`REPRODUCES` · owned by #19 (a)

Reproduces, but it is the headline instance of issue #19 (a), already annotated in the source
and pinned by an xfail. Duplicate.

**NSM/datasets/sdf_dataset.py:2158 — MultiSurfaceSDFSamples.__getitem__ raises UnboundLocalError when store_data_in_memory=True (verified)**  
`REPRODUCES` · owned by #22

Reproduces, but is issue #22 word for word, annotated in the source and pinned by
test_dataset_cache.TestConfigurationsThatDoNotRun. Duplicate.

**NSM/datasets/sdf_dataset.py:314 — read_mesh_get_sampled_pts returns 'xyz' or 'pts' depending on get_random; one consumer reads 'pts' unconditionally**  
`REPRODUCES` · owned by #15

Reproduces exactly, including the crash line. #15 already owns it by title.

**NSM/datasets/sdf_dataset.py:316 — pts_surface return type differs between the single- and multi-mesh functions**  
`REPRODUCES`

Reproduces, but no in-repo consumer takes .shape of the single-mesh value, so there is no
defect today — it is one more face of the sampler-divergence class (with :308 and :314). Fold
the observation into that class if it is fixed; do not file it.

**NSM/datasets/sdf_dataset.py:665 — include_surf_in_pts in the multi-mesh path concatenates a leaked loop variable**  
`REPRODUCES` · owned by #17

Reproduces precisely as described. #17 already owns it by title.

**NSM/datasets/sdf_dataset.py:671 — pts_surface concatenation raises ValueError when any surface is allocated zero points**  
`REPRODUCES` · owned by #23

Reproduces, but it is a second crash site for #23's exact fixable statement ("a zero-count
sampling combo samples nothing instead of raising"). #23's pinned test only exercises the
uniform case, so the :739 site is worth adding as evidence to #23 rather than filed
separately.

**NSM/datasets/sdf_dataset.py:820 — loc_save default is evaluated at import time, so setting LOC_SDF_CACHE later has no effect**  
`REPRODUCES` · owned by #24

Reproduces. #24 owns it and is pinned by TestCacheLocationDefault; the docstring at :848-853
now warns about it too, so the documentation half is already done.

**NSM/datasets/sdf_dataset.py:832 — `multiprocessing` is simultaneously a module, a constructor parameter, and an instance attribute**  
`REPRODUCES`

Reproduces as a naming collision, but every current use resolves correctly — the shadowing is
confined to __init__, and :1058 is a different scope. Readability, not a defect, and no
fixable statement beyond renaming.

**NSM/datasets/sdf_dataset.py:849 — check_probabilities type gate rejects integer probabilities**  
`REPRODUCES` · owned by adjacent to #23, but not covered by it · fold into #23

Not covered by #23 — #23's pinned test uses the list form [0.0, 0.0], which never reaches this
gate, and #23's stated failure (inside point_cloud_utils) is not where a scalar int fails. A
whole-number probability from a JSON config dies at construction with a message that is
factually wrong about what it accepts. "Fixed" means the gate accepts any real number
(excluding bool), so the value reaches the same handling a float gets.

**NSM/mesh/interpolate.py:98 — Three divergent hand-rolled decoder invocation conventions in one subsystem**  
`REPRODUCES`

The divergence is real and I demonstrated its consequence, but the consequence is
hypothetical: no decoder in NSM/ is fast-interface-only, and none is planned. `decode_sdf`'s
dispatch is a performance shortcut, not a compatibility requirement. The only live residue is
that interpolate never takes the fast path on the production TriplanarDecoder — a perf note,
not the 'would break half the subsystem' the entry claims. Overstated; delete.

**NSM/mesh/main.py:171 — scale_mesh silently overrides the caller's scale and offset**  
`REPRODUCES` · owned by #20 — this is a new instance for #20's enumeration table, not a separate issue. #20 explicitly says "The first task is the enumeration, not the fix" and the mechanical check it names (does the parameter name appear in the body before rebinding) catches this one. · fold into #20

Same shape as get_pts_center_and_scale, which #20 already owns: a named parameter is rebound
before it is read, silently. Fixed means the two are reconciled (raise, or drop the
parameters) and the instance is recorded in #20's table.

**NSM/models/deep_sdf.py:27 — Sine.__init__ is misspelled and never runs**  
`REPRODUCES`

Reproduces and is harmless, and the fact is already in ARCHITECTURE.md's name-trap table.
Nothing to promote.

**NSM/models/deep_sdf.py:47 — xyz_in_all is accepted and documented but never used**  
`REPRODUCES` · owned by #20

Reproduces, but already an Open KNOWN_ISSUES entry pointing at #20's class sweep. Duplicate.

**NSM/models/deep_sdf.py:87 — latent_noise_sigma is stored and never read**  
`REPRODUCES` · owned by #20

Reproduces, but it is one more instance of #20's class. #20's body says explicitly 'The
instances below are what we tripped over, not what exists. The check is mechanical' — the
sweep will find this. Do not file separately; add it to #20's instance table if you want it
named.

**NSM/models/loader.py:119 — Decoder output column -> surface identity is nowhere in the models package**  
`REPRODUCES`

True but already promoted into SCOPE.md, which is where a design/status fact belongs. Keeping
it in AUDIT_FINDINGS duplicates a ruling that already exists.

**NSM/models/modulated_periodic_activations.py:43 — Two different Sine classes in one package, with incompatible defaults**  
`REPRODUCES`

Reproduces exactly as described, but ARCHITECTURE.md §6 already owns this trap, and the
__init__ has been annotated with a pointer to it. Duplicate.

**NSM/models/triplanar.py:197 — The consumer's hand-rolled param mapping omits padding**  
`REPRODUCES` · owned by #26

Tracked in four places already (issue #26, KNOWN_ISSUES entry, SCOPE ruling, inline code
comment) with a pinning test. Nothing left in the AUDIT_FINDINGS copy.

**NSM/models/triplanar.py:312 — normalize_coordinates ignores its own padding parameter**  
`REPRODUCES` · owned by #20

Reproduces, and is the single most thoroughly tracked item in my slice — issue #20, a
KNOWN_ISSUES entry, an inline code comment, and a pinning test. Duplicate. (The `10e-6`
literal noted in the entry's last sentence is also still there at :337; that is cosmetic and
not worth carrying.)

**NSM/models/triplanar.py:87 — VAEDecoder builds no activation functions between its conv layers**  
`REPRODUCES`

Reproduces, but the full corrected analysis already lives in ARCHITECTURE.md §7.1 including
the trains-nonlinear/evaluates-affine hazard. Nothing here to promote. The UnboundLocalError
rider is an instance of the constructible-but-uncallable class issue and should be listed
there.

**NSM/models/triplanar.py:99 — VAEDecoder registers every submodule twice, doubling checkpoint keys**  
`REPRODUCES` · owned by #27

Fully tracked three ways already: GitHub #27, a KNOWN_ISSUES.md entry with a pinning test, and
an inline code comment. The AUDIT_FINDINGS copy is redundant.

**NSM/reconstruct/main.py:1118 — Undocumented mesh-index ordering is the load-bearing contract between reconstruct_mesh and its consumer**  
`REPRODUCES` · owned by Already ruled in docs/SCOPE.md §3.1 and pinned by testing/NSM/regression/test_reconstruction_regression.py::TestSurfaceOrderContract

True, but it is verbatim what docs/SCOPE.md §3.1 already rules ("reconstruct_mesh's result
mesh list is ordered, and the order is the contract", naming mesh_names and the same LR-bug
analogy), and TestSurfaceOrderContract already pins it geometrically. Duplicating it into an
issue adds nothing.

**NSM/reconstruct/main.py:1167 — time_calc_recon_loss is measured and thrown away while return_timing claims to report timings**  
`REPRODUCES` · owned by #29 — adjacent (both are "the result dict is missing what the caller asked for"); fix them together · fold into #29

Reproduced by execution. One line to add. Small enough that it should ride along with the #29
result-dict fix rather than becoming its own issue.

**NSM/reconstruct/main.py:393 — In hybrid-optimizer mode the LR decay interval is computed from the wrong iteration count**  
`REPRODUCES`

The mechanism is real but the entry's stated trigger is WRONG: `any lbfgs_iterations > 0` does
NOT break it — I measured an identical schedule. It breaks only when adam_iterations is set
explicitly and differs from num_iterations, in which case the requested decays never happen at
all. Fixed = derive adjust_lr_every from adam_iterations. Low priority (hybrid_optimizer is
off everywhere and has no test), and the duplicated if/else at :506-521 is a separate
simplification — the two arms are equivalent because current_optimizer is bound in both modes.

**NSM/reconstruct/main.py:588 — reconstruct_latent calls decoders with keyword-only interface that only TriplanarDecoder implements**  
`REPRODUCES` · owned by Duplicate of the entry immediately above it in AUDIT_FINDINGS.md (same anchor line :588); both already covered by docs/SCOPE.md §1

Verbatim duplicate of the preceding entry — same file, same line, same finding, written twice
under two titles. Delete this one; keep the SCOPE ruling.

**NSM/reconstruct/main.py:960 — Missing f-string prefix silently collapses per-mesh EMD results to one literal key**  
`REPRODUCES` · owned by #29 — same early-return block (which already carries a `KNOWN DEFECT, #29` comment at :982). Fold in as an extra bullet rather than filing separately. · fold into #29

Reproduced by execution. One missing `f`. It sits inside the exact block #29 owns and #29's
own statement ("drops keys the caller asked for") already covers it, so it belongs on #29, not
in a new issue.

**NSM/train/train_deep_sdf.py:279 — train_epoch accepts return_loss and verbose parameters that are never read**  
`REPRODUCES` · owned by #20 — "Sweep: parameters accepted and never read (read the traps before fixing)"

Confirmed, but #20 is explicitly the class sweep for exactly this and says "the instances
below are what we tripped over, not what exists". These belong in that enumeration, not as a
separate register entry.

**NSM/train/train_deep_sdf.py:281 — train_epoch's n_surfaces default of 2 contradicts train_deep_sdf's objects_per_decoder default of 1**  
`REPRODUCES`

The default mismatch is real, but the entry's stated consequence — "silently takes the multi-
surface branch" — is wrong: it raises a loud IndexError at line 429. It is also not an
exported symbol. Overstated on both counts, and the loud failure means nothing silent is at
stake.

**NSM/train/train_deep_sdf.py:575 — train_epoch hard-requires optional data-loading telemetry keys from the dataset**  
`REPRODUCES` · owned by #22 — "store_data_in_memory=True raises, and its workaround cannot train"

Reproduces exactly, but issue #22 already states this defect including the fact that the
workaround cannot reach training and quotes train_deep_sdf.py:578-581. Already tracked;
nothing to add.

**NSM/train/train_deep_sdf.py:84 — Dead duplicate of the resume_epoch default**  
`REPRODUCES`

Reproduces, but it is two unreachable lines with no behavioural consequence. Not worth an
issue; delete it in any pass that touches the function.

**NSM/train/train_deep_sdf_multi_head.py:85 — train_deep_sdf_multi_head builds the optimizer from a leaked loop variable — only the last decoder is trained**  
`REPRODUCES` · owned by docs/KNOWN_ISSUES.md §2 "train_deep_sdf_multi_head optimizes only the last model"; docs/SCOPE.md §2.1 ruling

Reproduces exactly, but it is already a KNOWN_ISSUES entry AND a SCOPE ruling AND a
DeprecationWarning in the module. Nothing left for the register to carry.

**NSM/utils.py:26 — LearningRateSchedule base class returns None instead of raising**  
`REPRODUCES`

The mechanism reproduces, but there is no reachable path in the repo or in any shipped config
that triggers it — it needs a user-written subclass that forgets to override. `raise
NotImplementedError` is a genuinely free one-line hardening to make while touching the file,
but it does not clear the 'worth fixing' bar as a standalone issue.

**NSM/utils.py:394 — get_optimizer silently drops weight_decay for the default 'Adam' optimizer**  
`REPRODUCES` · fold into #20) + KNOWN_ISSUES § History required on fix

Reproduces, and the shipped default config hits it: every default Adam run silently trains
with zero weight decay while the config says 1e-4. Note for whoever fixes it — this changes
training output for inputs that previously ran without error, so per CLAUDE.md § Numerical-
behaviour changes the fix needs a KNOWN_ISSUES.md § History entry. 'Fixed' = pass weight_decay
to Adam (with the History note), or delete the parameter and the config key.

</details>

