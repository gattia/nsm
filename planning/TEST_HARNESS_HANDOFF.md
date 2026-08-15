# Handoff: build the numerical regression harness (Phase 3 §7.1)

**For whoever picks up the testing work.** Written 2026-08-15 against tag `v0.1.0`,
branch `plan-code-health-refactor`.

You do not need to read the audit first. Read §1 and §3 of this document, then start.

---

## 1. Where things stand

An audit of NSM ran through Phases 0 and 1 of
`.claude/plans/NSM_CODE_HEALTH_REFACTOR.md`. Three documents came out of it:

| Document | What it is | How to use it |
|---|---|---|
| `docs/SCOPE.md` | What NSM is for; per-module status rulings; the public API contract | Read once, at the start |
| `docs/ARCHITECTURE.md` | Dependency graph, module ledger, defect classes | Read once, at the start |
| `docs/AUDIT_FINDINGS.md` | 216 findings with `file:line` | **Look things up in it. Do not read it through.** |

**The critical caveat about the findings register:** 178 of its 216 entries are
*inference from reading*, not executed checks. Two have since been tested and **both were
wrong, both overstated**. Treat any individual entry as a hypothesis with a line number
attached, not a fact. It is a map of where to look, not a list of things to fix.

Current baseline: **11,861 lines, 34% coverage, 159 tests + 1 skip, ~14s.**

---

## 2. Your job in one sentence

Build an end-to-end numerical regression harness that fails when NSM's training or
reconstruction output changes, so that the Phase 4 decomposition can proceed without
silently altering results.

This is the single highest-value artifact in the plan. Everything after it depends on it.

---

## 3. The spec

A tiny synthetic dataset — 2–3 analytic meshes (spheres/ellipsoids from `pyvista`), 8
epochs, CPU, fixed seed — run end to end, asserted against stored baselines.

### 3.1 Training

- [ ] Loss trajectory across all 8 epochs
- [ ] Final latent norms
- [ ] **Per-param-group learning rate at each epoch.** This is the one that matters most:
      it is the assertion that would have caught the bug that started all of this. There
      is already a good model for it in `testing/NSM/test_lr_schedules.py` — read it
      before writing yours.

### 3.2 Reconstruction

- [ ] Fitted latent, output mesh vertex positions, surface metrics
- [ ] **Go through `reconstruct_mesh`, not just `reconstruct_latent`.** `reconstruct_mesh`
      is what the downstream consumer actually calls, and it currently has **one executed
      line in the entire test suite** — its `def`. Call it the way
      `kneepipeline/steps/run_nsm.py:183-211` does: a *list* of mesh paths, all arguments
      by name.
- [ ] **Assert the returned `mesh` list order.** Index 0 = bone, index 1 = cartilage is a
      load-bearing contract that nothing in the signature, docstring, or result dict
      states, and the consumer hardcodes it.

### 3.3 The dataset cache

The layer with the worst findings, and the one an in-memory harness never touches.
`sdf_dataset.py` is 2,195 lines at 7% coverage.

- [ ] Build cache → reload → assert samples identical
- [ ] Assert that changing a **hashed** parameter changes the cache key
- [ ] Then check the unhashed ones. `mesh_to_scale`, `uniform_pts_buffer` and `subsample`
      all change cached content and are reportedly absent from `get_hash_params`, so two
      runs differing only in those reuse each other's data. **This is inference — verify
      it before treating it as a bug.** If it is real it is the highest-severity finding
      in the register, because it silently produces wrong training data.

### 3.4 Model save/load round-trip

- [ ] Train → save → load → assert **bitwise-identical forward output**.
      `testing/NSM/models/test_loader.py:232` currently loads a saved model and never
      compares its output to the original's, so a wrong-but-same-shaped forward passes
      every existing assertion.

### 3.5 Constraints

- [ ] Tolerances tight enough to catch a real change, loose enough to survive platform
      float noise. Store baselines as versioned artifacts.
- [ ] **Under 2 minutes**, in CI, on every PR. If it gets slower than that people will
      skip it and the whole exercise is wasted.
- [ ] CPU. Then, separately and opt-in: a GPU test asserting the seed-ordering constraint
      the consumer depends on — `torch.manual_seed` **after** `.cuda()`, not before (see
      `docs/KNOWN_ISSUES_HISTORY.md`). State plainly in the harness that CPU baselines do
      not bound GPU divergence.

---

## 4. What not to do

- **Do not work through the findings register.** There is deliberately no separate
  verification pass. Findings get confirmed or killed as a by-product of building these
  tests, so the effort leaves permanent tests behind instead of throwaway scripts.
- **Do not chase coverage.** The plan's "32% → ≥70%" is a lagging indicator, not a gate.
  The gate is: this harness exists, it is green, and it runs in CI.
- **Do not fix bugs you find.** Write the characterization test that pins the *current*
  behaviour, bugs included, and open an issue. A silent fix during test-writing defeats
  the purpose — you would be changing the baseline you are trying to establish. If a
  behaviour is genuinely wrong and worth fixing, it needs a
  `docs/KNOWN_ISSUES_HISTORY.md` entry and its own commit.
- **Do not reformat.** `black --check` fails on 9 pre-existing files and `make lint`
  reports 445 flake8 violations. That cleanup is its own PR; mixing it in makes your diff
  unreviewable.

---

## 5. Environment and commands

```bash
# working env (python 3.9, all deps present)
/mnt/data/conda-envs/nsm-dev/bin/python

# run the suite
make test                      # = pytest testing/ -v
pytest testing/ -q             # 159 passed, 1 skipped, ~14s

# coverage
pytest testing/ --cov=NSM --cov-report=term-missing
```

Tests live in `testing/`, **not** `tests/`. Note `pyproject.toml` sets
`testpaths = ["tests"]`, a directory that does not exist — a bare `pytest` works only
because pytest 8 warns and falls back to scanning from the root. Worth fixing while you
are in there; it is a one-line change.

Real production configs to test against, 131 keys each:

```
/mnt/data/programming/kneepipeline/NSM_MODELS/647_nsm_femur_v0.0.1/model_params_config.json   # bone+cart
/mnt/data/programming/kneepipeline/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model_params_config.json  # bone only
```

The shipped `NSM/configs/default_config.json` has only 61 keys and is DeepSDF-shaped —
do not use it to build a triplanar model; it silently falls back to a different
architecture.

---

## 6. Things that will bite you while building this

Each of these was found during the audit and will cost you an hour if you meet it cold.

1. **Importing NSM prints to stdout.** `utils.py` emits `schedulefree not found` on any
   `import NSM.*`. Harmless, but do not let it into an assertion on captured output.
2. **`import NSM.reconstruct` reconfigures the root logger** for your whole process
   (`reconstruct/main.py:26`, module-scope `logging.basicConfig`). If your test logging
   behaves strangely, that is why.
3. **`from NSM.reconstruct import adjust_learning_rate` gives you the wrong function.**
   There are two with that name and unrelated signatures; the star-import in the package
   `__init__` leaks `reconstruct/utils.py`'s copy over `NSM/utils.py`'s. Import
   explicitly from the module you mean.
4. **`bare import NSM` does not give you `NSM.models`.** `NSM/__init__.py` imports only
   `utils`. Use `from NSM.models import ...`.
5. **VTK saves mesh points as float32.** A save/load round-trip loses ~5e-9 per point, so
   any mesh-vertex baseline needs a tolerance above that, not exact equality.
6. **`eikonal_weight > 0` now raises `NotImplementedError`** by design. If you are writing
   a training test, leave it at 0. See plan §8.2.
7. **`pymskt.fix_mesh("pcu")` is non-deterministic** on cartilage meshes — point counts
   vary between runs. Do not put it inside the harness's deterministic path.
8. **`train/deprecated/` has no `__init__.py`**, so its 880 lines never appear in coverage
   output. Do not be surprised when the denominator does not match the line count.

---

## 7. Definition of done

- [ ] The harness exists, is green, and runs in CI on every PR
- [ ] It runs in under 2 minutes
- [ ] Deliberately break something — swap two learning-rate schedules, perturb a mesh
      vertex — and confirm the harness goes red. **A regression harness nobody has seen
      fail is not evidence of anything.**
- [ ] Baselines are stored as versioned artifacts with a documented regeneration command
- [ ] Any finding you confirmed or killed along the way is annotated in
      `docs/AUDIT_FINDINGS.md` the way the two existing corrections are — in place, with
      the original claim left visible

Once §7.1 is green, §7.2 (contract tests on `TriplanarDecoder`, `reconstruct_mesh`, and
historical-checkpoint compatibility) is the next bounded piece, and Phase 4 becomes safe
to start.

---

## 8. Open questions you may hit, and who decides

- **`nsosim` usage is unsurveyed.** It is not on this machine. It blocks the physical
  quarantine of `train/deprecated/`, nothing else. Ask the maintainer.
- **Config restructure vs. rename-and-validate** is undecided; see plan §8.1. Do not
  design around either outcome — the harness should assert behaviour, not config shape.
- **The decoder registry** (plan §8.1) will change how models are constructed. Write the
  harness against `load_model` where you can rather than hand-rolling constructor
  arguments, so it survives that change.
