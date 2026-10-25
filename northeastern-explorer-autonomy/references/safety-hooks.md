# The safety layer — PreToolUse hooks + per-job verifiers

Autonomy is only safe if the **submit** action is hard to get wrong. This layer makes
the verified path the only path: a hook intercepts the `sbatch` *before* it lands and
runs a per-job verifier that refuses to let a broken experiment burn GPU-hours.

Two complementary mechanisms:
1. **PreToolUse hooks** — intercept a tool call before it executes; block on failure.
2. **Per-job verifiers** — `verify_<job>.sh` scripts the hook dispatches, each
   enforcing experiment-specific invariants.

## PreToolUse hook wiring

In `.claude/settings.json`, declare hooks keyed by the tool they match. Each hook is a
command that receives the tool call as JSON on stdin and exits **0 to allow** / **2 to
block** (a blocking exit prints stderr so you see the reason inline):

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash",
        "hooks": [{ "type": "command",
          "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/pretooluse_hpc_submit_verify.sh",
          "timeout": 180 }] },
      { "matcher": "Edit|Write",
        "hooks": [{ "type": "command",
          "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/pretooluse_artifact_lint.sh",
          "timeout": 30 }] }
    ]
  }
}
```

The submit hook parses the command (stdlib Python, no `jq`), checks whether it's an
`sbatch <file>`, derives the job basename, looks up `scripts/verifiers/verify_<basename>.sh`,
and runs it. If the verifier fails, the hook blocks the submission. A **grace-period**
pattern (allow with a WARN until a cutoff date, then hard-block) lets you roll out a new
verifier without breaking in-flight work.

**Two layers, by design:** the PreToolUse hook catches the submit *before* it lands
(Claude sees the failure immediately); a git **pre-commit** hook is the backstop for
anything that bypasses the tool path. Never `--no-verify` past either.

### Distinguish safety hooks from discipline hooks
- **Safety** (mandatory for autonomy): the `sbatch` verifier — it's what stops a bad job
  from spending real money.
- **Discipline** (optional): artifact/consistency linters that keep your research
  records in sync. Valuable, but a new project can adopt the safety hook alone and add
  discipline hooks later.

## The per-job verifier pattern

Each experiment gets a hand-written `verify_<job>.sh`. Hand-written, because *what's
worth checking* needs human judgment — but they share a small helper library so they
stay short. The reusable `_lib.sh`:

```bash
# verify_outputs_differ FILE1 FILE2 MSG   -> 0 if byte-different, 1 if identical/missing
# verify_python_assertion 'CODE' MSG       -> exit code of `python -c CODE`
# verify_file_nonempty FILE MSG            -> 0 if exists and non-empty
# log_verifier_event NAME STATUS DETAILS   -> append to docs/VERIFIER_LOG.md (audit trail)
```

### The general checklist every verifier enforces
Synthesized across real verifiers — adapt the domain-specific ones:

1. **Existence** — the referenced Python script and sbatch file both exist.
2. **Static structure** — required flags present (`--smoke`), required pre-registration
   constant present (G2), required env in the sbatch (`HF_HUB_OFFLINE=1` and the inherited
   proxy unset — the offline rule, see `cluster-setup.md` §4). Give each job a dry-run
   mode and run it: `DRY_RUN=1 bash job.sbatch` prints every resolved setting and path
   and exits before modules or GPU. Inside the job, cheap pre-GPU guards (each input
   line carries the pre-registered settings; refuse to overwrite an existing output;
   refuse to write into another experiment's folder) turn a wasted GPU slot into an
   exit 2 within seconds (agent design under the exit-code contract, `autonomy-model.md`).
3. **Smoke integrity** — `--smoke` runs without crashing, and produces a *discriminating*
   signal (e.g. a trained direction's norm > 0, an AUC that differs from a shuffled
   baseline). A smoke test that passes trivially isn't a gate. A smoke also gates only
   what it exercises: cover production **shapes** and **every expensive path** (the
   largest model at the real n and length, every loader kind). Field-observed: a
   small-rank smoke adapter hid an IndexError that the production-rank adapter hit 73 s
   into the full job. Run each path as its own process, record name, rc and seconds per path, and keep
   going after a failure so one fix round covers every failure. Set the full jobs'
   `--time` from the per-call seconds.
4. **End-to-end wiring check** — for any with-intervention-vs-baseline
   experiment, assert the two outputs **differ**. This is the highest-value check: it
   catches the "hook was registered but never actually invoked" bug, where outputs are
   identical and a no-op masquerades as a real (null) result. A 4.5h H200 bake-off was
   lost to exactly this before the check existed.
5. **Norm-matched controls (where applicable)** — if you use an inverse/random control,
   verify its vector norm is within ~10% of the real perturbation, so a "control
   perturbs more" result isn't just norm-shock.
6. **The scheduler accepts it.** `sbatch --test-only <same flags> <file>` succeeds. This
   catches a wrong account, partition, gres type or time before a slot or a dependency
   chain is spent. Do not use its start estimate to choose a GPU type
   (`cluster-setup.md` §5.1 step 5). It is unverified whether it rejects a request over
   the per-job GPU cap, so check that cap (§5) yourself. Field-observed: `--test-only`
   is also refused with `QOSMaxSubmitJobPerUserLimit` whenever your submit slots are
   full (every queued job counts: array tasks, dependency-held links, other projects
   under your user; `cluster-setup.md` §5), so a verifier that requires it reports NOT
   READY on a job the scheduler accepts a minute later. Treat that one refusal as
   **retry when a slot frees**, not a failed check: block the submit, but do not count it
   as a verifier failure. Before passing a chain or a wrapper, count the slots it will
   use, nested `sbatch` calls included.
7. **A GPU job requests a GPU.** If job files leave `--gres` out so the type is chosen at
   submit time, the submit flags must carry it. Field-observed: a one-off diagnostic
   sbatch without it ran on a CPU node, failed in about a minute (`No devices were
   found`) and counted as agent code. Refuse any GPU job whose header and submit flags
   together request no GPU, and give scratch and diagnostic scripts their
   own `#SBATCH --gres` default (a command-line `--gres` overrides it). The template's
   driver gate exits 2 with `missing --gres?` when no GPU is allocated (and 4 when a GPU
   is allocated but its driver is unreadable).

Checks 6 and 7 together, for a GPU job (`SB` = the job file, `SB_FLAGS` = the exact
flags of the real submission; verifier exit 0 = pass, 1 = fail, 3 = retry later, not the
job exit codes of `autonomy-model.md`). It accepts `--gres=X`, `--gres X` and `--gpus*`,
refuses a leftover `<type>` placeholder and a zero count (`--gres=gpu:0`,
`--gres=gpu:<type>:0`, `--gpus=0`), and runs under the laptop's bash 3.2:

```bash
# Last request wins: header lines first, then the submit flags (bash 3.2 safe).
REQ=$(grep -E '^#SBATCH[[:space:]]' "$SB" | sed 's/[[:space:]]#.*//' |
  grep -oE -- '--(gres|gpus[a-z-]*)[=[:space:]]+[^[:space:]]+' | tail -n 1 | sed -E 's/[[:space:]]+/=/') || true
prev=
for f in ${SB_FLAGS[@]+"${SB_FLAGS[@]}"}; do
  case $prev in *=*) ;; --gres|--gpus|--gpus-per-*) REQ="$prev=$f" ;; esac
  case $f in --gres=*|--gpus=*|--gpus-per-*=*) REQ=$f ;; esac
  prev=$f
done
case $REQ in
  *'<'*) echo "FAIL: placeholder in $REQ" >&2; exit 1 ;;
  --gres=gpu*:0) echo "FAIL: zero GPUs requested ($REQ)" >&2; exit 1 ;;
  --gres=gpu*|--gpus*=[1-9]*|--gpus*=*:[1-9]*) ;;
  *) echo "FAIL: no GPU request (--gres=gpu:<type>:N or --gpus) in the file or the flags (would run on a CPU node)" >&2; exit 1 ;;
esac
if OUT=$(sbatch --test-only ${SB_FLAGS[@]+"${SB_FLAGS[@]}"} "$SB" 2>&1); then
  echo "PASS: scheduler accepts it ($REQ)"
elif grep -q QOSMaxSubmitJobPerUserLimit <<<"$OUT"; then
  echo "RETRY: submit slots full; re-run when one frees (not a failed check)" >&2; exit 3
else
  echo "FAIL: scheduler refuses it: $OUT" >&2; exit 1
fi
```

### Degrade gracefully, never block on missing infrastructure
A verifier run on a dev box (no GPU, no smoke fixture) should fall back to **static
checks only** and PASS with a WARN — never FAIL for lack of a fixture. In a short CPU
job on the cluster (fixture cached; never on the login node, which kills heavy work
silently), it runs the full smoke with numerical assertions. The rule:
always provide *some* coverage; never make the absence of infrastructure a hard stop.

**Run the test suite on a cluster CPU node before the smokes**; laptop passes do not
transfer. Field-observed:
- Cluster CPUs pick different BLAS kernels, so `torch.equal` and 1e-5 tolerances that
  passed locally failed there. Use `allclose` with realistic tolerances, and declare one
  platform of record for any 1e-9-level reproduction check (thresholds differed across
  platforms by up to 3.7e-9).
- Tolerances written on CPU fp32 fail in bf16 on the GPU (relative errors of about 0.015
  against a 1e-3 bound). Calibrate GPU-side assertions to the run's dtype, e.g. a looser
  relative tolerance plus a cosine floor of 0.99 (reruns reached 0.9999).
- On Linux, numpy/OpenBLAS starts worker threads at import, and they can receive process
  signals. Signal tests that blocked a signal only in the main thread after import passed
  on macOS and failed on the cluster: block signals before any import.

## Why this layer is what makes autonomy sustainable

The tier model says *what* may be automated; this layer makes the automated thing
*safe to get wrong*. Combined with the ledgers (`autonomy-model.md`):

- a broken experiment is caught by its **verifier** before it submits;
- a runaway spend is caught by the **budget** guardrail (G4);
- a bug-loop is caught by the **failure-streak** rollback (G5);
- a silent no-op is caught by the **end-to-end wiring check**.

Each defends a different failure mode. Together they're why a session can fire jobs
unattended without it being reckless. Start by adopting just the submit verifier + the
two ledgers; add the discipline linters once the safety path is trusted.

## Adapting for a non-LLM / different project
- The offline-mode check is HPC-LLM-specific — drop it if your jobs don't pull
  from a model hub, or replace it with your own "inputs are present and local" check.
- The smoke-fixture path should be configurable (an env var), not hard-coded.
- Consider a small `verifiers.yaml` registry (`{job, checks: [...]}`) so new jobs
  declare what to verify instead of each re-implementing bash — the existing
  hand-written approach is the main thing worth improving for scale.
